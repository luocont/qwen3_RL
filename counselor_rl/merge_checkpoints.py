"""
merge_checkpoints.py

将 output_dir 下的多个 checkpoint 权重平均（Model Soup），
合并到 base model 上，生成一个新的合并模型。

支持两种模式：
  1. soup   : 对选中的 checkpoint 权重取平均（默认）
  2. best   : 直接复制奖励最高的 checkpoint（需要 best_samples.jsonl）

用法示例：
  python merge_checkpoints.py --output_dir ./output --base_model /path/to/base
  python merge_checkpoints.py --output_dir ./output --base_model /path/to/base --mode best
  python merge_checkpoints.py --output_dir ./output --base_model /path/to/base --checkpoints checkpoint-5 checkpoint-10 checkpoint-15
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_checkpoints(output_dir: str) -> list[Path]:
    output_path = Path(output_dir)
    ckpts = sorted(
        [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    return ckpts


def find_best_checkpoint(output_dir: str, best_samples_path: str) -> Path | None:
    samples_path = Path(best_samples_path)
    if not samples_path.exists():
        print(f"[警告] best_samples 文件不存在: {samples_path}")
        return None

    step_rewards: dict[int, list[float]] = {}
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                step = sample.get("step", -1)
                reward = sample.get("best_reward", 0.0)
                step_rewards.setdefault(step, []).append(reward)
            except json.JSONDecodeError:
                continue

    if not step_rewards:
        return None

    best_step = max(step_rewards, key=lambda s: sum(step_rewards[s]) / len(step_rewards[s]))
    best_avg = sum(step_rewards[best_step]) / len(step_rewards[best_step])
    print(f"[最优 checkpoint] step={best_step}  平均奖励={best_avg:.4f}")

    ckpt_path = Path(output_dir) / f"checkpoint-{best_step}"
    if ckpt_path.exists():
        return ckpt_path

    print(f"[警告] checkpoint-{best_step} 目录不存在，回退到最新 checkpoint")
    return None


def soup_merge(
    checkpoints: list[Path],
    base_model_path: str,
    save_path: str,
    device: str = "cpu",
):
    print(f"\n[Model Soup] 合并 {len(checkpoints)} 个 checkpoint：")
    for c in checkpoints:
        print(f"  {c.name}")

    print(f"\n加载第一个 checkpoint 作为基础: {checkpoints[0]}")
    merged_model = AutoModelForCausalLM.from_pretrained(
        str(checkpoints[0]),
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=device,
    )
    merged_state = {k: v.clone().float() for k, v in merged_model.state_dict().items()}

    for ckpt in checkpoints[1:]:
        print(f"  累加权重: {ckpt.name}")
        ckpt_model = AutoModelForCausalLM.from_pretrained(
            str(ckpt),
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map=device,
        )
        ckpt_state = ckpt_model.state_dict()
        for k in merged_state:
            merged_state[k] += ckpt_state[k].float()
        del ckpt_model
        torch.cuda.empty_cache() if device != "cpu" else None

    n = len(checkpoints)
    print(f"\n  取平均（除以 {n}）...")
    for k in merged_state:
        merged_state[k] /= n

    merged_model.load_state_dict(merged_state)

    print(f"\n保存合并模型到: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    merged_model.save_pretrained(save_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoints[-1]),
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(save_path)
    print(f"[完成] 合并模型已保存到 {save_path}")


def best_copy(best_ckpt: Path, save_path: str):
    print(f"\n[Best Checkpoint] 复制 {best_ckpt.name} → {save_path}")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    shutil.copytree(str(best_ckpt), save_path)
    print(f"[完成] 最优模型已复制到 {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./output", help="训练输出目录（包含 checkpoint-* 子目录）")
    parser.add_argument("--base_model", default=None, help="原始 base model 路径（仅用于参考，soup 模式不需要）")
    parser.add_argument("--save_path", default=None, help="合并后模型保存路径，默认为 output_dir/merged_model")
    parser.add_argument(
        "--mode",
        choices=["soup", "best"],
        default="soup",
        help="soup: 权重平均；best: 选最优 checkpoint",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="指定要合并的 checkpoint 目录名（如 checkpoint-5 checkpoint-10），不指定则使用全部",
    )
    parser.add_argument(
        "--last_n",
        type=int,
        default=None,
        help="只取最后 N 个 checkpoint 做 soup（与 --checkpoints 互斥）",
    )
    parser.add_argument(
        "--best_samples_path",
        default=None,
        help="best_samples.jsonl 路径，用于 best 模式定位最优 checkpoint",
    )
    parser.add_argument("--device", default="cpu", help="加载权重的设备（cpu / cuda）")
    args = parser.parse_args()

    save_path = args.save_path or os.path.join(args.output_dir, "merged_model")

    all_ckpts = find_checkpoints(args.output_dir)
    if not all_ckpts:
        print(f"[错误] 在 {args.output_dir} 下未找到任何 checkpoint-* 目录")
        return

    print(f"发现 {len(all_ckpts)} 个 checkpoint：{[c.name for c in all_ckpts]}")

    if args.mode == "best":
        best_samples = args.best_samples_path or os.path.join(args.output_dir, "best_samples.jsonl")
        best_ckpt = find_best_checkpoint(args.output_dir, best_samples)
        if best_ckpt is None:
            best_ckpt = all_ckpts[-1]
            print(f"[回退] 使用最新 checkpoint: {best_ckpt.name}")
        best_copy(best_ckpt, save_path)
        return

    if args.checkpoints:
        selected = []
        for name in args.checkpoints:
            p = Path(args.output_dir) / name
            if p.exists():
                selected.append(p)
            else:
                print(f"[警告] 指定的 checkpoint 不存在，跳过: {p}")
        if not selected:
            print("[错误] 没有有效的 checkpoint 可合并")
            return
    elif args.last_n:
        selected = all_ckpts[-args.last_n:]
        print(f"使用最后 {args.last_n} 个 checkpoint")
    else:
        selected = all_ckpts

    soup_merge(selected, args.base_model, save_path, device=args.device)


if __name__ == "__main__":
    main()
