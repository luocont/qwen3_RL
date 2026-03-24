# -*- coding: utf-8 -*-
"""
counselor_rl/rl_train.py

基于 GRPO (TRL) 的心理咨询师强化学习训练脚本。
被训练模型：Qwen3（不开启思考模式，enable_thinking=False）

强化学习流程（对应 mermaid 图）：
  输入：上下文 Context + 暂存态度评估(PANAS)
  生成：智能体生成参考 → 模型采样 × group_size
  评估：PANAS 态度评定 × group_size + 格式评定 × group_size
  奖励：患者改善奖励 + 格式奖励 → 总奖励
  更新：GRPO 更新参数 + 保存最优样本
  输出：更新上下文 + 更新暂存评估
"""
import os
import re
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque

os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig, get_peft_model
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trl import GRPOTrainer, GRPOConfig

from agents.panas_agent import PANASAgent
from agents.counselor_agent import CounselorAgent
from agents.prompts import TRAINED_MODEL_INFERENCE_SYSTEM_PROMPT

_THINK_RE = re.compile(r"<thinking>([\s\S]*?)</thinking>", re.MULTILINE)
_5PS_RE = re.compile(r"<5Ps>([\s\S]*?)</5Ps>", re.MULTILINE)
_REPLY_AFTER_5PS_RE = re.compile(r"</5Ps>\s*([\s\S]+)$", re.MULTILINE)


# ─────────────────────────────────────────────
# Reward Tracker
# ─────────────────────────────────────────────
class RewardsTracker:
    def __init__(self, output_dir: str, is_main_process: bool = True, plot_every_n_steps: int = 10):
        self.is_main_process = is_main_process
        self.plot_every_n_steps = plot_every_n_steps
        self._reset_accum()
        self.format_history: deque = deque(maxlen=1000)
        self.panas_history: deque = deque(maxlen=1000)
        self.total_history: deque = deque(maxlen=1000)
        self.step_history: deque = deque(maxlen=1000)
        self.plots_dir = os.path.join(output_dir, "reward_plots")
        if self.is_main_process:
            os.makedirs(self.plots_dir, exist_ok=True)

    def _reset_accum(self):
        self.sum_format = 0.0
        self.sum_panas = 0.0
        self.sum_total = 0.0
        self.count = 0

    def add(self, r_format: float, r_panas: float, r_total: float):
        self.sum_format += float(r_format)
        self.sum_panas += float(r_panas)
        self.sum_total += float(r_total)
        self.count += 1

    def finalize_step(self, current_step: int):
        if self.count == 0:
            return None
        mf = self.sum_format / self.count
        mp = self.sum_panas / self.count
        mt = self.sum_total / self.count
        self._reset_accum()
        self.format_history.append(mf)
        self.panas_history.append(mp)
        self.total_history.append(mt)
        self.step_history.append(current_step)
        return mf, mp, mt

    def plot(self, current_step: int):
        if not self.is_main_process or len(self.step_history) <= 1:
            return
        try:
            fig, ax = plt.subplots(figsize=(12, 5))
            steps = list(self.step_history)
            ax.plot(steps, list(self.format_history), "g-", linewidth=2, label="R_format (mean)")
            ax.plot(steps, list(self.panas_history), "b-", linewidth=2, label="R_panas (mean)")
            ax.plot(steps, list(self.total_history), "r-", linewidth=2, label="R_total (mean)")
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Reward (mean)")
            ax.set_title(f"Reward Components (Step {current_step})")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            path = os.path.join(self.plots_dir, f"reward_step_{current_step}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            if wandb.run is not None:
                wandb.log({"reward_plot": wandb.Image(path), "custom/step": current_step})
        except Exception as e:
            print(f"[绘图错误] {e}")


# ─────────────────────────────────────────────
# KL Monitor Callback
# ─────────────────────────────────────────────
class KLMonitorCallback(TrainerCallback):
    def __init__(self, output_dir: str, is_main_process: bool = True, plot_every_n_steps: int = 10):
        self.output_dir = output_dir
        self.is_main_process = is_main_process
        self.plot_every_n_steps = plot_every_n_steps
        self.kl_history: deque = deque(maxlen=1000)
        self.step_history: deque = deque(maxlen=1000)
        self.plots_dir = os.path.join(output_dir, "kl_plots")
        if self.is_main_process:
            os.makedirs(self.plots_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.is_main_process or logs is None:
            return
        kl = logs.get("objective/kl") or logs.get("kl") or logs.get("train/kl")
        if kl is None:
            return
        step = state.global_step
        self.kl_history.append(kl)
        self.step_history.append(step)
        print(f"[KL] step={step} kl={kl:.6f}")
        if wandb.run is not None:
            wandb.log({"custom/kl": kl, "custom/step": step})
        if step % self.plot_every_n_steps == 0 and len(self.kl_history) > 1:
            self._plot(step)

    def _plot(self, step: int):
        try:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(list(self.step_history), list(self.kl_history), "b-", linewidth=2)
            ax.set_xlabel("Step")
            ax.set_ylabel("KL Divergence")
            ax.set_title(f"KL Divergence (Step {step})")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path = os.path.join(self.plots_dir, f"kl_step_{step}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            if wandb.run is not None:
                wandb.log({"kl_plot": wandb.Image(path), "custom/step": step})
        except Exception as e:
            print(f"[KL绘图错误] {e}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.is_main_process and len(self.kl_history) > 0:
            self._plot(state.global_step)


# ─────────────────────────────────────────────
# Reward Monitor Callback
# ─────────────────────────────────────────────
class RewardMonitorCallback(TrainerCallback):
    def __init__(self, tracker: RewardsTracker, is_main_process: bool = True, plot_every_n_steps: int = 10):
        self.tracker = tracker
        self.is_main_process = is_main_process
        self.plot_every_n_steps = plot_every_n_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.is_main_process:
            return
        step = state.global_step
        res = self.tracker.finalize_step(step)
        if res is None:
            return
        mf, mp, mt = res
        print(f"[奖励均值] step={step} R_format={mf:+.4f} R_panas={mp:+.4f} R_total={mt:+.4f}")
        if wandb.run is not None:
            wandb.log({"custom/r_format_mean": mf, "custom/r_panas_mean": mp, "custom/r_total_mean": mt, "custom/step": step})
        if step % self.plot_every_n_steps == 0:
            self.tracker.plot(step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.is_main_process:
            self.tracker.plot(state.global_step)


# ─────────────────────────────────────────────
# Rolling Dialogue Callback
# ─────────────────────────────────────────────
class RollingDialogueCallback(TrainerCallback):
    """
    滚动多轮对话回调：
    每个 step 结束后，将最优样本的输出追加到对话历史，
    并更新 dataset 中的 prompts（包含新的上下文 + 最新 PANAS 评估）。
    对应 mermaid 图中的「更新上下文」和「更新暂存评估」。
    """
    def __init__(self, dialogue_manager: "GlobalDialogueHistory", tokenizer, is_main_process: bool = True, group_size: int = 4):
        self.dialogue_manager = dialogue_manager
        self.tokenizer = tokenizer
        self.is_main_process = is_main_process
        self.group_size = group_size
        self.trainer = None

    def on_train_begin(self, args, state, control, **kwargs):
        if "trainer" in kwargs:
            self.trainer = kwargs["trainer"]

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        if not self.is_main_process:
            return
        pending = getattr(self.dialogue_manager, "pending_updates", {})
        if not pending:
            return
        for prompt_idx, best_completion in list(pending.items()):
            self.dialogue_manager.add_best_completion(prompt_idx, best_completion)
        self.dialogue_manager.pending_updates.clear()
        self._update_dataset()

    def _update_dataset(self):
        if self.trainer is None:
            return
        num = len(self.dialogue_manager.all_user_messages)
        new_prompts = [
            self.dialogue_manager.format_prompt_with_history(i, self.tokenizer)
            for i in range(num)
        ]
        self.trainer.train_dataset = Dataset.from_list([{"prompt": p} for p in new_prompts])
        if self.is_main_process:
            print(f"[滚动对话] Dataset 已更新，共 {len(new_prompts)} 条 prompts")


# ─────────────────────────────────────────────
# Global Dialogue History
# ─────────────────────────────────────────────
class GlobalDialogueHistory:
    """
    管理每个 prompt 的多轮对话历史（上下文 Context）和暂存 PANAS 评估。
    对应 mermaid 图中的「上下文 Context」和「暂存态度评估」。
    """
    def __init__(self, all_user_messages: list[list[dict]], all_5ps_cases: list[str], is_main_process: bool = True):
        self.is_main_process = is_main_process
        self.dialogue_history: dict[int, list[dict]] = {}
        self.all_user_messages: dict[int, list[dict]] = {}
        self.current_rounds: dict[int, int] = {}
        self.pending_updates: dict[int, str] = {}
        self.panas_cache: dict[int, dict | None] = {}
        self.case_cache: dict[int, str] = {}
        self._init(all_user_messages, all_5ps_cases)

    def _init(self, all_user_messages: list[list[dict]], all_5ps_cases: list[str]):
        for idx, user_msgs in enumerate(all_user_messages):
            self.all_user_messages[idx] = user_msgs
            self.current_rounds[idx] = 0
            self.dialogue_history[idx] = [user_msgs[0]] if user_msgs else []
            self.panas_cache[idx] = None
            self.case_cache[idx] = all_5ps_cases[idx] if idx < len(all_5ps_cases) else ""

    def get_history(self, prompt_idx: int) -> list[dict]:
        return self.dialogue_history.get(prompt_idx, []).copy()

    def get_panas(self, prompt_idx: int) -> dict | None:
        return self.panas_cache.get(prompt_idx)

    def get_case(self, prompt_idx: int) -> str:
        return self.case_cache.get(prompt_idx, "")

    def add_best_completion(self, prompt_idx: int, best_completion: str):
        if prompt_idx not in self.dialogue_history:
            return
        self.dialogue_history[prompt_idx].append({"role": "assistant", "content": best_completion})
        self.current_rounds[prompt_idx] += 1
        next_round = self.current_rounds[prompt_idx]
        user_msgs = self.all_user_messages.get(prompt_idx, [])
        if next_round < len(user_msgs):
            self.dialogue_history[prompt_idx].append(user_msgs[next_round])

    def update_panas(self, prompt_idx: int, panas: dict):
        self.panas_cache[prompt_idx] = panas

    def update_case(self, prompt_idx: int, case: str):
        self.case_cache[prompt_idx] = case

    def queue_best_completion(self, prompt_idx: int, best_completion: str):
        self.pending_updates[prompt_idx] = best_completion

    def format_prompt_with_history(self, prompt_idx: int, tokenizer) -> str:
        history = self.get_history(prompt_idx)
        case = self.get_case(prompt_idx)
        if not history:
            return ""
        system_content = TRAINED_MODEL_INFERENCE_SYSTEM_PROMPT.format(previous_5ps_case=case)
        full = [{"role": "system", "content": system_content}] + history
        if tokenizer is not None and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            try:
                return tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                print(f"[警告] apply_chat_template 失败: {e}")
        prompt = ""
        for m in full:
            prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt


# ─────────────────────────────────────────────
# Reward Computer
# ─────────────────────────────────────────────
class RewardComputer:
    """
    计算每个生成样本的奖励：
    - R_format：格式奖励（<thinking>、<5Ps>、reply 三段结构）
    - R_panas：患者改善奖励（基于 PANAS 评估的 PA↑ NA↓）
    - R_total = R_format + R_panas
    """
    def __init__(
        self,
        panas_agent: PANASAgent,
        is_main_process: bool = True,
    ):
        self.panas_agent = panas_agent
        self.is_main_process = is_main_process

    def _compute_r_format(self, sample: str) -> tuple[float, str, str, str]:
        think_m = _THINK_RE.search(sample)
        ps_m = _5PS_RE.search(sample)
        reply_m = _REPLY_AFTER_5PS_RE.search(sample)

        if not think_m or not ps_m or not reply_m:
            return 0.0, "", "", ""

        think = think_m.group(1).strip()
        ps = ps_m.group(1).strip()
        reply = reply_m.group(1).strip()

        if not reply:
            return 0.0, think, ps, reply

        return 0.1, think, ps, reply

    def _compute_r_panas(
        self,
        dialogue_turn: str,
        prev_panas: dict | None,
    ) -> tuple[float, dict]:
        try:
            curr_panas = self.panas_agent.evaluate(dialogue_turn)
        except Exception as e:
            if self.is_main_process:
                print(f"[PANAS评估错误] {e}")
            curr_panas = {
                "positive_affect": {"average": 3.0},
                "negative_affect": {"average": 3.0},
            }
        r_panas = PANASAgent.compute_improvement_reward(prev_panas, curr_panas)
        return r_panas, curr_panas

    def compute(
        self,
        sample: str,
        dialogue_turn: str,
        prev_panas: dict | None,
    ) -> tuple[float, float, float, dict]:
        r_format, think, ps, reply = self._compute_r_format(sample)

        if r_format <= 0.0:
            return 0.0, 0.0, 0.0, prev_panas or {}

        r_panas, curr_panas = self._compute_r_panas(dialogue_turn, prev_panas)
        r_total = r_format + r_panas

        if self.is_main_process:
            print(f"[奖励] R_format={r_format:.4f} R_panas={r_panas:+.4f} R_total={r_total:+.4f}")

        return r_total, r_format, r_panas, curr_panas


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def load_rl_data(path: str) -> tuple[list[list[dict]], list[str]]:
    print(f"[加载数据] {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_user_messages: list[list[dict]] = []
    all_5ps_cases: list[str] = []

    for item in data:
        messages = item.get("messages", [])
        user_msgs = [
            {"role": "user", "content": m["content"].strip()}
            for m in messages
            if m.get("role") == "user" and m.get("content", "").strip()
        ]
        if not user_msgs:
            continue
        all_user_messages.append(user_msgs)
        ps_history = item.get("5ps_history", [])
        all_5ps_cases.append(ps_history[0] if ps_history else "")

    print(f"[加载数据] 共 {len(all_user_messages)} 个会话")
    return all_user_messages, all_5ps_cases


# ─────────────────────────────────────────────
# Reward function builder
# ─────────────────────────────────────────────
def build_reward_fn(
    reward_computer: RewardComputer,
    dialogue_manager: GlobalDialogueHistory,
    tracker: RewardsTracker,
    group_size: int,
    is_main_process: bool = True,
):
    """
    构建 TRL GRPO 的 reward_fn。
    对应 mermaid 图中的完整 RL 循环：
      采样(模型采样×4) → PANAS评定×4 + 格式评定×4
      → 患者改善奖励 + 格式奖励 → 总奖励
      → GRPO更新参数 + 保存最优样本
      → 更新上下文 + 更新暂存评估
    """
    def reward_fn(*args, **kwargs):
        if kwargs:
            prompts = kwargs.get("prompts", [])
            completions = kwargs.get("completions", [])
        elif len(args) >= 2:
            prompts, completions = args[0], args[1]
        else:
            raise ValueError(f"Unexpected reward_fn args: {args}, {kwargs}")

        is_distributed = dist.is_available() and dist.is_initialized()
        current_rank = dist.get_rank() if is_distributed else 0

        if not isinstance(completions, list):
            completions = list(completions)
        if not isinstance(prompts, list):
            prompts = list(prompts)

        if current_rank == 0:
            rewards = [0.0] * len(completions)
            total_comps = len(completions)
            idx = 0

            while idx < total_comps:
                group_indices = list(range(idx, min(idx + group_size, total_comps)))
                prompt_idx = idx // group_size
                prompt_idx = min(prompt_idx, len(prompts) - 1)

                prev_panas = dialogue_manager.get_panas(prompt_idx)
                case = dialogue_manager.get_case(prompt_idx)

                if is_main_process:
                    print(f"\n{'#'*60}")
                    print(f"[Prompt #{prompt_idx+1}] 开始评估 {len(group_indices)} 个生成样本")
                    print(f"{'#'*60}")

                group_results = []

                def _eval_one(j: int):
                    sample = str(completions[j])
                    history = dialogue_manager.get_history(prompt_idx)
                    last_user = ""
                    for m in reversed(history):
                        if m["role"] == "user":
                            last_user = m["content"]
                            break
                    dialogue_turn = f"来访者: {last_user}\n咨询师: {sample}"
                    return j, reward_computer.compute(sample, dialogue_turn, prev_panas)

                with ThreadPoolExecutor(max_workers=min(group_size, len(group_indices))) as ex:
                    futures = {ex.submit(_eval_one, j): j for j in group_indices}
                    for fut in futures:
                        j, (r_total, r_format, r_panas, curr_panas) = fut.result()
                        rewards[j] = r_total
                        group_results.append((j, r_total, str(completions[j]), r_format, r_panas, curr_panas))
                        if tracker is not None:
                            tracker.add(r_format, r_panas, r_total)

                if group_results:
                    best = max(group_results, key=lambda x: x[1])
                    best_j, best_reward, best_completion, _, _, best_panas = best

                    if is_main_process:
                        print(f"[最优样本] Prompt #{prompt_idx+1} 最优奖励={best_reward:+.4f}")

                    dialogue_manager.queue_best_completion(prompt_idx, best_completion)
                    dialogue_manager.update_panas(prompt_idx, best_panas)

                    ps_m = _5PS_RE.search(best_completion)
                    if ps_m:
                        dialogue_manager.update_case(prompt_idx, ps_m.group(1).strip())

                idx = group_indices[-1] + 1
        else:
            rewards = [0.0] * len(completions)

        if is_distributed:
            obj = [rewards]
            dist.broadcast_object_list(obj, src=0)
            rewards = obj[0]

        return rewards

    return reward_fn


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl_json", type=str, default="./data/rl_data.json")
    parser.add_argument("--model_name", type=str, default="/path/to/qwen3-checkpoint")
    parser.add_argument("--output_dir", type=str, default="./outputs/grpo_counselor")

    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    parser.add_argument("--panas_api_base", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--panas_api_key", type=str, default=os.environ.get("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--panas_model", type=str, default="qwen-plus")

    parser.add_argument("--kl_coef", type=float, default=0.03)
    parser.add_argument("--kl_plot_every_n_steps", type=int, default=10)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--fsdp", type=str, default=None)
    parser.add_argument("--ddp_find_unused_parameters", action="store_true")

    parser.add_argument("--wandb_project", type=str, default="grpo-counselor-rl")

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        args.local_rank = local_rank
        is_main_process = local_rank == 0
    else:
        is_main_process = True

    def print0(*msg):
        if is_main_process:
            print(*msg)

    print0("=" * 60)
    print0("[counselor_rl] GRPO 强化学习训练启动")
    print0(f"模型: {args.model_name}")
    print0(f"输出: {args.output_dir}")
    print0("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'user' %}"
        "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}"
        "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )

    special_tokens_to_add = []
    vocab = tokenizer.get_vocab()
    if "<|im_start|>" not in vocab:
        special_tokens_to_add.extend(["<|im_start|>", "<|im_end|>"])
    if special_tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
        print0(f"[Tokenizer] 添加特殊 token: {special_tokens_to_add}")

    all_user_messages, all_5ps_cases = load_rl_data(args.rl_json)

    dialogue_manager = GlobalDialogueHistory(
        all_user_messages=all_user_messages,
        all_5ps_cases=all_5ps_cases,
        is_main_process=is_main_process,
    )

    formatted_prompts = [
        dialogue_manager.format_prompt_with_history(i, tokenizer)
        for i in range(len(all_user_messages))
    ]
    ds = Dataset.from_list([{"prompt": p} for p in formatted_prompts])
    print0(f"[数据集] 共 {len(ds)} 条 prompts")

    if args.deepspeed or args.fsdp:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
            trust_remote_code=True,
        )
    elif local_rank != -1:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )

    if special_tokens_to_add:
        model.resize_token_embeddings(len(tokenizer))

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if args.use_lora:
        target_modules = [s.strip() for s in args.target_modules.split(",") if s.strip()]
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        print0(f"[LoRA] r={args.lora_r} alpha={args.lora_alpha}")

    panas_agent = PANASAgent(
        api_base=args.panas_api_base,
        api_key=args.panas_api_key,
        model=args.panas_model,
    )
    reward_computer = RewardComputer(
        panas_agent=panas_agent,
        is_main_process=is_main_process,
    )

    if is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=f"grpo-{args.model_name.split('/')[-1]}",
            config=vars(args),
        )

    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        beta=args.kl_coef,
        report_to="wandb",
        local_rank=args.local_rank,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        deepspeed=args.deepspeed,
        fsdp=args.fsdp,
        ddp_backend="nccl" if local_rank != -1 else None,
        num_generations=args.group_size,
        max_completion_length=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    rewards_tracker = RewardsTracker(
        output_dir=args.output_dir,
        is_main_process=is_main_process,
        plot_every_n_steps=args.kl_plot_every_n_steps,
    )
    reward_fn = build_reward_fn(
        reward_computer=reward_computer,
        dialogue_manager=dialogue_manager,
        tracker=rewards_tracker,
        group_size=args.group_size,
        is_main_process=is_main_process,
    )

    kl_monitor = KLMonitorCallback(
        output_dir=args.output_dir,
        is_main_process=is_main_process,
        plot_every_n_steps=args.kl_plot_every_n_steps,
    )
    reward_monitor = RewardMonitorCallback(
        tracker=rewards_tracker,
        is_main_process=is_main_process,
        plot_every_n_steps=args.kl_plot_every_n_steps,
    )
    rolling_cb = RollingDialogueCallback(
        dialogue_manager=dialogue_manager,
        tokenizer=tokenizer,
        is_main_process=is_main_process,
        group_size=args.group_size,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_cfg,
        train_dataset=ds,
        reward_funcs=[reward_fn],
        callbacks=[kl_monitor, reward_monitor, rolling_cb],
    )

    rolling_cb.trainer = trainer

    print0("[训练] 开始 GRPO 训练...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print0(f"[训练完成] 模型已保存到 {args.output_dir}")

    if is_main_process:
        wandb.finish()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
