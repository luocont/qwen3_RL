"""
generate_rl_dataset.py

读取 RL.json，对每个对话里的每条 user 消息按顺序调用咨询师工作流
（5Ps 病例分析 + thinking & reply），将完整输出覆盖对应的 assistant content，
同时将结果保存为 ShareGPT 格式。

用法：
    python generate_rl_dataset.py
    python generate_rl_dataset.py --rl_json RL.json --output_rl RL_updated.json --output_sharegpt sharegpt.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from workflow import CaseAnalysisWorkflow


def load_env():
    env_file = ROOT / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")


def extract_reply(full_output: str) -> str:
    m = re.search(r"</thinking>([\s\S]*)$", full_output)
    if m:
        return m.group(1).strip()
    m = re.search(r"</5Ps>[\s\S]*?(?:</thinking>)?([\s\S]+)$", full_output)
    if m:
        return m.group(1).strip()
    return full_output.strip()


def save_progress(updated_data: list, sharegpt_data: list, out_rl: Path, out_sg: Path):
    with open(out_rl, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)
    with open(out_sg, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)


def process_conversation(
    conv: dict,
    api_key: str,
    on_turn_done=None,
) -> dict:
    messages = conv.get("messages", [])

    workflow = CaseAnalysisWorkflow(api_key=api_key)

    user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]

    updated_messages = list(messages)

    for turn_idx, user_idx in enumerate(user_indices):
        user_content = messages[user_idx]["content"]
        print(f"\n  [轮次 {turn_idx + 1}] user: {user_content[:60]}...")

        five_ps_raw = workflow.analyze_case_with_5ps(user_content)
        print(f"  5Ps 生成完成")

        thinking, reply = workflow.think_and_reply(user_content)
        print(f"  thinking+reply 生成完成")

        five_ps_block = ""
        m = re.search(r"<5Ps>[\s\S]*?</5Ps>", five_ps_raw)
        if m:
            five_ps_block = m.group(0)
        else:
            five_ps_block = five_ps_raw.strip()

        thinking_block = f"<thinking>\n{thinking.strip()}\n</thinking>" if thinking.strip() else ""
        full_content = "\n".join(filter(None, [five_ps_block, thinking_block, reply.strip()]))

        workflow.update_conversation_history(user_content, reply)

        if turn_idx < len(assistant_indices):
            asst_idx = assistant_indices[turn_idx]
            updated_messages[asst_idx] = {
                "role": "assistant",
                "content": full_content,
            }
        else:
            print(f"  [警告] 第 {turn_idx + 1} 轮没有对应的 assistant 消息，跳过覆盖")

        if on_turn_done is not None:
            result_so_far = dict(conv)
            result_so_far["messages"] = list(updated_messages)
            on_turn_done(result_so_far, turn_idx + 1)

        time.sleep(0.5)

    result = dict(conv)
    result["messages"] = updated_messages
    return result


def to_sharegpt(conv: dict) -> dict:
    messages = conv.get("messages", [])
    conversations = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            conversations.append({"from": "system", "value": msg["content"]})
        elif role == "user":
            conversations.append({"from": "human", "value": msg["content"]})
        elif role == "assistant":
            conversations.append({"from": "gpt", "value": msg["content"]})
    return {
        "id": str(conv.get("id", "")),
        "conversations": conversations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl_json", default="RL.json")
    parser.add_argument("--output_rl", default="RL_updated.json")
    parser.add_argument("--output_sharegpt", default="sharegpt.json")
    args = parser.parse_args()

    load_env()
    api_key = "sk-40fb3997d3ed485ba390a9c4ae3bd2d2"
    if not api_key:
        print("错误：未找到 DASHSCOPE_API_KEY，请在 .env 文件中配置")
        sys.exit(1)

    rl_path = ROOT / args.rl_json
    with open(rl_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"共 {len(data)} 个对话")

    updated_data = []
    sharegpt_data = []

    out_rl = ROOT / args.output_rl
    out_sg = ROOT / args.output_sharegpt

    for conv_idx, conv in enumerate(data):
        print(f"\n{'='*50}")
        print(f"处理对话 {conv_idx + 1}/{len(data)}  id={conv.get('id', conv_idx)}")

        updated_data.append(None)
        sharegpt_data.append(None)

        def on_turn_done(result_so_far, turn_idx, _cidx=conv_idx):
            updated_data[_cidx] = result_so_far
            sharegpt_data[_cidx] = to_sharegpt(result_so_far)
            save_progress(
                [x for x in updated_data if x is not None],
                [x for x in sharegpt_data if x is not None],
                out_rl,
                out_sg,
            )
            print(f"  💾 已保存（对话 {_cidx + 1} 第 {turn_idx} 轮）")

        try:
            updated_conv = process_conversation(conv, api_key, on_turn_done=on_turn_done)
        except Exception as e:
            import traceback
            print(f"[错误] 对话 {conv_idx + 1} 处理失败: {e}")
            traceback.print_exc()
            updated_conv = conv

        updated_data[conv_idx] = updated_conv
        sharegpt_data[conv_idx] = to_sharegpt(updated_conv)
        save_progress(
            [x for x in updated_data if x is not None],
            [x for x in sharegpt_data if x is not None],
            out_rl,
            out_sg,
        )
        print(f"  ✅ 对话 {conv_idx + 1} 全部完成并保存")

    print(f"\n完成！")
    print(f"  覆盖版本: {out_rl}")
    print(f"  ShareGPT: {out_sg}")


if __name__ == "__main__":
    main()
