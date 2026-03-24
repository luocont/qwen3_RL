# -*- coding: utf-8 -*-
"""
数据生成脚本：
使用 Agent Team（ClientAgent + CaseAgent + CounselorAgent）
模拟多轮心理咨询对话，生成 RL 训练数据。

输出格式（rl_data.json）：
[
  {
    "session_id": "...",
    "intake_form": "...",
    "messages": [
      {"role": "user", "content": "Client: ..."},
      {"role": "assistant", "content": "<thinking>...</thinking>\n<5Ps>...</5Ps>\n回复内容"},
      ...
    ],
    "5ps_history": [...],
    "panas_history": [...]
  },
  ...
]
"""
import os
import sys
import json
import uuid
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import ClientAgent, CaseAgent, PANASAgent, CounselorAgent
from config import DataGenConfig


def build_history_str(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = "咨询师" if m["role"] == "assistant" else "来访者"
        content = m["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def run_session(
    session_id: str,
    intake_form: str,
    client_agent: ClientAgent,
    case_agent: CaseAgent,
    panas_agent: PANASAgent,
    counselor_agent: CounselorAgent,
    max_turns: int = 12,
    verbose: bool = True,
) -> dict:
    messages: list[dict] = []
    case_history: list[str] = []
    panas_history: list[dict] = []

    current_case = None
    prev_panas = None
    attitude = "来访者对咨询持开放态度，愿意分享自己的困扰，但内心仍有些防御。"

    if verbose:
        print(f"\n{'='*60}")
        print(f"[会话 {session_id}] 开始")
        print(f"[人设] {intake_form[:80]}...")
        print(f"{'='*60}\n")

    for turn in range(max_turns):
        history_str = build_history_str(messages)

        client_utterance = client_agent.generate(
            intake_form=intake_form,
            attitude=attitude,
            history=history_str,
        )
        if verbose:
            print(f"[Turn {turn+1}] 来访者: {client_utterance}")

        messages.append({"role": "user", "content": client_utterance})

        current_case = case_agent.update(
            latest_dialogue_turn=client_utterance,
            current_case=current_case,
        )
        case_history.append(current_case)

        counselor_output = counselor_agent.generate(
            user_message=client_utterance,
            previous_5ps_case=current_case,
        )
        if verbose:
            print(f"[Turn {turn+1}] 咨询师: {counselor_output[:120]}...")

        messages.append({"role": "assistant", "content": counselor_output})

        dialogue_turn_text = f"来访者: {client_utterance}\n咨询师: {counselor_output}"
        curr_panas = panas_agent.evaluate(dialogue_turn_text)
        panas_history.append(curr_panas)

        if verbose:
            pa_avg = curr_panas["positive_affect"]["average"]
            na_avg = curr_panas["negative_affect"]["average"]
            print(f"[Turn {turn+1}] PANAS PA={pa_avg:.2f} NA={na_avg:.2f}")

        prev_panas = curr_panas

        if client_agent.is_end(client_utterance) and turn >= 8:
            if verbose:
                print(f"[会话 {session_id}] 来访者主动结束，共 {turn+1} 轮")
            break

    return {
        "session_id": session_id,
        "intake_form": intake_form,
        "messages": messages,
        "5ps_history": case_history,
        "panas_history": panas_history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_base", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api_key", type=str, default=os.environ.get("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--client_model", type=str, default="qwen-plus")
    parser.add_argument("--case_model", type=str, default="qwen-plus")
    parser.add_argument("--panas_model", type=str, default="qwen-plus")
    parser.add_argument("--counselor_model", type=str, default="qwen-plus")
    parser.add_argument("--num_sessions", type=int, default=50)
    parser.add_argument("--max_turns", type=int, default=12)
    parser.add_argument("--output", type=str, default="./data/rl_data.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    client_agent = ClientAgent(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.client_model,
    )
    case_agent = CaseAgent(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.case_model,
    )
    panas_agent = PANASAgent(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.panas_model,
    )
    counselor_agent = CounselorAgent(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.counselor_model,
    )

    cfg = DataGenConfig()
    intake_forms = cfg.intake_forms

    all_sessions = []
    for i in range(args.num_sessions):
        intake_form = intake_forms[i % len(intake_forms)]
        session_id = str(uuid.uuid4())[:8]
        try:
            session = run_session(
                session_id=session_id,
                intake_form=intake_form,
                client_agent=client_agent,
                case_agent=case_agent,
                panas_agent=panas_agent,
                counselor_agent=counselor_agent,
                max_turns=args.max_turns,
                verbose=args.verbose,
            )
            all_sessions.append(session)
            print(f"[进度] {i+1}/{args.num_sessions} 会话完成")
        except Exception as e:
            print(f"[错误] 会话 {session_id} 失败: {e}")
            continue

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_sessions, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 共生成 {len(all_sessions)} 个会话，保存到 {args.output}")


if __name__ == "__main__":
    main()
