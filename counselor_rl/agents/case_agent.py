# -*- coding: utf-8 -*-
import re
from openai import OpenAI
from .prompts import FIVE_PS_SYSTEM_PROMPT

_EMPTY_CASE = """\
P1 主诉：待完善

P2 易感因素：待完善

P3 诱发因素：待完善

P4 维持因素：待完善

P5 保护因素：待完善"""

_5PS_BLOCK_RE = re.compile(r"<5Ps>([\s\S]*?)</5Ps>", re.MULTILINE)


class CaseAgent:
    """
    5Ps 病例分析 Agent：
    根据最新对话轮次，迭代更新患者的 5Ps 病例。
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
        timeout: int = 60,
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def update(self, latest_dialogue_turn: str, current_case: str | None = None) -> str:
        case_text = current_case if current_case else _EMPTY_CASE
        system_content = FIVE_PS_SYSTEM_PROMPT.format(latest_5ps_case=case_text)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": latest_dialogue_turn},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"enable_thinking": False},
        )
        raw = resp.choices[0].message.content.strip()
        return self._extract_case(raw)

    def _extract_case(self, raw: str) -> str:
        m = _5PS_BLOCK_RE.search(raw)
        if m:
            return m.group(1).strip()
        return raw

    def count_filled_ps(self, case_text: str) -> int:
        count = 0
        for line in case_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("P") and "待完善" not in stripped and len(stripped) > 5:
                count += 1
        return count
