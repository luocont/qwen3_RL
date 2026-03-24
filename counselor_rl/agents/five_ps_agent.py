# -*- coding: utf-8 -*-
import re
from utils.api_client import chat_completions
from agents.prompts import FIVE_PS_SYSTEM_PROMPT

DEFAULT_5PS = """<5Ps>
P1 主诉：待完善

P2 易感因素：待完善

P3 诱发因素：待完善

P4 维持因素：待完善

P5 保护因素：待完善
</5Ps>"""


class FivePsAgent:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 800,
        timeout: int = 60,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def update(self, dialogue_context: str, current_5ps: str = "") -> str:
        if not current_5ps:
            current_5ps = DEFAULT_5PS

        system_prompt = FIVE_PS_SYSTEM_PROMPT.format(**{"最新的5Ps病例": current_5ps})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请根据以下对话内容更新5Ps病例：\n\n{dialogue_context}"},
        ]

        content = chat_completions(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            enable_thinking=False,
        )

        return self._extract_5ps(content) or current_5ps

    def _extract_5ps(self, text: str) -> str | None:
        m = re.search(r"<5Ps>([\s\S]*?)</5Ps>", text)
        if m:
            return f"<5Ps>{m.group(1)}</5Ps>"
        return None
