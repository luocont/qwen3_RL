# -*- coding: utf-8 -*-
import re
from utils.api_client import chat_completions
from agents.prompts import COUNSELOR_SYSTEM_PROMPT


class CounselorAgent:
    """
    REBT 思考与回复 Agent（参考智能体）
    输出格式：<thinking>...</thinking> + reply内容
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 60,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._think_pat = re.compile(r"<thinking>([\s\S]*?)</thinking>([\s\S]*)$", re.MULTILINE)

    def respond(self, dialogue_history: list[dict], current_5ps: str) -> str:
        system_prompt = COUNSELOR_SYSTEM_PROMPT.format(previous_5ps_case=current_5ps)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(dialogue_history)

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

        return content.strip()

    def extract_think_and_reply(self, output: str) -> tuple[str, str]:
        m = self._think_pat.search(output)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return "", output.strip()
