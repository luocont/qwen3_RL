# -*- coding: utf-8 -*-
from agents.prompts import CLIENT_SYSTEM_PROMPT, CLIENT_USER_PROMPT
from utils.api_client import chat_completions


class ClientAgent:
    """
    来访者模拟 Agent
    输出格式：Client: <内容>，结束时包含 [/END]
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float = 0.8,
        max_tokens: int = 256,
        timeout: int = 60,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def respond(
        self,
        intake_form: str,
        attitude: str,
        dialogue_history: str,
    ) -> str:
        system_content = CLIENT_SYSTEM_PROMPT.format(
            intake_form=intake_form,
            attitude=attitude,
        )
        user_content = CLIENT_USER_PROMPT.format(history=dialogue_history)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        return chat_completions(
            model=self.model,
            messages=messages,
            api_base=self.api_base,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            enable_thinking=False,
        ).strip()

    def is_session_ended(self, client_response: str) -> bool:
        return "[/END]" in client_response
