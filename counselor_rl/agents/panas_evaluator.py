# -*- coding: utf-8 -*-
from utils.api_client import chat_completions, safe_json_loads
from agents.prompts import PANAS_SYSTEM_PROMPT


class PanasEvaluator:
    """
    PANAS 态度评定 Agent
    输出 JSON 格式的积极/消极情绪评分
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 60,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def evaluate(self, dialogue_content: str, dialogue_id: str = "", timestamp: str = "") -> dict | None:
        system_prompt = PANAS_SYSTEM_PROMPT.format(**{
            "对话内容": dialogue_content,
            "current_dialogue": dialogue_content,
        })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请根据上述对话内容完成PANAS情绪评估，输出JSON格式结果。"},
        ]

        content = chat_completions(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            enable_thinking=False,
        )

        return safe_json_loads(content)

    def compute_improvement_reward(
        self,
        panas_before: dict | None,
        panas_after: dict | None,
    ) -> float:
        """
        计算患者情绪改善奖励。
        奖励 = (PA_after - PA_before) / 10 - (NA_after - NA_before) / 10
        范围约 [-1.0, 1.0]，正值表示改善。
        """
        if panas_before is None or panas_after is None:
            return 0.0

        try:
            pa_before = float(panas_before.get("positive_affect", {}).get("average", 2.5))
            na_before = float(panas_before.get("negative_affect", {}).get("average", 2.5))
            pa_after = float(panas_after.get("positive_affect", {}).get("average", 2.5))
            na_after = float(panas_after.get("negative_affect", {}).get("average", 2.5))
        except (TypeError, ValueError):
            return 0.0

        pa_delta = pa_after - pa_before
        na_delta = na_after - na_before

        reward = (pa_delta - na_delta) / 4.0
        return float(max(-1.0, min(1.0, reward)))
