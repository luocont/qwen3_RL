# -*- coding: utf-8 -*-
import json
from openai import OpenAI
from .prompts import PANAS_SYSTEM_PROMPT, PANAS_IMPROVEMENT_WEIGHT_PA, PANAS_IMPROVEMENT_WEIGHT_NA

_PA_KEYS = [
    "interested", "excited", "strong", "enthusiastic", "proud",
    "alert", "inspired", "determined", "attentive", "active",
]
_NA_KEYS = [
    "distressed", "upset", "guilty", "scared", "hostile",
    "irritable", "ashamed", "nervous", "jittery", "afraid",
]


def _safe_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start: end + 1])
            except Exception:
                pass
    return None


class PANASAgent:
    """
    PANAS 态度评定 Agent：
    对当前对话轮次进行情绪评估，返回结构化 JSON 结果。
    同时提供患者改善奖励计算方法。
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
        self.client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def evaluate(self, dialogue_content: str) -> dict:
        messages = [
            {"role": "system", "content": PANAS_SYSTEM_PROMPT},
            {"role": "user", "content": f"当前对话轮次：\n{dialogue_content}"},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
        )
        raw = resp.choices[0].message.content.strip()
        data = _safe_json(raw) or {}
        return self._normalize(data)

    def _normalize(self, data: dict) -> dict:
        pa = data.get("positive_affect", {})
        na = data.get("negative_affect", {})

        pa_scores = {k: int(pa.get(k, 1)) for k in _PA_KEYS}
        na_scores = {k: int(na.get(k, 1)) for k in _NA_KEYS}

        pa_total = sum(pa_scores.values())
        na_total = sum(na_scores.values())
        pa_avg = pa_total / len(_PA_KEYS)
        na_avg = na_total / len(_NA_KEYS)

        return {
            "positive_affect": {**pa_scores, "total": pa_total, "average": round(pa_avg, 2)},
            "negative_affect": {**na_scores, "total": na_total, "average": round(na_avg, 2)},
        }

    @staticmethod
    def compute_improvement_reward(prev_panas: dict | None, curr_panas: dict) -> float:
        """
        计算患者改善奖励：
        - PA 提升 → 正奖励
        - NA 降低 → 正奖励
        奖励归一化到 [-0.5, 0.5]
        """
        if prev_panas is None:
            return 0.0

        prev_pa = prev_panas.get("positive_affect", {}).get("average", 3.0)
        prev_na = prev_panas.get("negative_affect", {}).get("average", 3.0)
        curr_pa = curr_panas.get("positive_affect", {}).get("average", 3.0)
        curr_na = curr_panas.get("negative_affect", {}).get("average", 3.0)

        delta_pa = curr_pa - prev_pa
        delta_na = prev_na - curr_na

        raw = (
            PANAS_IMPROVEMENT_WEIGHT_PA * delta_pa
            + PANAS_IMPROVEMENT_WEIGHT_NA * delta_na
        )
        max_range = 4.0
        normalized = raw / max_range
        return float(max(-0.5, min(0.5, normalized)))
