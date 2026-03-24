# -*- coding: utf-8 -*-
import os
import json
from openai import OpenAI

ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def chat_completions(
    model: str,
    messages: list,
    api_base: str = ALIYUN_BASE_URL,
    api_key: str | None = None,
    response_format: dict | None = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    timeout: int = 60,
    enable_thinking: bool = False,
) -> str:
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key: set DASHSCOPE_API_KEY env var or pass api_key.")
    if not api_base or not model:
        raise RuntimeError("Missing API config (api_base, model).")

    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=timeout,
    )

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if response_format is not None:
        kwargs["response_format"] = response_format

    if not enable_thinking:
        kwargs["extra_body"] = {"enable_thinking": False}

    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content


def safe_json_loads(s: str) -> dict | None:
    try:
        return json.loads(s)
    except Exception:
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start: end + 1])
        except Exception:
            pass
    return None
