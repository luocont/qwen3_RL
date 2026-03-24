"""
通用 LLM 客户端模块
支持 OpenAI 兼容 API、阿里云百炼、OpenRouter 等多种 API 提供商
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI, AzureOpenAI


class APIProvider(Enum):
    """API 提供商枚举"""
    OPENAI = "openai"
    AZURE = "azure"
    ALIYUN = "aliyun"
    OPENROUTER = "openrouter"
    LOCAL = "local"  # vLLM 等本地部署


@dataclass
class APIConfig:
    """API 配置数据类"""
    provider: APIProvider
    api_key: str
    base_url: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 512
    # Azure 特定配置
    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    deployment: Optional[str] = None
    # OpenRouter 特定配置
    extra_headers: Optional[Dict[str, str]] = None
    extra_body: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """配置验证和默认值设置"""
        if self.provider == APIProvider.ALIYUN:
            if self.base_url is None:
                self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif self.provider == APIProvider.OPENROUTER:
            if self.base_url is None:
                self.base_url = "https://openrouter.ai/api/v1"
            if self.extra_headers is None:
                self.extra_headers = {
                    "HTTP-Referer": "https://github.com/your-repo",
                    "X-Title": "MagNet"
                }
        elif self.provider == APIProvider.LOCAL:
            if self.base_url is None:
                self.base_url = "http://localhost:8000/v1"


class LLMClient:
    """
    通用 LLM 客户端类
    支持多种 API 提供商的统一调用接口
    """

    def __init__(self, config: APIConfig):
        """
        初始化 LLM 客户端

        Args:
            config: APIConfig 配置对象
        """
        self.config = config
        self.client = self._create_client()
        self.model = config.deployment if config.provider == APIProvider.AZURE else config.model

    def _create_client(self):
        """根据配置创建对应的客户端"""
        if self.config.provider == APIProvider.AZURE:
            return AzureOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=self.config.azure_endpoint,
                api_version=self.config.api_version,
            )
        else:
            return OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ):
        """
        发送聊天补全请求

        Args:
            messages: 消息列表
            temperature: 温度参数（可选，默认使用配置值）
            max_tokens: 最大 token 数（可选，默认使用配置值）
            stream: 是否流式输出
            **kwargs: 其他参数

        Returns:
            API 响应对象
        """
        params = {
            "messages": messages,
            "model": self.model,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "stream": stream,
            **kwargs
        }

        # OpenRouter 额外配置
        if self.config.provider == APIProvider.OPENROUTER:
            if self.config.extra_headers:
                params["extra_headers"] = self.config.extra_headers
            if self.config.extra_body:
                params["extra_body"] = self.config.extra_body

        return self.client.chat.completions.create(**params)

    def completion(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ):
        """
        发送文本补全请求（用于兼容旧代码）
        注意：使用 chat.completions API 而不是 completions API
        因为许多提供商（如阿里云）只支持 chat.completions

        Args:
            prompt: 输入提示
            temperature: 温度参数
            max_tokens: 最大 token 数
            stream: 是否流式输出
            **kwargs: 其他参数

        Returns:
            API 响应对象
        """
        # 将 prompt 转换为 messages 格式
        messages = [{"role": "user", "content": prompt}]

        params = {
            "messages": messages,
            "model": self.model,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "stream": stream,
            **kwargs
        }

        # OpenRouter 额外配置
        if self.config.provider == APIProvider.OPENROUTER:
            if self.config.extra_headers:
                params["extra_headers"] = self.config.extra_headers
            if self.config.extra_body:
                params["extra_body"] = self.config.extra_body

        return self.client.chat.completions.create(**params)


def create_client_from_env() -> LLMClient:
    """
    从环境变量创建客户端

    环境变量:
        LLM_PROVIDER: API 提供商 (openai/azure/aliyun/openrouter/local)
        LLM_API_KEY: API 密钥
        LLM_BASE_URL: API 基础 URL (可选)
        LLM_MODEL: 模型名称 (默认: gpt-4o-mini)
        LLM_TEMPERATURE: 温度参数 (默认: 0.7)
        LLM_MAX_TOKENS: 最大 token 数 (默认: 512)
        LLM_AZURE_ENDPOINT: Azure 端点 (Azure 必需)
        LLM_API_VERSION: API 版本 (Azure 必需)
        LLM_DEPLOYMENT: 部署名称 (Azure 必需)

    Returns:
        LLMClient 实例
    """
    provider_str = os.getenv("LLM_PROVIDER", "local")
    provider = APIProvider(provider_str.lower())

    config = APIConfig(
        provider=provider,
        api_key=os.getenv("LLM_API_KEY", "dummy-key"),
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
        azure_endpoint=os.getenv("LLM_AZURE_ENDPOINT"),
        api_version=os.getenv("LLM_API_VERSION"),
        deployment=os.getenv("LLM_DEPLOYMENT"),
    )

    return LLMClient(config)


def create_aliyun_client(
    api_key: Optional[str] = None,
    model: str = "qwen2.5-7b-instruct",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> LLMClient:
    """
    创建阿里云百炼客户端便捷函数

    Args:
        api_key: API 密钥（默认从环境变量 DASHSCOPE_API_KEY 读取）
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大 token 数

    Returns:
        LLMClient 实例
    """
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError("阿里云 API Key 未提供，请设置 DASHSCOPE_API_KEY 环境变量")

    config = APIConfig(
        provider=APIProvider.ALIYUN,
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return LLMClient(config)


def create_openrouter_client(
    api_key: Optional[str] = None,
    model: str = "openai/chatgpt-4o-latest",
    temperature: float = 0.7,
    max_tokens: int = 512,
    referer: Optional[str] = None,
    title: Optional[str] = None,
) -> LLMClient:
    """
    创建 OpenRouter 客户端便捷函数

    Args:
        api_key: API 密钥（默认从环境变量 OPENROUTER_API_KEY 读取）
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大 token 数
        referer: HTTP Referer 头（用于 OpenRouter 排名）
        title: Site Title 头（用于 OpenRouter 排名）

    Returns:
        LLMClient 实例
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError("OpenRouter API Key 未提供，请设置 OPENROUTER_API_KEY 环境变量")

    extra_headers = {
        "HTTP-Referer": referer or "https://github.com/your-repo",
        "X-Title": title or "MagNet",
    }

    config = APIConfig(
        provider=APIProvider.OPENROUTER,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers=extra_headers,
    )

    return LLMClient(config)


def create_local_client(
    base_url: str = "http://localhost:8000/v1",
    model: str = "model_name",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> LLMClient:
    """
    创建本地部署客户端便捷函数（vLLM 等）

    Args:
        base_url: 本地 API 地址
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大 token 数

    Returns:
        LLMClient 实例
    """
    config = APIConfig(
        provider=APIProvider.LOCAL,
        api_key="dummy-key",
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return LLMClient(config)


# 预设配置示例
PRESET_CONFIGS = {
    "aliyun_qwen": {
        "provider": APIProvider.ALIYUN,
        "api_key": "",  # 需要填写
        "model": "qwen2.5-7b-instruct",
    },
    "openrouter_gpt4o": {
        "provider": APIProvider.OPENROUTER,
        "api_key": "",  # 需要填写
        "model": "openai/chatgpt-4o-latest",
    },
    "local_vllm": {
        "provider": APIProvider.LOCAL,
        "api_key": "dummy-key",
        "base_url": "http://localhost:8000/v1",
        "model": "model_name",
    },
    "azure_gpt4o": {
        "provider": APIProvider.AZURE,
        "api_key": "",  # 需要填写
        "azure_endpoint": "",  # 需要填写
        "api_version": "2024-02-15-preview",
        "deployment": "gpt-4o",
    },
}
