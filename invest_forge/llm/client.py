"""LLM client protocol + factory.

The whole agent layer talks to an ``LLMClient``; real providers (OpenAI,
Anthropic, vLLM) are constructed lazily by ``build_client`` from settings.
The fake client lives next door so tests + laptop dev never need a key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from invest_forge.common.config import LLMConfig, get_settings
from invest_forge.common.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class ChatMessage:
    role: str           # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatResponse:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


@runtime_checkable
class LLMClient(Protocol):
    """Minimal sync chat interface used by every agent node."""

    name: str

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
    ) -> ChatResponse: ...


def build_client(config: LLMConfig | None = None) -> LLMClient:
    """Instantiate the configured provider (or the fake client)."""
    cfg = config or get_settings().llm
    provider = cfg.provider.lower()
    if provider == "fake":
        from invest_forge.llm.fake import FakeLLMClient

        logger.info("LLM provider = fake (offline mode)")
        return FakeLLMClient()
    if provider == "openai":
        from invest_forge.llm.openai_client import OpenAILLMClient

        return OpenAILLMClient(model=cfg.model, api_key=cfg.openai_api_key, temperature=cfg.temperature)
    if provider == "anthropic":
        from invest_forge.llm.anthropic_client import AnthropicLLMClient

        return AnthropicLLMClient(model=cfg.model, api_key=cfg.anthropic_api_key, temperature=cfg.temperature)
    if provider == "local":
        from invest_forge.llm.openai_client import OpenAILLMClient

        # vLLM exposes an OpenAI-compatible endpoint
        if not cfg.local_base_url:
            raise ValueError("LLM_PROVIDER=local requires LOCAL_LLM_BASE_URL")
        return OpenAILLMClient(
            model=cfg.local_model or cfg.model,
            api_key=cfg.openai_api_key or "EMPTY",
            base_url=cfg.local_base_url,
            temperature=cfg.temperature,
        )
    raise ValueError(f"unsupported LLM provider: {provider}")
