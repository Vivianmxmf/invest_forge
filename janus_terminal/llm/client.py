"""LLM client protocol + factory.

The whole agent layer talks to an ``LLMClient``; real providers (OpenAI,
Anthropic, vLLM) are constructed lazily by ``build_client`` from settings.
The fake client lives next door so tests + laptop dev never need a key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from janus_terminal.common.config import LLMConfig, get_settings
from janus_terminal.common.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class ChatMessage:
    role: str           # "system" | "user" | "assistant"
    content: str
    # Tuple of data-URL strings (data:<mime>;base64,...) for vision payloads.
    # Empty tuple == text-only message; default keeps every existing
    # construction unchanged.
    images: tuple[str, ...] = ()


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
        from janus_terminal.llm.fake import FakeLLMClient

        logger.info("LLM provider = fake (offline mode)")
        return FakeLLMClient()
    if provider == "openai":
        from janus_terminal.llm.openai_client import OpenAILLMClient

        return OpenAILLMClient(model=cfg.model, api_key=cfg.openai_api_key, temperature=cfg.temperature)
    if provider == "anthropic":
        from janus_terminal.llm.anthropic_client import AnthropicLLMClient

        return AnthropicLLMClient(model=cfg.model, api_key=cfg.anthropic_api_key, temperature=cfg.temperature)
    if provider == "local":
        from janus_terminal.llm.openai_client import OpenAILLMClient

        # vLLM exposes an OpenAI-compatible endpoint
        if not cfg.local_base_url:
            raise ValueError("LLM_PROVIDER=local requires LOCAL_LLM_BASE_URL")
        return OpenAILLMClient(
            model=cfg.local_model or cfg.model,
            api_key=cfg.openai_api_key or "EMPTY",
            base_url=cfg.local_base_url,
            temperature=cfg.temperature,
        )
    if provider in ("hf", "huggingface", "transformers"):
        from janus_terminal.llm.hf_client import HFLLMClient

        logger.info("LLM provider = hf (in-process HuggingFace model: %s)", cfg.model)
        return HFLLMClient(
            model=cfg.model,
            device=cfg.hf_device,
            dtype=cfg.hf_dtype,
            temperature=cfg.temperature,
        )
    raise ValueError(f"unsupported LLM provider: {provider}")


def build_vision_client(cfg: LLMConfig) -> LLMClient | None:
    """Return a vision-capable LLMClient when configured, otherwise None.

    A vision client is ONLY constructed when:
      * ``cfg.provider == "local"``  (server-side vLLM path), AND
      * ``cfg.local_vision_model`` is set.

    When either condition is not met the function returns ``None`` and the
    vision node becomes a no-op, keeping the graph behaviour identical to
    how it ran before the multimodal feature was added.
    """
    if cfg.provider.lower() != "local" or not cfg.local_vision_model:
        return None
    from janus_terminal.llm.openai_client import OpenAILLMClient

    base_url = cfg.local_vision_base_url or cfg.local_base_url
    logger.info(
        "vision client → local model %s at %s",
        cfg.local_vision_model,
        base_url,
    )
    return OpenAILLMClient(
        model=cfg.local_vision_model,
        api_key=cfg.openai_api_key or "EMPTY",
        base_url=base_url,
        temperature=cfg.temperature,
    )


def build_analyst_client(config: LLMConfig | None = None) -> LLMClient:
    """Return an LLMClient pointed at the analyst adapter.

    Only diverges from ``build_client`` when the provider is ``local`` AND
    ``local_analyst_model`` is set (e.g. the ``investforge-analyst`` LoRA
    adapter name registered with vLLM).  All other providers — ``fake``,
    ``openai``, ``anthropic`` — fall through to the default client unchanged,
    keeping backward compatibility when running offline or against cloud APIs.
    """
    cfg = config or get_settings().llm
    if cfg.provider.lower() == "local" and cfg.local_analyst_model:
        from janus_terminal.llm.openai_client import OpenAILLMClient

        if not cfg.local_base_url:
            raise ValueError("LLM_PROVIDER=local requires LOCAL_LLM_BASE_URL")
        logger.info(
            "analyst client → local adapter %s at %s",
            cfg.local_analyst_model,
            cfg.local_base_url,
        )
        return OpenAILLMClient(
            model=cfg.local_analyst_model,
            api_key=cfg.openai_api_key or "EMPTY",
            base_url=cfg.local_base_url,
            temperature=cfg.temperature,
        )
    # No analyst-specific override; reuse the default client.
    return build_client(cfg)
