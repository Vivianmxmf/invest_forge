"""OpenAI / vLLM (OpenAI-compatible) client."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from invest_forge.common.logging_setup import get_logger
from invest_forge.llm.client import ChatMessage, ChatResponse

logger = get_logger(__name__)


def _to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert ChatMessage list to the OpenAI API message format.

    - Message with EMPTY images → ``{"role": ..., "content": <str>}`` —
      byte-identical to the format used before vision support was added.
    - Message with non-empty images → OpenAI vision multipart format:
      ``{"role": ..., "content": [{"type": "text", ...}, {"type": "image_url", ...}, ...]}``.

    This function is pure (no side effects) and is the single place where
    the wire format is determined.  When no message carries any image the
    output is byte-identical to ``[{"role": m.role, "content": m.content} for m in messages]``.
    """
    result: list[dict[str, Any]] = []
    for m in messages:
        if not m.images:
            # Text-only path — unchanged from pre-vision behaviour.
            result.append({"role": m.role, "content": m.content})
        else:
            # Vision path — build a multipart content list.
            parts: list[dict[str, Any]] = [{"type": "text", "text": m.content}]
            for data_url in m.images:
                parts.append({"type": "image_url", "image_url": {"url": data_url}})
            result.append({"role": m.role, "content": parts})
    return result


@dataclass
class OpenAILLMClient:
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.1
    name: str = "openai"

    def __post_init__(self) -> None:
        # Lazy import — keeps laptop CI fast and lets the package import
        # without the openai SDK installed.
        from openai import OpenAI

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
    ) -> ChatResponse:
        kwargs: dict = {
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = {}
        if resp.usage is not None:
            usage = {
                "prompt": resp.usage.prompt_tokens,
                "completion": resp.usage.completion_tokens,
            }
        return ChatResponse(text=text, usage=usage, raw=resp)
