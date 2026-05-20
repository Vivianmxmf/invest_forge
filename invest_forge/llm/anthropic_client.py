"""Anthropic Claude client."""
from __future__ import annotations

from dataclasses import dataclass

from invest_forge.common.logging_setup import get_logger
from invest_forge.llm.client import ChatMessage, ChatResponse

logger = get_logger(__name__)


@dataclass
class AnthropicLLMClient:
    model: str
    api_key: str | None = None
    temperature: float = 0.1
    name: str = "anthropic"

    def __post_init__(self) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(api_key=self.api_key)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = 2048,
        response_format: str | None = None,  # noqa: ARG002 - Anthropic uses tool_use for JSON
    ) -> ChatResponse:
        # Anthropic API splits "system" from "messages"
        system_text = "\n\n".join(m.content for m in messages if m.role == "system") or None
        chat = [
            {"role": "user" if m.role != "assistant" else "assistant", "content": m.content}
            for m in messages
            if m.role != "system"
        ]
        resp = self._client.messages.create(
            model=self.model,
            messages=chat,
            system=system_text,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or 2048,
        )
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        usage = {}
        if hasattr(resp, "usage"):
            usage = {"prompt": resp.usage.input_tokens, "completion": resp.usage.output_tokens}
        return ChatResponse(text=text, usage=usage, raw=resp)
