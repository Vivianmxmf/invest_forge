"""HuggingFace Transformers in-process LLM client.

This is a **GPU / server-side** path.  On CPU the model loads and runs but
can be extremely slow for 7B+ parameter models.  Import and use on Mac/CI is
safe because all heavy imports (torch, transformers) are deferred until the
constructor runs — importing this module never requires torch to be installed.

Usage (on GPU server):

    from invest_forge.llm.hf_client import HFLLMClient
    from invest_forge.llm.client import ChatMessage

    client = HFLLMClient(model="Qwen/Qwen2.5-7B-Instruct")
    resp = client.complete([ChatMessage(role="user", content="Hello")])
    print(resp.text)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from invest_forge.common.logging_setup import get_logger
from invest_forge.llm.client import ChatMessage, ChatResponse

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Dtype string -> torch attribute name mapping (strings only, so this module
# is importable without torch on Mac / CI).
# Mirrors the approach used in finetune/train_lora.py::_DTYPE_ALIASES.
# ---------------------------------------------------------------------------

_DTYPE_ALIASES: dict[str, str] = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "float32": "float32",
    "fp32": "float32",
}


def _resolve_torch_dtype(name: str, torch_mod: Any) -> Any:
    """Resolve a dtype string (e.g. ``"bf16"``) to a ``torch.dtype``.

    Passes ``"auto"`` through unchanged (transformers accepts it natively).
    Raises ``ValueError`` for unrecognised names so config typos fail loudly.
    """
    key = (name or "auto").strip().lower()
    if key == "auto":
        return "auto"
    attr = _DTYPE_ALIASES.get(key)
    if attr is None:
        raise ValueError(
            f"Unsupported hf_dtype {name!r}; "
            f"expected one of {sorted(set(_DTYPE_ALIASES))} or 'auto'."
        )
    return getattr(torch_mod, attr)


@dataclass
class HFLLMClient:
    """In-process HuggingFace Transformers chat client.

    All heavy imports (torch, transformers) happen inside ``__post_init__``,
    so importing this module on a machine without these packages installed
    (e.g. the Mac dev environment) never raises an ``ImportError``.

    The loaded model and tokenizer are cached on ``self`` — repeated calls to
    ``complete()`` reuse them without reloading from disk.

    Args:
        model: HuggingFace Hub model ID (e.g. ``"Qwen/Qwen2.5-7B-Instruct"``).
        device: Device map string passed to ``from_pretrained`` as
            ``device_map``.  ``"auto"`` lets accelerate distribute across
            available hardware.
        dtype: Dtype string (``"bf16"``, ``"fp16"``, ``"fp32"``, ``"auto"``).
            ``"auto"`` lets transformers pick the best dtype.
        temperature: Default sampling temperature.  Overridable per call.
        max_new_tokens: Default maximum new tokens to generate.
    """

    model: str
    device: str = "auto"
    dtype: str = "auto"
    temperature: float = 0.1
    max_new_tokens: int = 1024
    name: str = field(init=False)

    # Internal caches — not part of the public interface.
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _hf_model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = f"hf:{self.model}"
        self._lazy_init()

    def _lazy_init(self) -> None:
        """Load torch + transformers and initialise the model/tokenizer.

        Separated from ``__post_init__`` for clarity but called immediately.
        Raising here (e.g. torch not installed) surfaces a clear message:
        "install requirements-gpu.txt on the GPU server".
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HFLLMClient requires torch and transformers.  "
                "On the GPU server: pip install -r requirements-gpu.txt "
                "--extra-index-url https://download.pytorch.org/whl/cu121"
            ) from exc

        resolved_dtype = _resolve_torch_dtype(self.dtype, torch)

        logger.info(
            "Loading HF model %s  device_map=%s  dtype=%s",
            self.model,
            self.device,
            resolved_dtype,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=True,
        )
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model,
            device_map=self.device,
            torch_dtype=resolved_dtype,
            trust_remote_code=True,
        )
        self._hf_model.eval()

        logger.info("HF model loaded: %s", self.name)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,  # noqa: ARG002 — kept for Protocol parity
    ) -> ChatResponse:
        """Generate a completion for the given message list.

        Converts ``ChatMessage``s to the ``[{"role", "content"}]`` format,
        applies the tokenizer's chat template (which handles system/user/
        assistant turns correctly for models like Qwen), and decodes only the
        newly generated tokens.

        ``response_format="json"`` does NOT hard-constrain decoding — the
        analyst prompt already instructs the model to emit JSON.  The agent's
        ``_parse_json`` logic is tolerant of minor deviations.

        Returns a :class:`~invest_forge.llm.client.ChatResponse` with token
        usage counts from the generation output.
        """
        import torch

        effective_temp = temperature if temperature is not None else self.temperature
        effective_max = max_tokens if max_tokens is not None else self.max_new_tokens

        raw_messages = [{"role": m.role, "content": m.content} for m in messages]

        input_ids = self._tokenizer.apply_chat_template(
            raw_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Move to the same device as the model's first parameter.
        device = next(self._hf_model.parameters()).device
        input_ids = input_ids.to(device)

        prompt_len = input_ids.shape[-1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": effective_max,
            "pad_token_id": (
                self._tokenizer.eos_token_id
                if self._tokenizer.pad_token_id is None
                else self._tokenizer.pad_token_id
            ),
        }
        if effective_temp > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = effective_temp
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._hf_model.generate(input_ids, **gen_kwargs)

        # Decode only the newly generated tokens (slice off the prompt).
        new_ids = output_ids[0][prompt_len:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        usage = {
            "prompt": prompt_len,
            "completion": len(new_ids),
        }
        return ChatResponse(text=text, usage=usage)
