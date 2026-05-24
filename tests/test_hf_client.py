"""Offline tests for the HuggingFace in-process LLM provider.

All tests are network-free and do NOT require torch or transformers.
They verify:
  - invest_forge.llm.hf_client imports cleanly without torch installed.
  - build_client dispatches to HFLLMClient (or a clear non-"unsupported" path)
    for the "hf", "huggingface", and "transformers" provider aliases.
  - LLMConfig exposes hf_device / hf_dtype with correct defaults.
  - get_settings() honours HF_DEVICE / HF_DTYPE environment variables.
"""
from __future__ import annotations

import importlib
import sys

import pytest

import invest_forge.common.config as config_module


# ---------------------------------------------------------------------------
# 1. Module-level import must NOT require torch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hf_client_module_imports_without_torch() -> None:
    """invest_forge.llm.hf_client must be importable even when torch is absent."""
    # If torch is already installed in the test env this test still passes —
    # we are checking the import is unconditionally safe, not that torch is absent.
    mod = importlib.import_module("invest_forge.llm.hf_client")
    assert hasattr(mod, "HFLLMClient"), "HFLLMClient not found in hf_client module"


@pytest.mark.unit
def test_hf_client_dtype_aliases_are_strings() -> None:
    """_DTYPE_ALIASES must be defined at module level with no torch references."""
    mod = importlib.import_module("invest_forge.llm.hf_client")
    aliases = mod._DTYPE_ALIASES
    assert isinstance(aliases, dict)
    # All values should be plain strings (torch attribute names), not torch objects.
    for key, val in aliases.items():
        assert isinstance(key, str), f"key {key!r} is not a string"
        assert isinstance(val, str), f"value for {key!r} is not a string"


# ---------------------------------------------------------------------------
# 2. build_client dispatches to HFLLMClient for all provider aliases
# ---------------------------------------------------------------------------


class _StubHFLLMClient:
    """Lightweight stand-in for HFLLMClient — does not load any model."""

    def __init__(self, *, model: str, device: str, dtype: str, temperature: float) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.temperature = temperature
        self.name = f"hf:{model}"

    def complete(self, messages, *, temperature=None, max_tokens=None, response_format=None):
        raise NotImplementedError("stub — not for real inference")


@pytest.mark.unit
@pytest.mark.parametrize("alias", ["hf", "huggingface", "transformers"])
def test_build_client_dispatches_to_hf_for_all_aliases(
    alias: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_client must recognise all hf aliases and NOT raise 'unsupported provider'."""
    from invest_forge.common.config import LLMConfig
    import invest_forge.llm.client as client_mod

    # Patch HFLLMClient inside the client module's import namespace so the
    # factory uses our stub instead of trying to load a real model.
    monkeypatch.setattr(
        # We patch at the hf_client module level so the lazy import picks it up.
        "invest_forge.llm.hf_client.HFLLMClient",
        _StubHFLLMClient,
    )

    cfg = LLMConfig(
        provider=alias,
        model="test-model",
        openai_api_key=None,
        anthropic_api_key=None,
        gemini_api_key=None,
        local_base_url=None,
        local_model=None,
    )

    client = client_mod.build_client(cfg)
    assert client.name == "hf:test-model"
    assert isinstance(client, _StubHFLLMClient)


@pytest.mark.unit
def test_build_client_hf_raises_import_error_not_unsupported_when_no_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When torch is absent, build_client(hf) must fail with ImportError, not
    ValueError('unsupported provider').  This verifies that the hf branch IS
    reached and that the error is clearly from the model-load path.
    """
    from invest_forge.common.config import LLMConfig
    import invest_forge.llm.client as client_mod

    # Simulate torch being absent by making HFLLMClient raise ImportError on init.
    original_hf = None
    try:
        import invest_forge.llm.hf_client as hf_mod
        original_hf = hf_mod.HFLLMClient

        class _NoTorchStub:
            def __init__(self, **kwargs):
                raise ImportError("torch not installed (simulated)")

        monkeypatch.setattr(hf_mod, "HFLLMClient", _NoTorchStub)

        cfg = LLMConfig(
            provider="hf",
            model="some/model",
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            local_base_url=None,
            local_model=None,
        )

        with pytest.raises(ImportError):
            client_mod.build_client(cfg)

    finally:
        if original_hf is not None:
            import invest_forge.llm.hf_client as hf_mod2
            monkeypatch.setattr(hf_mod2, "HFLLMClient", original_hf)


# ---------------------------------------------------------------------------
# 3. LLMConfig has hf_device / hf_dtype with correct defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_llm_config_hf_fields_default_to_auto() -> None:
    """LLMConfig must expose hf_device and hf_dtype defaulting to 'auto'."""
    from invest_forge.common.config import LLMConfig

    cfg = LLMConfig(
        provider="fake",
        model="fake-llm",
        openai_api_key=None,
        anthropic_api_key=None,
        gemini_api_key=None,
        local_base_url=None,
        local_model=None,
    )
    assert cfg.hf_device == "auto"
    assert cfg.hf_dtype == "auto"


@pytest.mark.unit
def test_llm_config_hf_fields_accept_explicit_values() -> None:
    """LLMConfig must store explicitly provided hf_device and hf_dtype."""
    from invest_forge.common.config import LLMConfig

    cfg = LLMConfig(
        provider="hf",
        model="Qwen/Qwen2.5-7B-Instruct",
        openai_api_key=None,
        anthropic_api_key=None,
        gemini_api_key=None,
        local_base_url=None,
        local_model=None,
        hf_device="cuda:0",
        hf_dtype="bf16",
    )
    assert cfg.hf_device == "cuda:0"
    assert cfg.hf_dtype == "bf16"


# ---------------------------------------------------------------------------
# 4. get_settings() reads HF_DEVICE / HF_DTYPE from env
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_settings_hf_device_dtype_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_settings() must return hf_device='auto' and hf_dtype='auto' by default."""
    monkeypatch.delenv("HF_DEVICE", raising=False)
    monkeypatch.delenv("HF_DTYPE", raising=False)
    config_module.get_settings.cache_clear()

    settings = config_module.get_settings()
    assert settings.llm.hf_device == "auto"
    assert settings.llm.hf_dtype == "auto"

    config_module.get_settings.cache_clear()


@pytest.mark.unit
def test_get_settings_hf_device_dtype_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_settings() must honour HF_DEVICE and HF_DTYPE environment variables."""
    monkeypatch.setenv("HF_DEVICE", "cuda:2")
    monkeypatch.setenv("HF_DTYPE", "fp16")
    config_module.get_settings.cache_clear()

    settings = config_module.get_settings()
    assert settings.llm.hf_device == "cuda:2"
    assert settings.llm.hf_dtype == "fp16"

    config_module.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 5. _resolve_torch_dtype helper (no torch needed — uses string aliases only)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,expected_attr",
    [
        ("bf16", "bfloat16"),
        ("bfloat16", "bfloat16"),
        ("fp16", "float16"),
        ("float16", "float16"),
        ("half", "float16"),
        ("fp32", "float32"),
        ("float32", "float32"),
    ],
)
def test_resolve_torch_dtype_known_aliases(name: str, expected_attr: str) -> None:
    """_resolve_torch_dtype must map known aliases to the correct torch attribute name."""
    from invest_forge.llm.hf_client import _resolve_torch_dtype

    # Use a simple namespace object to simulate torch without importing it.
    class _FakeTorch:
        bfloat16 = "bfloat16_dtype"
        float16 = "float16_dtype"
        float32 = "float32_dtype"

    result = _resolve_torch_dtype(name, _FakeTorch())
    assert result == getattr(_FakeTorch, expected_attr)


@pytest.mark.unit
def test_resolve_torch_dtype_auto_passthrough() -> None:
    """_resolve_torch_dtype must return the string 'auto' unchanged."""
    from invest_forge.llm.hf_client import _resolve_torch_dtype

    result = _resolve_torch_dtype("auto", object())
    assert result == "auto"


@pytest.mark.unit
def test_resolve_torch_dtype_unknown_raises() -> None:
    """_resolve_torch_dtype must raise ValueError for an unrecognised dtype string."""
    from invest_forge.llm.hf_client import _resolve_torch_dtype

    with pytest.raises(ValueError, match="Unsupported hf_dtype"):
        _resolve_torch_dtype("int8", object())


# ---------------------------------------------------------------------------
# Teardown: always clear the settings cache to avoid cross-test pollution.
# ---------------------------------------------------------------------------


def teardown_module(_module) -> None:  # noqa: PT028
    config_module.get_settings.cache_clear()
