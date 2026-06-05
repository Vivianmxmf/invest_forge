"""Config loader tests."""
from __future__ import annotations

import importlib

import pytest

import janus_terminal.common.config as config_module


@pytest.mark.unit
def test_settings_load_with_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("LLM_MODEL", "test-llm")
    monkeypatch.setenv("POSTGRES_PORT", "5555")  # unrelated, just checks _int
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "1")
    monkeypatch.setenv("API_PORT", "9999")

    config_module.get_settings.cache_clear()
    settings = config_module.get_settings()
    assert settings.llm.provider == "fake"
    assert settings.llm.model == "test-llm"
    assert settings.observability.langsmith_tracing is True
    assert settings.api_port == 9999


@pytest.mark.unit
def test_invalid_int_raises(monkeypatch):
    monkeypatch.setenv("API_PORT", "not-a-number")
    config_module.get_settings.cache_clear()
    with pytest.raises(ValueError):
        config_module.get_settings()


@pytest.mark.unit
def test_use_langgraph_env_parsing(monkeypatch) -> None:
    """USE_LANGGRAPH env var is correctly parsed by get_settings."""
    monkeypatch.setenv("USE_LANGGRAPH", "true")
    config_module.get_settings.cache_clear()
    settings = config_module.get_settings()
    assert settings.use_langgraph is True

    monkeypatch.delenv("USE_LANGGRAPH", raising=False)
    config_module.get_settings.cache_clear()
    settings = config_module.get_settings()
    assert settings.use_langgraph is False


@pytest.mark.unit
def teardown_module(_module):  # noqa: PT028 - module-level teardown
    config_module.get_settings.cache_clear()
