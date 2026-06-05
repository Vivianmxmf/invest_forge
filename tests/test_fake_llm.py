"""FakeLLMClient unit tests."""
from __future__ import annotations

import json

import pytest

from janus_terminal.llm.client import ChatMessage
from janus_terminal.llm.fake import FakeLLMClient, with_handlers


@pytest.mark.unit
def test_default_responses_match_role_prompts():
    fake = FakeLLMClient()
    r = fake.complete([ChatMessage(role="system", content="[Researcher] task"),
                       ChatMessage(role="user", content="please write research memo")])
    assert "盈利质量" in r.text


@pytest.mark.unit
def test_analyst_default_is_valid_json():
    fake = FakeLLMClient()
    r = fake.complete([ChatMessage(role="user", content="[Analyst] please rate")])
    data = json.loads(r.text)
    assert data["rating"] == "BUY"
    assert 0 <= data["confidence"] <= 1


@pytest.mark.unit
def test_register_overrides_priority():
    fake = FakeLLMClient()
    fake.register(r"\[Researcher\]", "MY CUSTOM MEMO")
    r = fake.complete([ChatMessage(role="user", content="[Researcher] go")])
    assert r.text == "MY CUSTOM MEMO"


@pytest.mark.unit
def test_register_callable():
    fake = FakeLLMClient()
    fake.register(r"echo", lambda msgs: msgs[-1].content.upper())
    r = fake.complete([ChatMessage(role="user", content="echo hello")])
    assert r.text == "ECHO HELLO"


@pytest.mark.unit
def test_with_handlers_helper():
    fake = with_handlers({r"foo": "bar"})
    r = fake.complete([ChatMessage(role="user", content="foo please")])
    assert r.text == "bar"


@pytest.mark.unit
def test_fallback_used_when_no_match():
    fake = FakeLLMClient()
    fake.set_fallback("FALLBACK")
    r = fake.complete([ChatMessage(role="user", content="<<nothing>>")])
    assert r.text == "FALLBACK"


@pytest.mark.unit
def test_records_calls():
    fake = FakeLLMClient()
    fake.complete([ChatMessage(role="user", content="hi")])
    fake.complete([ChatMessage(role="user", content="bye")])
    assert len(fake.calls) == 2
    fake.reset()
    assert fake.calls == []
