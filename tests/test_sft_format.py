"""Unit tests for finetune/sft_format.py.

Tests:
  - ``build_analyst_user_prompt`` parity with ``ANALYST_PROMPT.format``
  - JSONL round-trip (write then read)
  - ``to_chat_example`` shape and content
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune.sft_format import (
    SFTExample,
    build_analyst_user_prompt,
    read_jsonl,
    to_chat_example,
    write_jsonl,
)
from janus_terminal.agents.prompts import ANALYST_PROMPT


@pytest.mark.unit
def test_build_analyst_user_prompt_contains_memo() -> None:
    """The returned string must contain the (possibly truncated) memo text."""
    memo = "这是研究员备忘录内容，描述公司基本面情况。"
    result = build_analyst_user_prompt(memo)
    assert memo in result


@pytest.mark.unit
def test_build_analyst_user_prompt_matches_analyst_prompt_format() -> None:
    """Output must be byte-identical to ANALYST_PROMPT.format(...)."""
    memo = "test memo content"
    feedback = "风控意见: 需要补充流动性风险"
    expected = ANALYST_PROMPT.format(
        research_memo=memo[:4000],
        risk_feedback=feedback,
    )
    actual = build_analyst_user_prompt(memo, risk_feedback=feedback)
    assert actual == expected


@pytest.mark.unit
def test_build_analyst_user_prompt_truncates_long_memo() -> None:
    """Memo longer than 4000 chars must be truncated to exactly 4000."""
    long_memo = "A" * 5000
    result = build_analyst_user_prompt(long_memo)
    # The raw 4000-char prefix must appear in the prompt.
    assert "A" * 4000 in result
    # And we should NOT find 4001 consecutive A's.
    assert "A" * 4001 not in result


@pytest.mark.unit
def test_build_analyst_user_prompt_no_feedback() -> None:
    """With empty risk_feedback the prompt still matches ANALYST_PROMPT.format."""
    memo = "短备忘录"
    expected = ANALYST_PROMPT.format(research_memo=memo[:4000], risk_feedback="")
    assert build_analyst_user_prompt(memo) == expected


@pytest.mark.unit
def test_to_chat_example_shape() -> None:
    """SFTExample must have exactly 2 messages: user / assistant (no system).

    Parity: inference sends a single user turn, so training must too.
    """
    gold = {
        "rating": "BUY",
        "confidence": 0.8,
        "rationale": "test rationale",
        "risk_factors": ["风险A", "风险B", "风险C"],
    }
    ex = to_chat_example("memo text", gold)
    assert isinstance(ex, SFTExample)
    assert len(ex.messages) == 2
    roles = [m["role"] for m in ex.messages]
    assert roles == ["user", "assistant"]


@pytest.mark.unit
def test_to_chat_example_has_no_system_turn() -> None:
    """No system message may appear — it would break chat-template parity."""
    ex = to_chat_example("memo", {"rating": "HOLD", "confidence": 0.5, "rationale": "x", "risk_factors": ["a", "b", "c"]})
    roles = [m["role"] for m in ex.messages]
    assert "system" not in roles


@pytest.mark.unit
def test_to_chat_example_user_matches_prompt() -> None:
    """User message (first turn) must equal build_analyst_user_prompt output."""
    memo = "公司财务稳健，成长性良好。"
    feedback = "补充 ESG 风险"
    gold = {"rating": "BUY", "confidence": 0.7, "rationale": "x", "risk_factors": ["a", "b", "c"]}
    ex = to_chat_example(memo, gold, risk_feedback=feedback)
    expected_user = build_analyst_user_prompt(memo, risk_feedback=feedback)
    assert ex.messages[0]["content"] == expected_user


@pytest.mark.unit
def test_to_chat_example_assistant_is_json_dumps() -> None:
    """Assistant content must be json.dumps(gold_json, ensure_ascii=False)."""
    gold = {"rating": "SELL", "confidence": 0.4, "rationale": "熊市信号", "risk_factors": ["r1", "r2", "r3"]}
    ex = to_chat_example("memo", gold)
    assert ex.messages[1]["content"] == json.dumps(gold, ensure_ascii=False)


@pytest.mark.unit
def test_write_read_jsonl_roundtrip(tmp_path: Path) -> None:
    """write_jsonl followed by read_jsonl must return equivalent data."""
    gold1 = {"rating": "BUY", "confidence": 0.8, "rationale": "x", "risk_factors": ["a", "b", "c"]}
    gold2 = {"rating": "HOLD", "confidence": 0.5, "rationale": "y", "risk_factors": ["d", "e", "f"]}
    ex1 = to_chat_example("memo one", gold1)
    ex2 = to_chat_example("memo two", gold2, risk_feedback="prior feedback")

    out_path = tmp_path / "test.jsonl"
    write_jsonl(out_path, [ex1, ex2])

    rows = read_jsonl(out_path)
    assert len(rows) == 2

    # Each row should round-trip back to an equivalent SFTExample.
    for row, original in zip(rows, [ex1, ex2]):
        assert "messages" in row
        assert len(row["messages"]) == len(original.messages)
        for loaded_msg, orig_msg in zip(row["messages"], original.messages):
            assert loaded_msg["role"] == orig_msg["role"]
            assert loaded_msg["content"] == orig_msg["content"]


@pytest.mark.unit
def test_write_jsonl_creates_parent_directories(tmp_path: Path) -> None:
    """write_jsonl must create parent directories if they do not exist."""
    deep_path = tmp_path / "a" / "b" / "c" / "examples.jsonl"
    ex = to_chat_example("memo", {"rating": "HOLD", "confidence": 0.5, "rationale": "x", "risk_factors": ["a", "b", "c"]})
    write_jsonl(deep_path, [ex])  # should not raise
    assert deep_path.exists()


@pytest.mark.unit
def test_sft_example_is_frozen() -> None:
    """SFTExample must be immutable (frozen dataclass)."""
    ex = to_chat_example("memo", {"rating": "BUY", "confidence": 0.9, "rationale": "x", "risk_factors": ["a", "b", "c"]})
    with pytest.raises((AttributeError, TypeError)):
        ex.messages = ()  # type: ignore[misc]
