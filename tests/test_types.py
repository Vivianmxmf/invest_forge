"""Tests for domain enums + dataclasses."""
from __future__ import annotations

import pytest

from janus_terminal.common.types import Rating, RiskLevel


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BUY", Rating.BUY),
        ("buy", Rating.BUY),
        ("HOLD", Rating.HOLD),
        ("SELL", Rating.SELL),
        ("买入", Rating.BUY),
        ("持有", Rating.HOLD),
        ("观望", Rating.HOLD),
        ("卖出", Rating.SELL),
        ("BUY_with_conviction", Rating.BUY),
        ("unknown_label", Rating.HOLD),
        ("", Rating.HOLD),
        (None, Rating.HOLD),
    ],
)
def test_rating_parse(raw, expected):
    assert Rating.parse(raw) == expected


@pytest.mark.unit
def test_risk_level_enum():
    assert RiskLevel("LOW") == RiskLevel.LOW
    with pytest.raises(ValueError):
        RiskLevel("ABSURD")
