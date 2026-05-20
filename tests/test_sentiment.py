"""Lexicon sentiment scorer tests."""
from __future__ import annotations

import pytest

from invest_forge.tools.sentiment import LexiconSentiment


@pytest.mark.unit
@pytest.mark.parametrize(
    "text,expected_label",
    [
        ("公司 Q1 净利润超预期, 业绩强劲增长", "positive"),
        ("公司业绩下滑, 亏损扩大", "negative"),
        ("公司发布年报", "neutral"),
        ("strong earnings beat estimates", "positive"),
        ("missed estimates, weak guidance", "negative"),
        ("", "neutral"),
    ],
)
def test_lexicon_labels(text, expected_label):
    sentiment = LexiconSentiment()
    score, label = sentiment.score(text)
    assert label == expected_label
    if expected_label == "positive":
        assert score > 0
    elif expected_label == "negative":
        assert score < 0
    else:
        assert -0.16 <= score <= 0.16


@pytest.mark.unit
def test_lexicon_no_tokens():
    sentiment = LexiconSentiment()
    score, label = sentiment.score("...!!!???")
    assert score == 0.0 and label == "neutral"
