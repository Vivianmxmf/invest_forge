"""Sentiment scoring for financial news.

Two backends:
  * ``LexiconSentiment`` — no-dependency Chinese/English keyword scorer; used
    by tests and the offline laptop path.
  * ``FinBertSentiment`` — wraps ``ProsusAI/finbert`` (or a CN equivalent);
    loaded lazily, never imported at module-import time so the package
    stays light on machines without ``transformers``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from invest_forge.common.logging_setup import get_logger

logger = get_logger(__name__)


@runtime_checkable
class Sentiment(Protocol):
    def score(self, text: str) -> tuple[float, str]: ...


# ───────────────────── Lexicon (offline) ─────────────────────

_POSITIVE_WORDS = {
    # cn
    "增长", "上涨", "突破", "盈利", "改善", "扩张", "新高", "向好", "复苏",
    "利好", "上调", "买入", "推荐", "强劲", "超预期",
    # en
    "growth", "increase", "beat", "buy", "outperform", "strong", "improve",
    "expansion", "record", "surge", "rally",
}

_NEGATIVE_WORDS = {
    "下滑", "亏损", "下跌", "下调", "回落", "承压", "风险", "退市", "减持",
    "卖出", "不及预期", "诉讼", "罚款", "处罚",
    "miss", "decline", "drop", "lawsuit", "downgrade", "loss", "underperform",
    "sell", "weak", "selloff", "plunge",
}

_TOKEN_RE = re.compile(r"[A-Za-z一-龥]+", re.UNICODE)


@dataclass
class LexiconSentiment:
    name: str = "lexicon"

    def score(self, text: str) -> tuple[float, str]:
        if not text:
            return 0.0, "neutral"
        tokens = _TOKEN_RE.findall(text)
        if not tokens:
            return 0.0, "neutral"
        pos = sum(1 for t in tokens if t in _POSITIVE_WORDS or any(p in t for p in _POSITIVE_WORDS))
        neg = sum(1 for t in tokens if t in _NEGATIVE_WORDS or any(n in t for n in _NEGATIVE_WORDS))
        if pos == 0 and neg == 0:
            return 0.0, "neutral"
        # Normalised signed score in [-1, +1]
        denom = max(pos + neg, 1)
        signed = (pos - neg) / denom
        label = "positive" if signed > 0.15 else ("negative" if signed < -0.15 else "neutral")
        return float(signed), label


# ───────────────────── FinBERT (server) ─────────────────────

@dataclass
class FinBertSentiment:
    model_id: str = "ProsusAI/finbert"
    name: str = "finbert"

    def __post_init__(self) -> None:
        from transformers import pipeline  # lazy

        self._pipe = pipeline("text-classification", model=self.model_id)

    def score(self, text: str) -> tuple[float, str]:
        if not text:
            return 0.0, "neutral"
        result = self._pipe(text[:512])[0]
        label = result["label"].lower()
        prob = float(result["score"])
        signed = prob if label == "positive" else (-prob if label == "negative" else 0.0)
        return signed, label


# ───────────────────── Factory ─────────────────────

def build_sentiment(prefer_real: bool = False) -> Sentiment:
    if not prefer_real:
        return LexiconSentiment()
    try:
        return FinBertSentiment()
    except Exception as exc:  # noqa: BLE001 - any import / weight failure → fallback
        logger.warning("FinBERT unavailable (%s); falling back to lexicon", exc)
        return LexiconSentiment()
