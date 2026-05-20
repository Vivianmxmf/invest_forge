"""Domain dataclasses + enums shared across InvestForge modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any


class Rating(str, Enum):
    """Investment rating returned by the analyst node."""

    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"

    @classmethod
    def parse(cls, raw: str) -> "Rating":
        """Lenient parser: accepts CN labels too ('买入'/'持有'/'卖出')."""
        if raw is None:
            return cls.HOLD
        s = str(raw).strip().upper()
        if s in cls._value2member_map_:
            return cls(s)
        cn = {"买入": cls.BUY, "持有": cls.HOLD, "观望": cls.HOLD, "卖出": cls.SELL}
        if s in cn:
            return cn[s]
        # Soft fall-back: any "BUY"/"SELL" prefix wins
        if s.startswith("BUY"):
            return cls.BUY
        if s.startswith("SELL"):
            return cls.SELL
        return cls.HOLD


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class FinancialMetrics:
    """A snapshot of fundamentals for a single fiscal year."""

    ts_code: str
    fiscal_year: int
    revenue: float           # in CNY
    net_profit: float
    gross_margin: float      # 0–1
    net_margin: float
    roe: float
    roic: float
    debt_ratio: float
    free_cash_flow: float
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    market_cap: float | None = None


@dataclass(frozen=True)
class NewsItem:
    title: str
    published: date
    source: str
    sentiment: float           # -1 .. +1
    label: str                 # "positive" / "neutral" / "negative"
    summary: str | None = None


@dataclass(frozen=True)
class MacroContext:
    cpi_yoy: float
    ppi_yoy: float
    m1_yoy: float
    m2_yoy: float
    pmi: float
    us10y_yield: float
    cny_dxy: float
    summary: str               # short narrative the LLM can quote


@dataclass(frozen=True)
class Recommendation:
    rating: Rating
    confidence: float          # 0–1
    rationale: str
    risk_factors: tuple[str, ...] = field(default_factory=tuple)
    target_price: float | None = None


@dataclass(frozen=True)
class RAGHit:
    """A single document chunk returned by hybrid retrieval."""

    text: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
