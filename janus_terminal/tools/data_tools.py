"""Data-layer tools: fundamentals, macro, news.

All providers are wrapped in a lightweight ``DataProvider`` protocol so
tests inject a fake (``FakeDataProvider``) without any network or token.
Production builds inject ``TushareProvider`` / ``AkshareProvider``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

from janus_terminal.common.config import get_settings
from janus_terminal.common.logging_setup import get_logger
from janus_terminal.common.types import FinancialMetrics, MacroContext, NewsItem

logger = get_logger(__name__)


# ───────────────────── Protocol ─────────────────────

@runtime_checkable
class DataProvider(Protocol):
    name: str

    def fundamentals(self, ts_code: str) -> FinancialMetrics: ...
    def news(self, ts_code: str, *, limit: int = 20) -> list[NewsItem]: ...
    def macro(self) -> MacroContext: ...


def _parse_tushare_date(raw: str) -> date:
    """Parse Tushare's various date encodings into ``datetime.date``.

    Tushare returns ``YYYYMMDD`` (most common), ``YYYY-MM-DD``, and sometimes
    ``YYYY/MM/DD``.  ``date.fromisoformat`` only accepts ``YYYY-MM-DD`` on
    Python 3.10 (3.11+ is laxer), so normalise first.
    """
    from datetime import datetime

    s = (raw or "").strip()[:19]
    if not s or s.lower() in {"nan", "none"}:
        return date.today()
    # Replace separators so a YYYY-MM-DD or YYYY/MM/DD remains valid.
    s = s.replace("/", "-")
    # 8-digit compact form (YYYYMMDD).
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").date()
    # 10+-char date or datetime.
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        return date.today()


# ───────────────────── Fake provider (offline + tests) ─────────────────────

@dataclass
class FakeDataProvider:
    """Reads from data/sample/ if present; otherwise returns a deterministic
    in-memory snapshot.  Never touches the network.
    """

    sample_dir: Path | None = None
    name: str = "fake"

    def _load_json(self, filename: str) -> dict | None:
        sample_dir = self.sample_dir or get_settings().paths.sample_dir
        path = sample_dir / filename
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def fundamentals(self, ts_code: str) -> FinancialMetrics:
        raw = self._load_json("fundamentals.json") or {}
        row = raw.get(ts_code) or _DEFAULT_FUNDAMENTALS.get(ts_code) or _DEFAULT_FUNDAMENTALS["DEFAULT"]
        return FinancialMetrics(ts_code=ts_code, **row)

    def news(self, ts_code: str, *, limit: int = 20) -> list[NewsItem]:
        raw = self._load_json("news.json") or {}
        items = raw.get(ts_code) or _DEFAULT_NEWS.get(ts_code, _DEFAULT_NEWS["DEFAULT"])
        out: list[NewsItem] = []
        for it in items[:limit]:
            out.append(
                NewsItem(
                    title=it["title"],
                    published=date.fromisoformat(it["published"]),
                    source=it.get("source", "fake"),
                    sentiment=float(it.get("sentiment", 0.0)),
                    label=it.get("label", "neutral"),
                    summary=it.get("summary"),
                )
            )
        return out

    def macro(self) -> MacroContext:
        raw = self._load_json("macro.json") or _DEFAULT_MACRO
        return MacroContext(**raw)


# ───────────────────── Defaults (no sample files present) ─────────────────────

_DEFAULT_FUNDAMENTALS: dict[str, dict] = {
    "DEFAULT": {
        "fiscal_year": 2024,
        "revenue": 4.32e10,
        "net_profit": 6.5e9,
        "gross_margin": 0.285,
        "net_margin": 0.151,
        "roe": 0.17,
        "roic": 0.13,
        "debt_ratio": 0.42,
        "free_cash_flow": 8.9e9,
        "pe_ratio": 22.0,
        "pb_ratio": 3.4,
        "market_cap": 1.43e11,
    },
}

_DEFAULT_NEWS: dict[str, list[dict]] = {
    "DEFAULT": [
        {
            "title": "公司发布 2024 年报, 净利润同比 +28%",
            "published": "2025-03-20",
            "source": "fake",
            "sentiment": 0.42,
            "label": "positive",
            "summary": "营收和净利润双增, 现金流稳健.",
        },
        {
            "title": "行业景气度回暖, 公司订单饱满",
            "published": "2025-04-05",
            "source": "fake",
            "sentiment": 0.31,
            "label": "positive",
            "summary": "下游需求恢复, 排产至 Q3.",
        },
        {
            "title": "海外销售敞口加大, 汇率波动风险",
            "published": "2025-04-18",
            "source": "fake",
            "sentiment": -0.22,
            "label": "negative",
            "summary": "出口占比 38%, 美元走强对盈利构成扰动.",
        },
    ]
}

_DEFAULT_MACRO: dict = {
    "cpi_yoy": 0.4,
    "ppi_yoy": -2.7,
    "m1_yoy": 1.2,
    "m2_yoy": 7.0,
    "pmi": 50.4,
    "us10y_yield": 4.21,
    "cny_dxy": 104.3,
    "summary": "国内 CPI 温和回升, PPI 仍处通缩区间; 制造业 PMI 重回扩张; 海外美元利率高位运行.",
}


# ───────────────────── Tushare provider (server-only, lazy imports) ─────────────────────

@dataclass
class TushareProvider:
    token: str
    name: str = "tushare"
    # Optional: pin a specific fiscal period like "20231231"; default = latest.
    period: str | None = None

    def __post_init__(self) -> None:
        import tushare as ts

        ts.set_token(self.token)
        self._pro = ts.pro_api()

    def fundamentals(self, ts_code: str) -> FinancialMetrics:
        # 1) Margins / ratios → fina_indicator (latest *annual*).
        if self.period:
            ind = self._pro.fina_indicator(ts_code=ts_code, period=self.period)
        else:
            ind = self._pro.fina_indicator(ts_code=ts_code)
            if ind is not None and not ind.empty:
                # Tushare returns Q1/Q2/Q3 + annual rows; downstream prompts
                # treat these as a fiscal-year snapshot.  Keep only annual
                # reports (end_date ending in '1231') so we never expose
                # a quarterly figure as annual.
                annual = ind[ind["end_date"].astype(str).str.endswith("1231")]
                ind = (annual if not annual.empty else ind).sort_values(
                    "end_date", ascending=False
                ).head(1)
        if ind is None or ind.empty:
            raise RuntimeError(f"Tushare fina_indicator empty for {ts_code}")
        row = ind.iloc[0]
        period_str = str(row.get("end_date", "20241231"))

        # 2) Absolute revenue / net_profit → income statement.
        # fina_indicator does NOT reliably expose absolute amounts; the income
        # endpoint is the canonical source.  Both report values in 元.
        inc_row: dict = {}
        try:
            inc = self._pro.income(ts_code=ts_code, period=period_str)
            if inc is not None and not inc.empty:
                inc_row = inc.iloc[0].to_dict()
        except Exception:  # noqa: BLE001 - older tokens lack income access
            inc_row = {}

        # 3) Free cash flow → cash-flow statement (op cf − capex).
        cfs_row: dict = {}
        try:
            cfs = self._pro.cashflow(ts_code=ts_code, period=period_str)
            if cfs is not None and not cfs.empty:
                cfs_row = cfs.iloc[0].to_dict()
        except Exception:  # noqa: BLE001
            cfs_row = {}
        op_cf = float(cfs_row.get("n_cashflow_act") or 0)
        capex = float(cfs_row.get("c_pay_acq_const_fiolta") or 0)
        fcf = op_cf - capex

        # 4) Market cap / PE / PB → daily_basic.
        b: dict = {}
        try:
            basic = self._pro.daily_basic(ts_code=ts_code, limit=1)
            if basic is not None and not basic.empty:
                b = basic.iloc[0].to_dict()
        except Exception:  # noqa: BLE001
            b = {}

        def _absolute(*names: str) -> float:
            for n in names:
                v = inc_row.get(n)
                if v not in (None, 0, "0", ""):
                    return float(v)
            return 0.0

        return FinancialMetrics(
            ts_code=ts_code,
            fiscal_year=int(period_str[:4]),
            revenue=_absolute("revenue", "total_revenue", "oper_inc"),
            net_profit=_absolute("n_income", "net_profit"),
            gross_margin=float(row.get("grossprofit_margin", 0) or 0) / 100.0,
            net_margin=float(row.get("netprofit_margin", 0) or 0) / 100.0,
            roe=float(row.get("roe", 0) or 0) / 100.0,
            roic=float(row.get("roic", 0) or 0) / 100.0,
            debt_ratio=float(row.get("debt_to_assets", 0) or 0) / 100.0,
            free_cash_flow=fcf,
            pe_ratio=(float(b.get("pe") or 0) or None),
            pb_ratio=(float(b.get("pb") or 0) or None),
            # daily_basic reports total_mv in 万元.
            market_cap=(float(b.get("total_mv") or 0) * 1e4 if b.get("total_mv") else None),
        )

    def news(self, ts_code: str, *, limit: int = 20) -> list[NewsItem]:
        """Per-ticker disclosures via ``anns``; fall back to the global Sina
        news feed *filtered* by ts_code substring when no announcements found.
        """
        end = date.today()
        start = end - timedelta(days=30)

        # Prefer per-stock announcements (anns) — exchange filings keyed by ts_code.
        items: list[NewsItem] = []
        try:
            anns = self._pro.anns(
                ts_code=ts_code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )
        except Exception:  # noqa: BLE001 - older tokens may lack anns access
            anns = None
        if anns is not None and not anns.empty:
            for _, r in anns.head(limit).iterrows():
                items.append(
                    NewsItem(
                        title=str(r.get("title", "")),
                        published=_parse_tushare_date(str(r.get("ann_date"))),
                        source="anns",
                        sentiment=0.0,
                        label="neutral",
                        summary=str(r.get("content", ""))[:200],
                    )
                )
            return items

        # Fall back to global Sina news, FILTERED to the ts_code or its bare
        # 6-digit symbol so unrelated headlines don't pollute sentiment.
        df = self._pro.news(src="sina", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
        if df is None or df.empty:
            return items
        symbol = ts_code.split(".", 1)[0]
        mask = df.apply(
            lambda r: symbol in str(r.get("title", "")) or symbol in str(r.get("content", "")),
            axis=1,
        )
        df = df[mask]
        for _, r in df.head(limit).iterrows():
            items.append(
                NewsItem(
                    title=str(r.get("title", "")),
                    published=_parse_tushare_date(str(r.get("datetime"))),
                    source="sina",
                    sentiment=0.0,
                    label="neutral",
                    summary=str(r.get("content", ""))[:200],
                )
            )
        return items

    def macro(self) -> MacroContext:  # pragma: no cover - thin wrapper
        raise NotImplementedError("Macro pulls go through AkshareProvider; compose them at the agent layer.")


# ───────────────────── Akshare provider (macro only — Tushare lacks it) ─────────────────────

@dataclass
class AkshareMacroProvider:
    """Pulls macro indicators from AKShare. Fundamentals/news are not supported here."""

    name: str = "akshare_macro"

    def fundamentals(self, ts_code: str) -> FinancialMetrics:  # pragma: no cover
        raise NotImplementedError("AkshareMacroProvider only serves macro; compose with Tushare or Fake.")

    def news(self, ts_code: str, *, limit: int = 20) -> list[NewsItem]:  # pragma: no cover
        return []

    def macro(self) -> MacroContext:  # pragma: no cover - network call
        import akshare as ak

        # NOTE: MacroContext stores YoY *percentage points* (e.g. 0.4 means
        # +0.4 %), NOT decimal fractions.  AKShare's 今值 is also reported as
        # a percentage point, so we keep the value as-is.
        try:
            cpi = ak.macro_china_cpi_yearly()
            cpi_val = float(cpi.iloc[-1]["今值"]) if "今值" in cpi.columns else 0.0
        except Exception:
            cpi_val = 0.0
        try:
            pmi = ak.macro_china_pmi_yearly()
            pmi_val = float(pmi.iloc[-1]["今值"]) if "今值" in pmi.columns else 50.0
        except Exception:
            pmi_val = 50.0
        return MacroContext(
            cpi_yoy=cpi_val, ppi_yoy=0.0, m1_yoy=0.0, m2_yoy=0.0, pmi=pmi_val,
            us10y_yield=0.0, cny_dxy=0.0,
            summary=f"AKShare CPI YoY={cpi_val:.2f}pp PMI={pmi_val:.1f}",
        )


# ───────────────────── Composite (Tushare fundamentals + AKShare macro + fallback) ─────────────────────

@dataclass
class CompositeProvider:
    """Routes each method to a provider that can satisfy it.

    fundamentals/news → primary  (e.g. Tushare)
    macro              → secondary (e.g. AKShare) or tertiary (Fake)
    """

    fundamentals_news: "DataProvider"
    macro_provider: "DataProvider"
    name: str = "composite"

    def fundamentals(self, ts_code: str) -> FinancialMetrics:
        return self.fundamentals_news.fundamentals(ts_code)

    def news(self, ts_code: str, *, limit: int = 20) -> list[NewsItem]:
        return self.fundamentals_news.news(ts_code, limit=limit)

    def macro(self) -> MacroContext:
        try:
            return self.macro_provider.macro()
        except NotImplementedError:
            return FakeDataProvider().macro()


# ───────────────────── Factory ─────────────────────

def build_provider(prefer_real: bool = False) -> DataProvider:
    """Return a real provider when configured; fall back to fake.

    With ``prefer_real=True`` and ``TUSHARE_TOKEN`` present, we compose:
        fundamentals / news ← Tushare
        macro               ← AKShare (best-effort) → Fake fallback
    """
    settings = get_settings()
    if prefer_real and settings.tushare_token:
        primary = TushareProvider(token=settings.tushare_token)
        try:
            secondary = AkshareMacroProvider()
        except Exception:  # noqa: BLE001 - AKShare import errors etc.
            secondary = FakeDataProvider()
        return CompositeProvider(fundamentals_news=primary, macro_provider=secondary)
    return FakeDataProvider()


def fetch_for_codes(provider: DataProvider, ts_codes: Iterable[str]) -> dict[str, FinancialMetrics]:
    return {code: provider.fundamentals(code) for code in ts_codes}
