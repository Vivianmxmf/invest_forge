"""Lightweight backtest helpers — IC / IR / Sharpe / MaxDrawdown.

We intentionally do NOT pull in qlib / alphalens at import time; the
``run_simple_backtest`` function uses only pandas + numpy so it works on a
laptop with the synthetic sample dataset.  Server deployments can call the
``alphalens_report`` adapter, which lazy-imports alphalens.

All signal-construction helpers are *lookahead-bias safe*: positional
signals at row ``t`` are shifted before computing the realised return at
``t``, so the same-day leak from JD6 §4 Day 2 cannot happen.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


# ───────────────────── Pure-pandas backtest ─────────────────────

@dataclass(frozen=True)
class BacktestResult:
    ic_mean: float
    ic_ir: float
    annualised_return: float
    sharpe: float
    max_drawdown: float
    n_obs: int


def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-12 else 0.0


def make_lookahead_safe_signal(prices: pd.Series, window: int = 20) -> pd.Series:
    """Momentum signal that does NOT leak the current day's price.

    Signal at day ``t`` = price[t-1] / rolling_mean(price[t-window:t]),
    expressed as the natural form ``price / MA(window)`` then shifted by 1.
    """
    ma = prices.rolling(window=window, min_periods=max(window // 2, 1)).mean()
    return (prices / ma).shift(1)


def realised_return(prices: pd.Series) -> pd.Series:
    """Day-over-day return realised at row ``t`` (uses prices through ``t``)."""
    return prices.pct_change()


def run_simple_backtest(
    prices: pd.Series,
    signal: pd.Series | None = None,
    *,
    window: int = 20,
) -> BacktestResult:
    """Long-when-signal>1 / short-when-signal<1 daily strategy on one stock.

    Returns IC / IR / annualised return / Sharpe / max drawdown.  Trading
    cost is not modelled — keep it minimal.
    """
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    if signal is None:
        signal = make_lookahead_safe_signal(prices, window=window)
    aligned = pd.concat([signal.rename("sig"), realised_return(prices).rename("ret")], axis=1).dropna()
    if aligned.empty:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    # IC = Pearson(signal, next-day return).  We already shifted signal, so
    # aligned ``sig`` at row t pairs with realised return at row t.
    ic = float(aligned["sig"].corr(aligned["ret"]))
    ic_std = float(aligned["sig"].rolling(60, min_periods=20).corr(aligned["ret"]).std() or 0.0)

    position = np.where(aligned["sig"] > 1, 1.0, np.where(aligned["sig"] < 1, -1.0, 0.0))
    pnl = position * aligned["ret"].to_numpy()
    equity = np.cumprod(1.0 + pnl)
    n = len(pnl)
    ann_ret = float(equity[-1] ** (252.0 / max(n, 1)) - 1) if n else 0.0
    pnl_std = float(np.std(pnl))
    sharpe = float(_safe_div(np.mean(pnl), pnl_std) * np.sqrt(252)) if pnl_std else 0.0
    running_max = np.maximum.accumulate(equity) if n else np.array([1.0])
    mdd = float(np.min(equity / running_max - 1)) if n else 0.0

    return BacktestResult(
        ic_mean=ic if not np.isnan(ic) else 0.0,
        ic_ir=_safe_div(ic, ic_std) if not np.isnan(ic) else 0.0,
        annualised_return=ann_ret,
        sharpe=sharpe,
        max_drawdown=mdd,
        n_obs=int(n),
    )


def portfolio_long_short(
    panel_prices: pd.DataFrame,
    panel_signal: pd.DataFrame,
    *,
    top_n: int = 5,
) -> BacktestResult:
    """Cross-sectional long-top / short-bottom backtest.

    ``panel_prices`` / ``panel_signal`` share the same ``date × ts_code``
    layout (rows are dates, columns are tickers).  Signals are aligned by
    column.  Same-day leak is prevented by shifting the signal one row.
    """
    if panel_prices.shape[1] == 0 or panel_signal.shape[1] == 0:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    sig = panel_signal.shift(1)            # use yesterday's signal
    ret = panel_prices.pct_change()
    common = panel_prices.columns.intersection(panel_signal.columns)
    if len(common) == 0:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    sig, ret = sig[common], ret[common]

    # Clamp top_n so the long + short buckets stay disjoint.  In a 5-name
    # universe with the default top_n=5 every column ranks into both
    # buckets, the weights cancel and the backtest silently reports zero.
    max_each_side = max(1, len(common) // 2)
    effective_n = min(top_n, max_each_side)
    long_mask = sig.rank(axis=1, ascending=False) <= effective_n
    short_mask = sig.rank(axis=1, ascending=True) <= effective_n
    weights = long_mask.astype(float) - short_mask.astype(float)
    weights = weights.div(weights.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    pnl_series = (weights * ret).sum(axis=1).dropna()
    if pnl_series.empty:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    equity = (1.0 + pnl_series).cumprod()
    n = len(pnl_series)
    ann_ret = float(equity.iloc[-1] ** (252.0 / max(n, 1)) - 1)
    pnl_std = float(pnl_series.std())
    sharpe = float(_safe_div(pnl_series.mean(), pnl_std) * np.sqrt(252)) if pnl_std else 0.0
    running_max = equity.cummax()
    mdd = float((equity / running_max - 1).min())

    # IC computed per-row (cross-sectional correlation) and averaged.
    per_day_ic = sig.corrwith(ret, axis=1)
    ic_mean = float(per_day_ic.mean()) if not per_day_ic.empty else 0.0
    ic_ir = _safe_div(ic_mean, float(per_day_ic.std() or 0.0))

    return BacktestResult(
        ic_mean=ic_mean if not np.isnan(ic_mean) else 0.0,
        ic_ir=ic_ir,
        annualised_return=ann_ret,
        sharpe=sharpe,
        max_drawdown=mdd,
        n_obs=n,
    )


def load_price_panel(csv_path: str | Path) -> pd.DataFrame:
    """Read the long-format prices CSV and pivot ``close`` to a wide panel.

    Returns a ``pd.DataFrame`` with:
    - index: ``pd.DatetimeIndex`` named ``"trade_date"``, sorted ascending.
    - columns: ``ts_code`` strings, sorted.
    - values: daily close prices (float).  All-NaN columns are dropped.

    alphalens requires a DatetimeIndex; this function always converts.
    """
    df = pd.read_csv(csv_path, parse_dates=["trade_date"])
    panel = (
        df.pivot(index="trade_date", columns="ts_code", values="close")
        .sort_index()
    )
    panel.index = pd.DatetimeIndex(panel.index, name="trade_date")
    panel = panel.dropna(axis=1, how="all")
    return panel


def build_factor_panel(
    panel_prices: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.Series:
    """Build an alphalens-compatible factor Series from a wide price panel.

    Applies ``make_lookahead_safe_signal(col, window)`` to every ticker column
    (already lookahead-safe: signal at ``t`` uses prices up to ``t-1``), then
    stacks the result into a **MultiIndex Series** with index levels
    ``("date", "asset")``.  NaN rows are dropped.

    Parameters
    ----------
    panel_prices:
        Wide DataFrame (DatetimeIndex × tickers) as returned by
        ``load_price_panel``.
    window:
        Rolling-MA window passed to ``make_lookahead_safe_signal``.

    Returns
    -------
    pd.Series with a 2-level MultiIndex ``(date, asset)`` and float values.
    """
    wide = panel_prices.apply(lambda col: make_lookahead_safe_signal(col, window))
    # Stack wide → long MultiIndex; level order after stack is (row, col).
    stacked = wide.stack(future_stack=True).dropna()
    stacked.index.names = ["date", "asset"]
    return stacked


def alphalens_report(
    factor: pd.Series,
    prices: pd.DataFrame,
    *,
    periods: Iterable[int] = (1, 5, 20),
    quantiles: int = 2,
    out_png: str | Path | None = None,
) -> dict[str, Any]:  # pragma: no cover - alphalens lines; exercised on server only
    """Generate alphalens analytics and return a data dict (server-only).

    Parameters
    ----------
    factor:
        MultiIndex Series ``(date, asset)`` — use ``build_factor_panel``.
    prices:
        Wide DataFrame (DatetimeIndex × tickers) — use ``load_price_panel``.
    periods:
        Forward-return horizons in trading days.
    quantiles:
        Number of quantile buckets.  Defaults to 2 for small universes (≤5
        assets) to avoid duplicate bin-edge errors.
    out_png:
        If given, attempt to save a full tear-sheet PNG here.  A tiny-universe
        failure is caught and only logged — data is still returned.

    Returns
    -------
    dict with keys:
    - ``"mean_ic"``: ``{period_str: float}`` — per-period mean IC.
    - ``"mean_return_by_quantile"``: ``{period_str: {quantile_str: float}}``.
    - ``"n_obs"``: int — number of (date, asset) observations.
    """
    # All alphalens imports are lazy so the module is usable without alphalens.
    from alphalens.utils import get_clean_factor_and_forward_returns  # pragma: no cover
    import alphalens.performance as alperf  # pragma: no cover

    periods_tuple = tuple(periods)  # pragma: no cover
    factor_data = get_clean_factor_and_forward_returns(  # pragma: no cover
        factor, prices, periods=periods_tuple, quantiles=quantiles
    )

    # ── Mean IC per forward-return period ─────────────────────────────────
    ic_series = alperf.factor_information_coefficient(factor_data).mean()  # pragma: no cover
    mean_ic: dict[str, float] = {str(k): float(v) for k, v in ic_series.items()}  # pragma: no cover

    # ── Mean return by quantile ────────────────────────────────────────────
    mean_ret_raw, _ = alperf.mean_return_by_quantile(factor_data)  # pragma: no cover
    mean_return_by_quantile: dict[str, dict[str, float]] = {}  # pragma: no cover
    for col in mean_ret_raw.columns:  # pragma: no cover
        mean_return_by_quantile[str(col)] = {  # pragma: no cover
            str(q): float(v) for q, v in mean_ret_raw[col].items()  # pragma: no cover
        }  # pragma: no cover

    result: dict[str, Any] = {  # pragma: no cover
        "mean_ic": mean_ic,  # pragma: no cover
        "mean_return_by_quantile": mean_return_by_quantile,  # pragma: no cover
        "n_obs": int(len(factor_data)),  # pragma: no cover
    }  # pragma: no cover

    # ── Optional PNG tear-sheet (best-effort, non-fatal) ──────────────────
    if out_png is not None:  # pragma: no cover
        import matplotlib  # pragma: no cover
        matplotlib.use("Agg")  # must be set BEFORE importing pyplot  # pragma: no cover
        import matplotlib.pyplot as plt  # pragma: no cover
        from alphalens.tears import create_full_tear_sheet  # pragma: no cover
        try:  # pragma: no cover
            create_full_tear_sheet(factor_data)  # pragma: no cover
            plt.savefig(out_png, bbox_inches="tight", dpi=150)  # pragma: no cover
            plt.close("all")  # pragma: no cover
        except Exception as exc:  # pragma: no cover
            _log.warning(  # pragma: no cover
                "alphalens tear-sheet PNG generation failed (likely tiny universe): %s", exc  # pragma: no cover
            )  # pragma: no cover
            plt.close("all")  # pragma: no cover

    return result  # pragma: no cover
