"""Offline tests for the pure-pandas price-panel helpers (W4).

These tests use the real ``data/sample/prices.csv`` (5 tickers × 261 days)
and require NO alphalens installation — they run on the laptop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from invest_forge.tools.backtest_tools import (
    build_factor_panel,
    load_price_panel,
    make_lookahead_safe_signal,
    portfolio_long_short,
)

# Resolve the sample CSV relative to the project root.
_SAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "sample" / "prices.csv"


# ── load_price_panel ──────────────────────────────────────────────────────────

@pytest.mark.unit
def test_load_price_panel_shape() -> None:
    """Wide panel: DatetimeIndex, 5 ticker columns, ~261 rows."""
    panel = load_price_panel(_SAMPLE_CSV)

    # Index must be a DatetimeIndex.
    assert isinstance(panel.index, pd.DatetimeIndex), (
        f"Expected DatetimeIndex, got {type(panel.index)}"
    )

    # Exactly 5 tickers (300750.SZ, 600276.SH, 600519.SH, 601398.SH, 688981.SH).
    assert panel.shape[1] == 5, f"Expected 5 ticker columns, got {panel.shape[1]}"

    # At least 250 trading days in 2024 (usually 261 in the sample).
    assert len(panel) >= 250, f"Expected ≥250 rows, got {len(panel)}"

    # No all-NaN columns.
    assert not panel.isna().all(axis=0).any(), "Found all-NaN ticker column(s)"

    # Sorted ascending.
    assert panel.index.is_monotonic_increasing, "DatetimeIndex is not sorted ascending"


@pytest.mark.unit
def test_load_price_panel_expected_tickers() -> None:
    """Verify the expected tickers are present as columns."""
    panel = load_price_panel(_SAMPLE_CSV)
    expected = {"300750.SZ", "600276.SH", "600519.SH", "601398.SH", "688981.SH"}
    assert set(panel.columns) == expected, (
        f"Unexpected columns: {set(panel.columns) - expected}"
    )


# ── build_factor_panel ────────────────────────────────────────────────────────

@pytest.mark.unit
def test_build_factor_panel_multiindex() -> None:
    """Factor is a Series with a 2-level (date, asset) MultiIndex."""
    panel = load_price_panel(_SAMPLE_CSV)
    factor = build_factor_panel(panel, window=20)

    assert isinstance(factor, pd.Series), "build_factor_panel must return a pd.Series"
    assert factor.index.nlevels == 2, (
        f"Expected 2-level MultiIndex, got {factor.index.nlevels} levels"
    )
    assert factor.index.names == ["date", "asset"], (
        f"Expected names ['date', 'asset'], got {factor.index.names}"
    )

    # Level-0 (date) should be datetime dtype.
    date_level = factor.index.get_level_values("date")
    assert hasattr(date_level, "dtype") and np.issubdtype(
        date_level.dtype, np.datetime64
    ), f"date level dtype is not datetime64: {date_level.dtype}"

    # No NaN values (dropna applied).
    assert not factor.isna().any(), "Factor contains NaN values after dropna"

    # Non-empty.
    assert len(factor) > 0, "Factor Series is empty"


@pytest.mark.unit
def test_build_factor_panel_lookahead_safe() -> None:
    """Factor value at date t must equal make_lookahead_safe_signal(col)[t]."""
    panel = load_price_panel(_SAMPLE_CSV)
    factor = build_factor_panel(panel, window=20)

    # Pick a specific ticker and spot-check a date well inside the series.
    ticker = panel.columns[0]
    expected_signal = make_lookahead_safe_signal(panel[ticker], window=20)

    # Grab a date where both the panel factor and expected signal are non-NaN.
    factor_for_ticker = factor.xs(ticker, level="asset")
    common_dates = factor_for_ticker.index.intersection(expected_signal.dropna().index)
    assert len(common_dates) > 0, "No common dates found for spot-check"

    spot_date = common_dates[len(common_dates) // 2]
    panel_val = factor_for_ticker.loc[spot_date]
    expected_val = expected_signal.loc[spot_date]

    assert pytest.approx(panel_val, rel=1e-9) == expected_val, (
        f"Factor mismatch at {spot_date}: panel={panel_val}, expected={expected_val}"
    )


# ── portfolio_long_short on real sample ──────────────────────────────────────

@pytest.mark.unit
def test_portfolio_long_short_on_sample() -> None:
    """Full pipeline on the real 5-ticker sample produces finite metrics."""
    panel = load_price_panel(_SAMPLE_CSV)
    factor_wide = panel.apply(lambda col: make_lookahead_safe_signal(col, 20))
    result = portfolio_long_short(panel, factor_wide, top_n=2)

    assert result.n_obs > 0, "n_obs must be > 0"
    assert np.isfinite(result.ic_mean), f"ic_mean not finite: {result.ic_mean}"
    assert np.isfinite(result.ic_ir), f"ic_ir not finite: {result.ic_ir}"
    assert np.isfinite(result.annualised_return), (
        f"annualised_return not finite: {result.annualised_return}"
    )
    assert np.isfinite(result.sharpe), f"Sharpe not finite: {result.sharpe}"
    assert np.isfinite(result.max_drawdown), (
        f"max_drawdown not finite: {result.max_drawdown}"
    )
    # Max-drawdown should be ≤ 0.
    assert result.max_drawdown <= 0.0, (
        f"max_drawdown should be ≤ 0, got {result.max_drawdown}"
    )
