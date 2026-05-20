"""Backtest helpers tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from invest_forge.tools.backtest_tools import (
    make_lookahead_safe_signal,
    portfolio_long_short,
    run_simple_backtest,
)


def _trending_series(n: int = 200, drift: float = 0.001, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.01, n)
    prices = 100 * np.exp(np.cumsum(rets))
    return pd.Series(prices)


@pytest.mark.unit
def test_signal_is_lookahead_safe():
    prices = _trending_series()
    signal = make_lookahead_safe_signal(prices, window=20)
    # At row t, the signal must NOT use prices[t].
    # We test this by verifying that shifting prices forward by 1 doesn't
    # change the signal at the *same* index t (i.e. signal depends on past only).
    shifted_prices = prices.shift(0).copy()
    shifted_prices.iloc[10] = shifted_prices.iloc[10] * 1.5  # perturb day 10
    new_signal = make_lookahead_safe_signal(shifted_prices, window=20)
    # signal at index 10 must remain unchanged (it depends on indices < 10).
    assert pytest.approx(signal.iloc[10], rel=1e-6) == new_signal.iloc[10]


@pytest.mark.unit
def test_run_simple_backtest_outputs():
    prices = _trending_series(seed=7)
    result = run_simple_backtest(prices)
    assert result.n_obs > 0
    # IC and Sharpe should be finite
    assert np.isfinite(result.sharpe)
    assert np.isfinite(result.ic_mean)
    assert -1.0 <= result.max_drawdown <= 0.0


@pytest.mark.unit
def test_run_simple_backtest_empty_input():
    result = run_simple_backtest(pd.Series(dtype=float))
    assert result.n_obs == 0
    assert result.sharpe == 0.0


@pytest.mark.unit
def test_portfolio_long_short():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2024-01-02", periods=120, freq="B")
    codes = [f"X{i}" for i in range(10)]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, (len(dates), len(codes))), axis=0)),
        index=dates, columns=codes,
    )
    # Use *lagged* prices as the signal so it's defensible.
    signal = prices.shift(20) / prices.shift(40)
    result = portfolio_long_short(prices, signal, top_n=3)
    assert result.n_obs > 0
    assert np.isfinite(result.sharpe)


@pytest.mark.unit
def test_portfolio_long_short_empty():
    result = portfolio_long_short(pd.DataFrame(), pd.DataFrame())
    assert result.n_obs == 0
