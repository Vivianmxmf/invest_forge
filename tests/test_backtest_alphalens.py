"""alphalens integration tests — SERVER ONLY.

This entire module is skipped when alphalens is not installed (laptop).
On the server (where alphalens is available) these tests exercise the
alphalens_report adapter end-to-end.
"""
from __future__ import annotations

from pathlib import Path

import pytest

alphalens = pytest.importorskip(
    "alphalens",
    reason="alphalens not installed — skip on laptop, run on server",
)

from invest_forge.tools.backtest_tools import (  # noqa: E402
    alphalens_report,
    build_factor_panel,
    load_price_panel,
)

_SAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "sample" / "prices.csv"


@pytest.mark.unit
def test_alphalens_report_returns_expected_keys() -> None:
    """alphalens_report returns a dict with mean_ic, mean_return_by_quantile, n_obs."""
    panel = load_price_panel(_SAMPLE_CSV)
    factor = build_factor_panel(panel, window=20)

    report = alphalens_report(factor, panel, periods=(1, 5, 20), quantiles=2)

    # Top-level keys must be present.
    assert "mean_ic" in report, "'mean_ic' key missing from report"
    assert "mean_return_by_quantile" in report, "'mean_return_by_quantile' key missing"
    assert "n_obs" in report, "'n_obs' key missing from report"

    # mean_ic should have entries for all 3 periods.
    assert len(report["mean_ic"]) == 3, (
        f"Expected 3 period keys in mean_ic, got {len(report['mean_ic'])}"
    )

    # n_obs should be positive.
    assert report["n_obs"] > 0, f"n_obs should be > 0, got {report['n_obs']}"
