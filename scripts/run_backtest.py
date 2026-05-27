"""Run the pure-pandas cross-sectional backtest on the 5-ticker sample data.

Basic usage (no API keys required):

    python scripts/run_backtest.py

With alphalens analytics (server only — alphalens must be installed):

    python scripts/run_backtest.py --alphalens --out-png /tmp/tearsheet.png

All alphalens code is gated behind ``--alphalens`` so the script is fully
import-safe on a laptop that does not have alphalens installed.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from invest_forge.tools.backtest_tools import (
    build_factor_panel,
    load_price_panel,
    make_lookahead_safe_signal,
    portfolio_long_short,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="InvestForge W4: cross-sectional backtest on sample data."
    )
    parser.add_argument(
        "--prices",
        default="data/sample/prices.csv",
        help="Path to the long-format prices CSV (default: data/sample/prices.csv)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling MA window for the momentum signal (default: 20)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=2,
        help="Long top-N / short bottom-N tickers per day (default: 2)",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=2,
        help="Quantile count for alphalens (default: 2, safe for tiny universes)",
    )
    parser.add_argument(
        "--alphalens",
        action="store_true",
        help="Run alphalens analytics (requires alphalens to be installed — server only)",
    )
    parser.add_argument(
        "--out-png",
        default=None,
        help="If provided and --alphalens is set, save a tear-sheet PNG here",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prices_path = Path(args.prices)
    if not prices_path.is_absolute():
        prices_path = _ROOT / prices_path

    logger.info("Loading price panel from %s", prices_path)
    panel = load_price_panel(prices_path)
    logger.info(
        "Panel loaded: %d dates × %d tickers  [%s … %s]",
        len(panel),
        len(panel.columns),
        panel.index[0].date(),
        panel.index[-1].date(),
    )

    # ── Pure-pandas backtest (always runs, no alphalens) ────────────────────
    window: int = args.window
    factor_wide = panel.apply(lambda col: make_lookahead_safe_signal(col, window))
    result = portfolio_long_short(panel, factor_wide, top_n=args.top_n)

    print("\n" + "=" * 55)
    print("  InvestForge — 5-Ticker Cross-Sectional Backtest")
    print("=" * 55)
    print(f"  Tickers      : {', '.join(panel.columns.tolist())}")
    print(f"  Date range   : {panel.index[0].date()} → {panel.index[-1].date()}")
    print(f"  MA window    : {window}")
    print(f"  Top-N (each) : {args.top_n}")
    print("-" * 55)
    print(f"  IC mean      : {result.ic_mean:+.4f}")
    print(f"  IC IR        : {result.ic_ir:+.4f}")
    print(f"  Ann. return  : {result.annualised_return:+.2%}")
    print(f"  Sharpe       : {result.sharpe:+.4f}")
    print(f"  Max drawdown : {result.max_drawdown:+.2%}")
    print(f"  N obs        : {result.n_obs}")
    print("=" * 55 + "\n")

    # ── alphalens analytics (gated, server-only) ─────────────────────────────
    if args.alphalens:
        logger.info("Building MultiIndex factor for alphalens …")
        factor = build_factor_panel(panel, window=window)
        logger.info("Factor observations: %d", len(factor))

        # Lazy import — only reached when --alphalens is passed.
        from invest_forge.tools.backtest_tools import alphalens_report  # noqa: PLC0415

        out_png = Path(args.out_png) if args.out_png else None
        try:
            report = alphalens_report(
                factor,
                panel,
                quantiles=args.quantiles,
                out_png=out_png,
            )
        except ImportError as exc:
            # alphalens is server-only; surface a clean one-liner instead of a
            # raw ModuleNotFoundError traceback when run on a laptop.
            print(
                f"alphalens not installed — run --alphalens on the server (ImportError: {exc})",
                file=sys.stderr,
            )
            sys.exit(1)

        print("  alphalens — Mean IC per period")
        print("  " + "-" * 35)
        for period, ic_val in sorted(report["mean_ic"].items()):
            print(f"    {period:>6}  {float(ic_val):+.4f}")

        print("\n  alphalens — Mean Return by Quantile")
        print("  " + "-" * 35)
        for period, quant_dict in sorted(report["mean_return_by_quantile"].items()):
            for q, ret_val in sorted(quant_dict.items(), key=lambda kv: kv[0]):
                print(f"    period={period:>6}  quantile={q}  mean_ret={float(ret_val):+.6f}")

        print(f"\n  n_obs: {report['n_obs']}")
        if out_png is not None:
            print(f"  Tear-sheet PNG → {out_png}")
        print()


if __name__ == "__main__":
    main()
