"""Offline smoke test: prove the LangGraph runtime runs end-to-end.

Uses FakeLLMClient + FakeDataProvider (no API keys, no network) to build
a compiled LangGraph and invoke it with a seed ticker.  Requires the
``langgraph`` package — run this on the server where langgraph is installed.

Usage
-----
    python scripts/smoke_langgraph.py                 # default 688981.SH
    python scripts/smoke_langgraph.py 600519.SH       # positional (matches analyze_client.sh)
    python scripts/smoke_langgraph.py --ts-code 600519.SH
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test the LangGraph runtime end-to-end (offline)."
    )
    # Accept the ticker positionally (matches scripts/analyze_client.sh) OR via
    # --ts-code; positional wins when both are given.
    parser.add_argument(
        "ts_code_pos",
        nargs="?",
        default=None,
        metavar="TS_CODE",
        help="Ticker to analyse, positional (default: 688981.SH).",
    )
    parser.add_argument(
        "--ts-code",
        dest="ts_code_opt",
        default=None,
        help="Ticker to analyse (alternative to the positional form).",
    )
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    ts_code: str = args.ts_code_pos or args.ts_code_opt or "688981.SH"

    # These imports are langgraph-free (build_invest_graph imports langgraph
    # lazily inside its own body, so importing it here does NOT require the
    # package — the ImportError only fires when we CALL it below).
    from invest_forge.agents.graph import GraphDeps, build_invest_graph, _initial_state
    from invest_forge.llm.fake import FakeLLMClient
    from invest_forge.tools.data_tools import FakeDataProvider
    from invest_forge.tools.sentiment import LexiconSentiment

    deps = GraphDeps(
        llm=FakeLLMClient(),
        data_provider=FakeDataProvider(),
        sentiment=LexiconSentiment(),
        retriever=None,
    )

    logger.info("Compiling LangGraph for ticker %s …", ts_code)
    # Guard the COMPILE call: build_invest_graph does `from langgraph.graph
    # import ...` internally, so a missing langgraph surfaces here (local
    # laptop), not at the import above.
    try:
        compiled = build_invest_graph(deps)
    except ImportError as exc:
        print(
            f"langgraph not installed — run this on the server (ImportError: {exc})",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info("Invoking compiled graph …")
    state = compiled.invoke(_initial_state(ts_code, None))

    rec = state.get("final_recommendation", {})
    memo: str = state.get("research_memo", "")
    iterations: int = int(state.get("iteration_count", 0))

    print("\n── LangGraph smoke result ──────────────────────────────────────")
    print(f"  ticker          : {ts_code}")
    print(f"  rating          : {rec.get('rating', 'N/A')}")
    print(f"  confidence      : {rec.get('confidence', 'N/A')}")
    print(f"  iteration_count : {iterations}")
    print(f"  memo snippet    : {memo[:200]!r}")
    print("────────────────────────────────────────────────────────────────")
    logger.info("Smoke test PASSED.")


if __name__ == "__main__":
    main()
