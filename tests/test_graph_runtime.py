"""LangGraph runtime tests.

The entire module is skipped when ``langgraph`` is not installed (local
laptop dev workflow).  On the server — where langgraph is available — these
tests exercise the compiled runtime and verify parity with the inline runner.
"""
from __future__ import annotations

import pytest

# Skip the WHOLE module when langgraph is absent.  Gate on the actual
# submodule the runtime needs (from langgraph.graph import END, StateGraph),
# not the bare namespace — a partial/namespace-only install must still skip.
pytest.importorskip("langgraph.graph")

from janus_terminal.agents.graph import (  # noqa: E402
    GraphDeps,
    _initial_state,
    build_invest_graph,
    run_pipeline_graph,
    run_pipeline_inline,
)
from janus_terminal.common.types import Rating  # noqa: E402
from janus_terminal.llm.fake import FakeLLMClient  # noqa: E402
from janus_terminal.tools.data_tools import FakeDataProvider  # noqa: E402
from janus_terminal.tools.sentiment import LexiconSentiment  # noqa: E402


@pytest.fixture
def rt_deps() -> GraphDeps:
    """Deterministic offline deps for runtime tests."""
    return GraphDeps(
        llm=FakeLLMClient(),
        data_provider=FakeDataProvider(),
        sentiment=LexiconSentiment(),
        retriever=None,
    )


@pytest.mark.unit
def test_langgraph_runtime_end_to_end(rt_deps: GraphDeps) -> None:
    """build_invest_graph().invoke() completes and returns a valid state."""
    compiled = build_invest_graph(rt_deps)
    state = compiled.invoke(_initial_state("688981.SH", None))

    rec = state.get("final_recommendation", {})
    assert rec, "final_recommendation must be non-empty"
    assert rec["rating"] in {r.value for r in Rating}, (
        f"rating {rec['rating']!r} is not a valid Rating value"
    )
    assert 0.0 <= rec["confidence"] <= 1.0, (
        f"confidence {rec['confidence']} out of [0, 1]"
    )
    assert int(state.get("iteration_count", 0)) >= 1, (
        "iteration_count must be ≥ 1 after at least one analyst pass"
    )


@pytest.mark.unit
def test_runtime_parity_with_inline(rt_deps: GraphDeps) -> None:
    """run_pipeline_graph and run_pipeline_inline must agree on outputs.

    Both runners use identical deps (same FakeLLMClient seeded identically
    because FakeLLMClient is deterministic by design).  The final
    recommendation rating and analyst_report must be equal.
    """
    state_graph = run_pipeline_graph(rt_deps, ts_code="688981.SH")
    state_inline = run_pipeline_inline(rt_deps, ts_code="688981.SH")

    assert (
        state_graph["final_recommendation"]["rating"]
        == state_inline["final_recommendation"]["rating"]
    ), "rating mismatch between runtime and inline runners"

    assert state_graph["analyst_report"] == state_inline["analyst_report"], (
        "analyst_report mismatch between runtime and inline runners"
    )
