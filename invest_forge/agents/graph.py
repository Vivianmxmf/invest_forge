"""LangGraph wiring for the InvestForge agent pipeline.

The compile step is gated behind ``langgraph`` availability so unit tests
can exercise the node logic directly via ``run_pipeline_inline`` without
needing the LangGraph runtime installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from invest_forge.agents.nodes import (
    make_analyst_node,
    make_data_fetcher_node,
    make_output_node,
    make_researcher_node,
    make_risk_control_node,
    make_vision_node,
    should_revise,
)
from invest_forge.agents.state import InvestState, empty_state
from invest_forge.knowledge_base.retriever import HybridRetriever
from invest_forge.llm.client import LLMClient
from invest_forge.tools.data_tools import DataProvider
from invest_forge.tools.sentiment import Sentiment


@dataclass
class GraphDeps:
    """Bundle of dependencies the agents need; injected by the caller.

    ``analyst_client`` is optional.  When set (provider=local AND
    LOCAL_ANALYST_MODEL is configured) the analyst node uses this client so it
    targets the LoRA adapter while researcher + risk-control keep the base
    model client.  Defaults to ``None``, in which case all nodes share ``llm``.

    ``vision_client`` is optional.  When set (provider=local AND
    LOCAL_VISION_MODEL is configured) the vision node runs before the
    researcher to analyse uploaded report/chart images.  When ``None``,
    the vision node is a no-op and the graph topology is unchanged.
    """

    llm: LLMClient
    data_provider: DataProvider
    sentiment: Sentiment
    retriever: HybridRetriever | None = None
    analyst_client: LLMClient | None = None
    vision_client: LLMClient | None = None
    max_iterations: int = 2


# ───────────────────── LangGraph build ─────────────────────

def build_invest_graph(deps: GraphDeps):  # pragma: no cover - exercised by integration tests
    """Compile a LangGraph StateGraph; requires the ``langgraph`` package."""
    from langgraph.graph import END, StateGraph

    # Use the dedicated analyst client when one is configured, otherwise fall
    # back to the shared llm so existing behaviour is unchanged.
    _analyst_llm = deps.analyst_client if deps.analyst_client is not None else deps.llm

    data_fetcher = make_data_fetcher_node(deps.data_provider, deps.sentiment, deps.retriever)
    researcher = make_researcher_node(deps.llm)
    analyst = make_analyst_node(_analyst_llm)
    risk_control = make_risk_control_node(deps.llm)
    output = make_output_node()

    graph = StateGraph(InvestState)
    graph.add_node("data_fetcher", data_fetcher)
    graph.add_node("researcher", researcher)
    graph.add_node("analyst", analyst)
    graph.add_node("risk_control", risk_control)
    graph.add_node("output", output)

    if deps.vision_client is not None:
        # Insert a vision node between data_fetcher and researcher ONLY when
        # vision is configured.  This keeps the graph topology identical to
        # the non-vision path (data_fetcher → researcher edge) when
        # vision_client is None, so existing graph tests pass unchanged.
        vision = make_vision_node(deps.vision_client)
        graph.add_node("vision", vision)
        graph.set_entry_point("data_fetcher")
        graph.add_edge("data_fetcher", "vision")
        graph.add_edge("vision", "researcher")
    else:
        graph.set_entry_point("data_fetcher")
        graph.add_edge("data_fetcher", "researcher")

    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "risk_control")
    graph.add_conditional_edges(
        "risk_control",
        lambda s: should_revise(s, max_iterations=deps.max_iterations),
        {"revise": "analyst", "finalize": "output"},
    )
    graph.add_edge("output", END)

    return graph.compile()


# ───────────────────── In-process runner ─────────────────────

def run_pipeline_inline(
    deps: GraphDeps,
    *,
    ts_code: str,
    input_images: list[str] | None = None,
) -> InvestState:
    """Execute the same DAG without LangGraph (for tests + simple scripts).

    Same edges, same conditional revision loop, no asyncio.

    ``input_images`` — optional list of validated data-URL strings from the
    API layer.  When provided (and ``deps.vision_client`` is set), the
    vision node runs before the researcher; otherwise both are no-ops and
    the pipeline behaves exactly as before.
    """
    # Use the dedicated analyst client when one is configured, otherwise fall
    # back to the shared llm so existing behaviour is unchanged.
    _analyst_llm = deps.analyst_client if deps.analyst_client is not None else deps.llm

    data_fetcher = make_data_fetcher_node(deps.data_provider, deps.sentiment, deps.retriever)
    vision = make_vision_node(deps.vision_client)
    researcher = make_researcher_node(deps.llm)
    analyst = make_analyst_node(_analyst_llm)
    risk_control = make_risk_control_node(deps.llm)
    output = make_output_node()

    state: InvestState = empty_state()
    state["ts_code"] = ts_code
    if input_images:
        state["input_images"] = list(input_images)

    state = _merge(state, data_fetcher(state))
    # vision is a no-op when vision_client is None or no images present.
    state = _merge(state, vision(state))
    state = _merge(state, researcher(state))

    while True:
        state = _merge(state, analyst(state))
        state = _merge(state, risk_control(state))
        if should_revise(state, max_iterations=deps.max_iterations) == "finalize":
            break

    state = _merge(state, output(state))
    return state


def _merge(state: InvestState, patch: dict[str, Any]) -> InvestState:
    """Shallow-merge a node patch onto state with simple list accumulation."""
    merged = dict(state)
    for k, v in patch.items():
        if k == "errors" and isinstance(v, list):
            existing = list(merged.get("errors") or [])
            merged["errors"] = existing + v
        else:
            merged[k] = v
    return merged  # type: ignore[return-value]
