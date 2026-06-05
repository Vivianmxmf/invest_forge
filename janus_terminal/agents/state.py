"""Shared LangGraph state for the InvestForge multi-agent pipeline."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Sequence, TypedDict

# Avoid hard dependency on langgraph at import time so the unit tests can
# walk the state-transition logic without installing the full framework.
try:
    from langchain_core.messages import BaseMessage  # pragma: no cover
except ImportError:  # pragma: no cover
    BaseMessage = Any  # type: ignore[assignment, misc]


class InvestState(TypedDict, total=False):
    """LangGraph shared state.

    All keys are optional so an Airflow / FastAPI caller can supply a
    sparse seed (typically just ``ts_code``).  Nodes mutate by returning
    partial dicts; LangGraph merges into the shared state.
    """

    ts_code: str
    universe: tuple[str, ...]                  # optional supplementary tickers
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Data layer
    fundamental_data: dict[str, Any]
    macro_context: dict[str, Any]
    news_items: list[dict[str, Any]]

    # RAG layer
    rag_evidence: list[dict[str, Any]]

    # Multimodal vision input (validated data-URL strings from the API layer).
    # Empty list == no images supplied → vision node is a no-op.
    input_images: list[str]
    # Output of the vision node.  Empty string == vision node did not run.
    vision_analysis: str

    # Agent outputs
    research_memo: str
    analyst_report: dict[str, Any]
    risk_assessment: dict[str, Any]
    final_recommendation: dict[str, Any]

    # Control
    iteration_count: int
    errors: list[str]


def empty_state() -> InvestState:
    """Return a fresh InvestState seed that node code can mutate freely."""
    return InvestState(  # type: ignore[typeddict-item]
        messages=[],
        fundamental_data={},
        macro_context={},
        news_items=[],
        rag_evidence=[],
        input_images=[],
        vision_analysis="",
        research_memo="",
        analyst_report={},
        risk_assessment={},
        final_recommendation={},
        iteration_count=0,
        errors=[],
    )
