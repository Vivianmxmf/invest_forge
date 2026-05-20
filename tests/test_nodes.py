"""Node-level tests for the agent pipeline."""
from __future__ import annotations

import json

import pytest

from invest_forge.agents.nodes import (
    _parse_json,
    make_analyst_node,
    make_data_fetcher_node,
    make_output_node,
    make_researcher_node,
    make_risk_control_node,
    should_revise,
)
from invest_forge.agents.state import empty_state
from invest_forge.common.types import Rating


@pytest.mark.unit
def test_parse_json_tolerates_fences():
    raw = "```json\n{\n \"rating\": \"BUY\"\n}\n```"
    assert _parse_json(raw) == {"rating": "BUY"}


@pytest.mark.unit
def test_parse_json_returns_empty_on_garbage():
    assert _parse_json("no json here at all") == {}


@pytest.mark.unit
def test_data_fetcher_populates_state(fake_data, lexicon_sentiment, tiny_retriever):
    node = make_data_fetcher_node(fake_data, lexicon_sentiment, tiny_retriever)
    state = empty_state()
    state["ts_code"] = "TEST.SH"
    patch = node(state)
    assert "fundamental_data" in patch and patch["fundamental_data"]["ts_code"] == "TEST.SH"
    assert "macro_context" in patch
    assert "news_items" in patch and len(patch["news_items"]) >= 1
    # Sentiment must have been re-scored by the lexicon.
    labels = {n["label"] for n in patch["news_items"]}
    assert labels & {"positive", "negative", "neutral"}
    # RAG hits should come back from the tiny in-memory retriever.
    assert "rag_evidence" in patch


@pytest.mark.unit
def test_data_fetcher_missing_ts_code(fake_data, lexicon_sentiment):
    node = make_data_fetcher_node(fake_data, lexicon_sentiment, None)
    patch = node(empty_state())
    assert patch.get("errors")
    assert "ts_code" in patch["errors"][0]


@pytest.mark.unit
def test_researcher_node_calls_llm(fake_llm):
    node = make_researcher_node(fake_llm)
    state = empty_state()
    state["fundamental_data"] = {"revenue": 1}
    state["macro_context"] = {"summary": "test"}
    patch = node(state)
    assert "盈利质量" in patch["research_memo"]
    assert len(fake_llm.calls) == 1


@pytest.mark.unit
def test_analyst_node_returns_canonical_rating(fake_llm):
    node = make_analyst_node(fake_llm)
    state = empty_state()
    state["research_memo"] = "memo"
    patch = node(state)
    assert patch["analyst_report"]["rating"] in (r.value for r in Rating)
    assert 0 <= patch["analyst_report"]["confidence"] <= 1
    assert patch["iteration_count"] == 1


@pytest.mark.unit
def test_analyst_node_clamps_confidence_out_of_range(fake_llm):
    fake_llm.register(r"\[Analyst\]", json.dumps({"rating": "BUY", "confidence": 1.7, "rationale": "x", "risk_factors": []}))
    node = make_analyst_node(fake_llm)
    patch = node(empty_state())
    assert patch["analyst_report"]["confidence"] == 1.0


@pytest.mark.unit
def test_risk_control_defaults_when_llm_returns_garbage(fake_llm):
    fake_llm.register(r"\[风控\]|\[Risk Control\]|risk assessment", "not json at all")
    node = make_risk_control_node(fake_llm)
    patch = node(empty_state())
    assert patch["risk_assessment"]["risk_level"] == "MEDIUM"
    assert patch["risk_assessment"]["compliance_ok"] is True


@pytest.mark.unit
def test_should_revise_routes_correctly():
    state = empty_state()
    state["risk_assessment"] = {"risk_level": "HIGH", "require_revision": True}
    state["iteration_count"] = 0
    assert should_revise(state) == "revise"

    state["iteration_count"] = 5
    assert should_revise(state) == "finalize"

    state["risk_assessment"] = {"risk_level": "LOW", "require_revision": False}
    state["iteration_count"] = 0
    assert should_revise(state) == "finalize"


@pytest.mark.unit
def test_should_revise_handles_string_booleans():
    """Codex round-7 P2: LLM-returned string booleans must not loop forever."""
    state = empty_state()
    state["risk_assessment"] = {"risk_level": "LOW", "require_revision": "false"}
    state["iteration_count"] = 0
    assert should_revise(state) == "finalize"
    state["risk_assessment"]["require_revision"] = "true"
    assert should_revise(state) == "revise"


@pytest.mark.unit
def test_data_fetcher_boosts_matching_ticker_in_rag():
    """Codex round-4 P1: when the KB only carries the ts_code in metadata,
    the data_fetcher must rank the matching doc first."""
    from invest_forge.knowledge_base.retriever import HybridRetriever
    from invest_forge.tools.data_tools import FakeDataProvider
    from invest_forge.tools.sentiment import LexiconSentiment

    texts = [
        "公司是行业龙头, 营收稳健.",
        "公司风险: 客户集中度高, 海外汇率敞口.",
        "另一家公司, 周期性强, 估值 PE=12.",
    ]
    metas = [
        {"source": "600276_excerpt.md"},
        {"source": "688981_excerpt.md"},
        {"source": "601398_excerpt.md"},
    ]
    retriever = HybridRetriever(texts=texts, metadatas=metas)
    node = make_data_fetcher_node(FakeDataProvider(), LexiconSentiment(), retriever)
    state = empty_state()
    state["ts_code"] = "688981.SH"
    patch = node(state)
    assert patch["rag_evidence"]
    assert "688981" in patch["rag_evidence"][0]["source"]


@pytest.mark.unit
def test_output_node_assembles_recommendation():
    node = make_output_node()
    state = empty_state()
    state["analyst_report"] = {"rating": "BUY", "confidence": 0.7, "rationale": "x",
                                "target_price": 100, "risk_factors": ["r1", "r2"]}
    state["risk_assessment"] = {"risk_level": "MEDIUM", "compliance_ok": True, "notes": "OK"}
    state["iteration_count"] = 2
    patch = node(state)
    rec = patch["final_recommendation"]
    assert rec["rating"] == "BUY"
    assert rec["risk_factors"] == ["r1", "r2"]
    assert rec["iteration_count"] == 2
