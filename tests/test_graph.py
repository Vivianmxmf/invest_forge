"""End-to-end pipeline test (in-process, no LangGraph runtime needed)."""
from __future__ import annotations

import json

import pytest

from janus_terminal.agents.graph import run_pipeline_inline
from janus_terminal.common.types import Rating


@pytest.mark.unit
def test_pipeline_runs_end_to_end(deps):
    state = run_pipeline_inline(deps, ts_code="TEST.SH")
    rec = state["final_recommendation"]
    assert rec["rating"] in (r.value for r in Rating)
    assert 0 <= rec["confidence"] <= 1
    assert state["research_memo"]
    assert state["analyst_report"]
    assert state["risk_assessment"]
    # First successful pass: exactly one analyst iteration
    assert state["iteration_count"] == 1


@pytest.mark.unit
def test_pipeline_revises_on_high_risk(deps):
    # Force the risk-control node to demand revision on the first pass.
    n_calls = {"risk": 0}

    def risk_resp(_msgs):
        n_calls["risk"] += 1
        if n_calls["risk"] == 1:
            return json.dumps(
                {"risk_level": "HIGH", "compliance_ok": False, "notes": "force revise", "require_revision": True}
            )
        return json.dumps(
            {"risk_level": "MEDIUM", "compliance_ok": True, "notes": "ok now", "require_revision": False}
        )

    deps.llm.register(r"\[风控\]|\[Risk Control\]|risk assessment", risk_resp)
    state = run_pipeline_inline(deps, ts_code="TEST.SH")
    assert state["iteration_count"] >= 2
    assert state["risk_assessment"]["risk_level"] == "MEDIUM"


@pytest.mark.unit
def test_pipeline_caps_iterations(deps):
    deps.llm.register(
        r"\[风控\]|\[Risk Control\]|risk assessment",
        json.dumps({"risk_level": "HIGH", "compliance_ok": False, "notes": "loop", "require_revision": True}),
    )
    state = run_pipeline_inline(deps, ts_code="TEST.SH")
    # max_iterations defaults to 2; iteration_count must not run away.
    assert state["iteration_count"] <= deps.max_iterations + 1
