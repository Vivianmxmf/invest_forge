"""Quality scoring tests."""
from __future__ import annotations

import pytest

from invest_forge.eval.quality import HeuristicScorer
from invest_forge.eval.ragas_eval import stub_evaluate


@pytest.mark.unit
def test_heuristic_perfect_report():
    report = {
        "rating": "BUY",
        "confidence": 0.8,
        "rationale": "营收 432 亿元, 净利率 15.1%, ROE 17%; 估值合理.",
        "risk_factors": ["客户集中", "汇率敞口", "出口管制"],
    }
    evidence = "公司 2024 年营业收入 432 亿元, ROE 17%, 净利率 15.1%."
    score = HeuristicScorer().score(report, evidence=evidence)
    assert score.coherence > 0.7
    assert score.grounding > 0.5
    assert score.risk_coverage == 1.0
    assert 0 < score.overall <= 1


@pytest.mark.unit
def test_heuristic_no_risks():
    report = {"rating": "HOLD", "confidence": 0.5, "rationale": "略", "risk_factors": []}
    score = HeuristicScorer().score(report, evidence="")
    assert score.risk_coverage == 0.0


@pytest.mark.unit
def test_heuristic_decisive_low_confidence_penalised():
    decisive_lowconf = HeuristicScorer().score(
        {"rating": "BUY", "confidence": 0.2, "rationale": "x", "risk_factors": ["a", "b", "c"]}
    )
    decisive_highconf = HeuristicScorer().score(
        {"rating": "BUY", "confidence": 0.9, "rationale": "x", "risk_factors": ["a", "b", "c"]}
    )
    assert decisive_highconf.coherence > decisive_lowconf.coherence


@pytest.mark.unit
def test_stub_ragas_report():
    answers = ["公司净利率 15.1%"]
    contexts = [["公司 2024 年净利率 15.1%, 营收 432 亿元"]]
    report = stub_evaluate(answers, contexts)
    assert 0 < report.faithfulness <= 1
    assert 0 < report.answer_relevancy <= 1


@pytest.mark.unit
def test_stub_ragas_empty_context_penalised():
    answers = ["公司净利率 15.1%"]
    contexts = [[]]
    report = stub_evaluate(answers, contexts)
    # When there's no context, faithfulness defaults to a small floor.
    assert report.faithfulness <= 0.5
