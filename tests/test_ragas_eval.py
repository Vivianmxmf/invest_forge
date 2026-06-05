"""Offline tests for RAGAS eval wiring.

All tests are network-free and do NOT require ragas or langchain to be
installed — they exercise only the paths that run before any lazy import.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from janus_terminal.eval.ragas_eval import RagasReport, stub_evaluate

# ---------------------------------------------------------------------------
# locate the committed eval set
# ---------------------------------------------------------------------------

_EVAL_SET = _ROOT / "data" / "sample" / "ragas_eval_set.jsonl"


# ---------------------------------------------------------------------------
# 1. Length mismatch raises ValueError BEFORE any ragas import
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_evaluate_ragas_length_mismatch_raises() -> None:
    """evaluate_ragas must raise ValueError for mismatched list lengths.

    This check runs before the lazy ragas import, so it works even when ragas
    is not installed.
    """
    from janus_terminal.eval.ragas_eval import evaluate_ragas

    with pytest.raises(ValueError, match="equal length"):
        evaluate_ragas(
            questions=["q1", "q2"],
            answers=["a1"],          # wrong length
            contexts=[["c1"], ["c2"]],
            ground_truths=["g1", "g2"],
        )


@pytest.mark.unit
def test_evaluate_ragas_unknown_metric_raises() -> None:
    """An unknown metric name must raise before the lazy ragas import."""
    from janus_terminal.eval.ragas_eval import evaluate_ragas

    with pytest.raises(ValueError, match="unknown RAGAS metric"):
        evaluate_ragas(
            questions=["q1"],
            answers=["a1"],
            contexts=[["c1"]],
            ground_truths=["g1"],
            metrics=["faithfulness", "bogus_metric"],
        )


@pytest.mark.unit
def test_to_scalar_reduces_lists_nan_safe() -> None:
    """_to_scalar: list → mean of finite values; scalar passthrough; junk → NaN.

    Newer ragas returns per-row score lists (possibly with NaN for failed
    jobs); this guards the regression where float(list) crashed.
    """
    import math

    from janus_terminal.eval.ragas_eval import _to_scalar

    assert _to_scalar([0.6, 0.8, 1.0]) == pytest.approx(0.8)
    # NaN entries (failed jobs) are dropped from the mean.
    assert _to_scalar([0.5, float("nan"), 0.7]) == pytest.approx(0.6)
    assert _to_scalar(0.42) == pytest.approx(0.42)
    # Non-numeric elements INSIDE a list (str/dict/None) are skipped, not raised.
    assert _to_scalar(["not-a-number", 0.8]) == pytest.approx(0.8)
    assert _to_scalar([{}, None, 0.6]) == pytest.approx(0.6)
    # All-NaN / empty / non-numeric → NaN (report, don't crash).
    assert math.isnan(_to_scalar([float("nan")]))
    assert math.isnan(_to_scalar(["junk", {}, None]))
    assert math.isnan(_to_scalar([]))
    assert math.isnan(_to_scalar("not-a-number"))


@pytest.mark.unit
def test_default_metrics_exclude_answer_relevancy() -> None:
    """The chat-only default must omit answer_relevancy (it needs embeddings)."""
    from janus_terminal.eval.ragas_eval import DEFAULT_METRICS

    assert "answer_relevancy" not in DEFAULT_METRICS
    assert "faithfulness" in DEFAULT_METRICS


# ---------------------------------------------------------------------------
# 2. Committed eval set parses correctly
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_eval_set_loads() -> None:
    """The committed JSONL eval set must parse to exactly 6 rows with the
    four required keys and all lists of equal length."""
    assert _EVAL_SET.exists(), f"Eval set missing: {_EVAL_SET}"

    rows = []
    for lineno, raw in enumerate(_EVAL_SET.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        rows.append(row)

    assert len(rows) == 6, f"Expected 6 rows, got {len(rows)}"

    required_keys = {"question", "answer", "contexts", "ground_truth"}
    for i, row in enumerate(rows):
        missing = required_keys - set(row.keys())
        assert not missing, f"Row {i + 1} missing keys: {missing}"
        assert isinstance(row["contexts"], list), f"Row {i + 1}: 'contexts' must be a list"
        assert row["question"], f"Row {i + 1}: 'question' must be non-empty"
        assert row["answer"], f"Row {i + 1}: 'answer' must be non-empty"
        assert row["ground_truth"], f"Row {i + 1}: 'ground_truth' must be non-empty"

    # All four parallel lists are the same length (trivially 6, but check anyway)
    questions = [r["question"] for r in rows]
    answers = [r["answer"] for r in rows]
    contexts = [r["contexts"] for r in rows]
    ground_truths = [r["ground_truth"] for r in rows]
    assert len(questions) == len(answers) == len(contexts) == len(ground_truths) == 6


# ---------------------------------------------------------------------------
# 3. stub_evaluate returns a valid RagasReport
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stub_evaluate_returns_report() -> None:
    """stub_evaluate must return a RagasReport with all 4 metrics in [0, 1]."""
    answers = [
        "贵州茅台2024年净利润约768亿元，净利率52.3%。",
        "中芯国际市值约1435亿，市盈率14.17倍。",
        "宁德时代营收3600亿，自由现金流281亿。",
    ]
    contexts = [
        ["贵州茅台净利润768亿元，净利率52.3%，营收1470亿元。"],
        ["中芯国际市值1435亿，市盈率14.17倍，净利润65亿。"],
        ["宁德时代营收3600亿，净利润331亿，自由现金流281亿。"],
    ]
    report = stub_evaluate(answers, contexts)

    assert isinstance(report, RagasReport)
    assert 0.0 <= report.faithfulness <= 1.0
    assert 0.0 <= report.answer_relevancy <= 1.0
    assert 0.0 <= report.context_precision <= 1.0
    assert 0.0 <= report.context_recall <= 1.0


# ---------------------------------------------------------------------------
# 4. Threshold gate helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runner_threshold_gate() -> None:
    """meets_faithfulness must correctly classify reports above/below threshold."""
    from scripts.run_ragas_eval import meets_faithfulness

    high_report = RagasReport(
        faithfulness=0.85,
        answer_relevancy=0.80,
        context_precision=0.78,
        context_recall=0.82,
    )
    low_report = RagasReport(
        faithfulness=0.72,
        answer_relevancy=0.75,
        context_precision=0.70,
        context_recall=0.74,
    )
    exactly_at = RagasReport(
        faithfulness=0.80,
        answer_relevancy=0.80,
        context_precision=0.80,
        context_recall=0.80,
    )

    assert meets_faithfulness(high_report, 0.8) is True
    assert meets_faithfulness(low_report, 0.8) is False
    assert meets_faithfulness(exactly_at, 0.8) is True  # boundary is inclusive


def _patch_runner_real_path(monkeypatch: pytest.MonkeyPatch, report: RagasReport) -> None:
    """Stub the runner's lazy-imported real-path deps so main() runs offline.

    main() does ``from janus_terminal.eval.ragas_eval import build_ragas_judge,
    evaluate_ragas`` and ``from janus_terminal.common.config import get_settings``
    AT CALL TIME, so patching the attributes on the SOURCE modules is picked up
    when main() resolves the from-import — no real ragas/langchain needed.
    """
    import janus_terminal.common.config as cfg_mod
    import janus_terminal.eval.ragas_eval as ragas_mod

    monkeypatch.setattr(ragas_mod, "evaluate_ragas", lambda **_kw: report)
    monkeypatch.setattr(ragas_mod, "build_ragas_judge", lambda _settings=None: (object(), object()))
    monkeypatch.setattr(cfg_mod, "get_settings", lambda: MagicMock())


@pytest.mark.unit
def test_runner_real_path_passes_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive main() through the REAL (non-stub) path with a passing report.

    Exercises arg parsing → load_eval_set → build_ragas_judge → evaluate_ragas →
    faithfulness gate, fully offline. faithfulness 0.85 >= 0.8 → PASS → main()
    returns normally (no SystemExit).
    """
    from scripts.run_ragas_eval import main

    _patch_runner_real_path(
        monkeypatch,
        RagasReport(faithfulness=0.85, answer_relevancy=0.85, context_precision=0.83, context_recall=0.86),
    )
    # Real path (no --stub); must not raise — PASS means no sys.exit.
    main(["--eval-set", str(_EVAL_SET)])


@pytest.mark.unit
def test_runner_real_path_fails_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real path must exit non-zero when faithfulness < threshold (CI gate)."""
    from scripts.run_ragas_eval import main

    _patch_runner_real_path(
        monkeypatch,
        RagasReport(faithfulness=0.50, answer_relevancy=0.60, context_precision=0.55, context_recall=0.58),
    )
    with pytest.raises(SystemExit) as exc:
        main(["--eval-set", str(_EVAL_SET)])
    assert exc.value.code == 1


@pytest.mark.unit
def test_runner_threshold_gate_boundary() -> None:
    """meets_faithfulness boundary is inclusive at the threshold."""
    from scripts.run_ragas_eval import meets_faithfulness

    assert meets_faithfulness(RagasReport(0.85, 0.8, 0.78, 0.82), 0.8) is True
    assert meets_faithfulness(RagasReport(0.72, 0.75, 0.70, 0.74), 0.8) is False
    assert meets_faithfulness(RagasReport(0.80, 0.80, 0.80, 0.80), 0.8) is True
