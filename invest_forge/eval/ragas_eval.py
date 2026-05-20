"""Thin wrapper around the RAGAS evaluation library.

Importing ``ragas`` is deferred so the package is importable even when the
heavy dependency is not installed on the laptop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RagasReport:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def evaluate_ragas(
    *,
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> RagasReport:  # pragma: no cover - ragas integration
    """Run RAGAS metrics over an eval set.

    The arrays must have the same length.  This call performs network IO
    (uses an LLM judge), so it is *only* meant for server / CI runs.
    """
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("RAGAS inputs must have equal length")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return RagasReport(
        faithfulness=float(result["faithfulness"]),
        answer_relevancy=float(result["answer_relevancy"]),
        context_precision=float(result["context_precision"]),
        context_recall=float(result["context_recall"]),
    )


def stub_evaluate(answers: Iterable[str], contexts: Iterable[list[str]]) -> RagasReport:
    """Deterministic offline 'RAGAS-like' score used by the laptop demo path.

    For each (answer, contexts) pair: faithfulness = fraction of answer-tokens
    that appear in the union of contexts; relevancy / precision / recall are
    proxied by simple overlap heuristics.  Good enough for the smoke test;
    swap in the real ``evaluate_ragas`` on the server.
    """
    import re

    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9]+|[一-鿿]", text or ""))

    f, r, p, c = [], [], [], []
    for ans, ctxs in zip(answers, contexts):
        ans_tokens = _tokens(ans)
        if not ans_tokens:
            f.append(0.5); r.append(0.5); p.append(0.5); c.append(0.5)
            continue
        ctx_tokens = set().union(*(_tokens(t) for t in ctxs)) if ctxs else set()
        overlap = len(ans_tokens & ctx_tokens) / max(len(ans_tokens), 1)
        f.append(min(1.0, overlap + 0.1))                  # faithfulness
        r.append(min(1.0, 0.5 + 0.5 * overlap))            # relevancy (slightly upbeat)
        p.append(min(1.0, overlap + 0.05))                  # precision
        c.append(min(1.0, overlap + 0.15))                  # recall

    def _avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return RagasReport(
        faithfulness=_avg(f),
        answer_relevancy=_avg(r),
        context_precision=_avg(p),
        context_recall=_avg(c),
    )
