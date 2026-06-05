"""Thin wrapper around the RAGAS evaluation library.

Importing ``ragas`` is deferred so the package is importable even when the
heavy dependency is not installed on the laptop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

# All RAGAS metrics this wrapper knows how to run.  ``answer_relevancy`` is the
# only one that needs an EMBEDDINGS endpoint; the other three are LLM-only and
# work against a plain chat endpoint (e.g. a local vLLM server with no
# /v1/embeddings).  Default selection therefore excludes answer_relevancy so a
# chat-only judge produces the W3 faithfulness number without connection errors.
_ALL_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
_EMBEDDING_METRICS: frozenset[str] = frozenset({"answer_relevancy"})
DEFAULT_METRICS: tuple[str, ...] = (
    "faithfulness",
    "context_precision",
    "context_recall",
)


@dataclass(frozen=True)
class RagasReport:
    # NaN defaults so a partial run (a subset of metrics, or jobs that failed
    # against an unreachable endpoint) still yields a report instead of crashing.
    faithfulness: float = float("nan")
    answer_relevancy: float = float("nan")
    context_precision: float = float("nan")
    context_recall: float = float("nan")


def _to_scalar(value: Any) -> float:
    """Reduce a RAGAS metric result to a single float, NaN-safe.

    Newer ragas returns per-row score *lists* from ``result[name]`` (older
    versions returned a scalar).  A per-row list may also contain NaN for rows
    whose judge/embedding call failed.  Average the finite values; return NaN
    when nothing usable is present or the value is not numeric.
    """
    if isinstance(value, (list, tuple)):
        finite: list[float] = []
        for v in value:
            try:
                f = float(v)  # per-element cast may fail on str/dict/None
            except (TypeError, ValueError):
                continue
            if not math.isnan(f):
                finite.append(f)
        return sum(finite) / len(finite) if finite else float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def evaluate_ragas(
    *,
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    judge_llm: Any = None,
    judge_embeddings: Any = None,
    metrics: list[str] | None = None,
) -> RagasReport:  # pragma: no cover - ragas integration
    """Run RAGAS metrics over an eval set.

    The arrays must have the same length.  This call performs network IO
    (uses an LLM judge), so it is *only* meant for server / CI runs.

    Args:
        questions: List of evaluation questions.
        answers: List of model-generated answers.
        contexts: List of context lists (one list of strings per question).
        ground_truths: List of reference ground-truth answers.
        judge_llm: Optional pre-built ragas LLM wrapper (e.g. from
            ``build_ragas_judge``).  When None, ragas defaults to OpenAI.
        judge_embeddings: Optional pre-built ragas embeddings wrapper.
            When None, ragas defaults to OpenAI embeddings.  Only needed for
            ``answer_relevancy``.
        metrics: Which metrics to run; defaults to ``DEFAULT_METRICS`` (the
            three LLM-only metrics).  Include ``"answer_relevancy"`` only when
            an embeddings endpoint is available, or its jobs fail with
            connection errors (reported as NaN).

    Returns:
        A ``RagasReport``; metrics not selected (or whose jobs all failed) are
        reported as NaN.
    """
    # Both validations run BEFORE the ragas import so they are testable offline.
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("RAGAS inputs must have equal length")
    selected = list(metrics) if metrics else list(DEFAULT_METRICS)
    unknown = [m for m in selected if m not in _ALL_METRICS]
    if unknown:
        raise ValueError(f"unknown RAGAS metric(s): {unknown}; valid: {list(_ALL_METRICS)}")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    metric_objs = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    eval_kwargs: dict[str, Any] = {
        "dataset": ds,
        "metrics": [metric_objs[m] for m in selected],
    }
    if judge_llm is not None:
        eval_kwargs["llm"] = judge_llm
    if judge_embeddings is not None:
        eval_kwargs["embeddings"] = judge_embeddings

    result = evaluate(**eval_kwargs)

    def _extract(name: str) -> float:
        if name not in selected:
            return float("nan")
        try:
            return _to_scalar(result[name])
        except (KeyError, TypeError, IndexError):
            return float("nan")

    return RagasReport(
        faithfulness=_extract("faithfulness"),
        answer_relevancy=_extract("answer_relevancy"),
        context_precision=_extract("context_precision"),
        context_recall=_extract("context_recall"),
    )


def build_ragas_judge(settings: Any = None) -> tuple[Any, Any]:  # pragma: no cover
    """Build a LangChain-backed RAGAS judge LLM and embeddings wrapper.

    Reads LLM settings to construct a ChatOpenAI-compatible judge that works
    with either the OpenAI API or a local vLLM endpoint (set
    ``LOCAL_LLM_BASE_URL`` and ``LOCAL_LLM_MODEL``).

    Note: answer_relevancy and context_precision require a working embeddings
    endpoint.  When using a local vLLM server, ensure it exposes the
    ``/v1/embeddings`` endpoint (e.g. via ``--served-model-name`` or a
    dedicated embedding model container).

    Args:
        settings: A ``Settings`` object from ``get_settings()``.  When None,
            ``get_settings()`` is called automatically.

    Returns:
        ``(wrapped_llm, wrapped_embeddings)`` — both are ragas-compatible
        wrappers ready to pass to ``evaluate_ragas``.
    """
    import os

    from janus_terminal.common.config import get_settings as _get_settings

    resolved = settings or _get_settings()
    cfg = resolved.llm
    vec_cfg = resolved.vector_store

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    model_name = cfg.local_model or cfg.model
    base_url = cfg.local_base_url or None
    api_key = cfg.openai_api_key or "EMPTY"

    chat_model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,  # type: ignore[arg-type]
    )
    wrapped_llm = LangchainLLMWrapper(chat_model)

    # Embeddings (only used by answer_relevancy).  A local vLLM CHAT server has
    # no /v1/embeddings, so allow a separate endpoint via RAGAS_EMBEDDINGS_BASE_URL
    # (e.g. point it at OpenAI or a dedicated embedding container).  Falls back
    # to the judge's base_url otherwise.
    embedding_model = getattr(vec_cfg, "embedding_model", None) or "text-embedding-3-small"
    embeddings_base_url = os.getenv("RAGAS_EMBEDDINGS_BASE_URL") or base_url
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        base_url=embeddings_base_url,
        api_key=api_key,  # type: ignore[arg-type]
    )
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return wrapped_llm, wrapped_embeddings


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
