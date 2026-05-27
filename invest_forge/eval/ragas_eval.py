"""Thin wrapper around the RAGAS evaluation library.

Importing ``ragas`` is deferred so the package is importable even when the
heavy dependency is not installed on the laptop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


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
    judge_llm: Any = None,
    judge_embeddings: Any = None,
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
            When None, ragas defaults to OpenAI embeddings.  Required for
            answer_relevancy and context_precision metrics.
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

    eval_kwargs: dict[str, Any] = {
        "dataset": ds,
        "metrics": [faithfulness, answer_relevancy, context_precision, context_recall],
    }
    if judge_llm is not None:
        eval_kwargs["llm"] = judge_llm
    if judge_embeddings is not None:
        eval_kwargs["embeddings"] = judge_embeddings

    result = evaluate(**eval_kwargs)
    return RagasReport(
        faithfulness=float(result["faithfulness"]),
        answer_relevancy=float(result["answer_relevancy"]),
        context_precision=float(result["context_precision"]),
        context_recall=float(result["context_recall"]),
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
    from invest_forge.common.config import get_settings as _get_settings

    cfg = (settings or _get_settings()).llm
    vec_cfg = (settings or _get_settings()).vector_store

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

    # Use the embedding model from vector_store config when available,
    # otherwise fall back to a sensible OpenAI default.
    embedding_model = getattr(vec_cfg, "embedding_model", None) or "text-embedding-3-small"
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        base_url=base_url,
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
