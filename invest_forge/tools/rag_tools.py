"""Thin RAG facade used by the agent nodes.

We keep the agent code free of vector-store specifics by routing every
retrieval through this module.  The default retriever is built from the
synthetic sample directory; production builds inject a real one (Qdrant +
BGE embeddings + reranker).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

from invest_forge.common.config import get_settings
from invest_forge.common.logging_setup import get_logger
from invest_forge.common.types import RAGHit
from invest_forge.knowledge_base.builder import build_in_memory
from invest_forge.knowledge_base.retriever import HybridRetriever
from invest_forge.llm.client import ChatMessage, LLMClient

logger = get_logger(__name__)


def make_hyde_fn(client: LLMClient) -> Callable[[str], str]:
    """Build a HyDE-style query expander backed by the supplied LLM client.

    The agent passes the same client it uses elsewhere; with the FakeLLMClient
    this is a deterministic short paragraph (see ``invest_forge/llm/fake.py``).
    """

    def expand(query: str) -> str:
        resp = client.complete(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "You generate a *hypothetical document* (3-5 sentences) "
                        "that would answer the user's financial question.  The "
                        "document must read like an excerpt from an annual report, "
                        "containing realistic numbers and CN/EN financial vocabulary. "
                        "[hyde] [hypothetical document]"
                    ),
                ),
                ChatMessage(role="user", content=query),
            ],
            temperature=0.2,
        )
        return resp.text

    return expand


@lru_cache(maxsize=1)
def _default_retriever() -> HybridRetriever:
    sample = get_settings().paths.sample_dir / "kb"
    if not sample.exists():
        logger.warning("no KB directory at %s — RAG will return empty hits", sample)
        return HybridRetriever(texts=[], metadatas=[])
    return build_in_memory(sample)


def retrieve(
    query: str,
    *,
    retriever: HybridRetriever | None = None,
    k: int = 5,
    use_hyde: bool = False,
    use_reranker: bool = False,
) -> list[RAGHit]:
    retriever = retriever or _default_retriever()
    return retriever.retrieve(
        query=query, k=k, use_hyde=use_hyde, use_reranker=use_reranker
    )


def reset_cache() -> None:
    """Clear the lru_cache (used in tests after writing fresh KB fixtures)."""
    _default_retriever.cache_clear()
