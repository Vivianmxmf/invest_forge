"""Hybrid retrieval: BM25 + vector + (optional) reranker + (optional) HyDE.

The class is designed to run completely offline for tests:
  * BM25 falls back to a pure-Python `rank_bm25` wrapper or a tiny tf-idf if
    that library is missing.
  * Embedding similarity uses an in-memory cosine when no real embedding
    backend is supplied.
  * The reranker step is a no-op when no model is loaded.

Server deployments inject real ``embed_fn`` / ``reranker_fn`` callables
(BGE small + BGE-reranker-v2-m3 typically).  See ``builder.py``.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

from invest_forge.common.types import RAGHit


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    # Mixed CN+EN tokenizer: split on non-alphanumeric, then per-character for CJK
    out: list[str] = []
    for chunk in re.findall(r"[A-Za-z0-9]+|[一-鿿]+", text):
        if chunk.isascii():
            out.append(chunk)
        else:
            # Bigram + unigram for Chinese
            out.extend(chunk)
            for i in range(len(chunk) - 1):
                out.append(chunk[i : i + 2])
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if (na and nb) else 0.0


# ───────────────────── BM25 ─────────────────────

@dataclass
class _BM25:
    """Tiny pure-Python BM25 (k1=1.5, b=0.75) — sufficient for the unit tests."""

    corpus: list[list[str]]
    k1: float = 1.5
    b: float = 0.75
    avgdl: float = 0.0
    df: Counter = field(default_factory=Counter)
    idf: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.corpus)
        if not n:
            return
        self.avgdl = sum(len(d) for d in self.corpus) / n
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1
        for term, n_t in self.df.items():
            self.idf[term] = math.log(1 + (n - n_t + 0.5) / (n_t + 0.5))

    def score(self, query: list[str]) -> list[float]:
        scores = [0.0] * len(self.corpus)
        for i, doc in enumerate(self.corpus):
            if not doc:
                continue
            dl = len(doc)
            tf = Counter(doc)
            for term in query:
                if term not in self.idf:
                    continue
                freq = tf[term]
                num = freq * (self.k1 + 1)
                den = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1.0))
                scores[i] += self.idf[term] * num / den
        return scores


# ───────────────────── Retriever ─────────────────────

@dataclass
class HybridRetriever:
    """Stateless container for hybrid retrieval over a small in-memory corpus."""

    texts: list[str]
    metadatas: list[dict]
    embeddings: list[list[float]] = field(default_factory=list)
    embed_fn: Callable[[str], list[float]] | None = None
    reranker_fn: Callable[[str, list[str]], list[float]] | None = None
    hyde_fn: Callable[[str], str] | None = None
    bm25_weight: float = 0.4
    vector_weight: float = 0.6

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.metadatas):
            raise ValueError("texts and metadatas must be the same length")
        self._bm25 = _BM25([_tokenize(t) for t in self.texts])

    # ----------------------- public API -----------------------

    def add(self, text: str, metadata: dict, embedding: list[float] | None = None) -> None:
        self.texts.append(text)
        self.metadatas.append(metadata)
        # Keep embeddings in sync.  If the retriever was constructed with an
        # ``embed_fn`` and at least one embedding exists, dynamically computed
        # entries must be appended too; otherwise vector scoring will silently
        # drop the new document from ranking.
        if embedding is not None:
            self.embeddings.append(embedding)
        elif self.embed_fn is not None and (self.embeddings or not self.texts[:-1]):
            self.embeddings.append(self.embed_fn(text))
        # If embeddings were never used, leave the list empty (BM25 only mode).

        self._bm25 = _BM25([_tokenize(t) for t in self.texts])

    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        use_hyde: bool = False,
        use_reranker: bool = False,
    ) -> list[RAGHit]:
        if not self.texts:
            return []

        effective_query = query
        if use_hyde and self.hyde_fn is not None:
            effective_query = self.hyde_fn(query) or query

        # --- BM25 leg ---
        bm25_scores = self._bm25.score(_tokenize(effective_query))
        bm25_norm = _normalise(bm25_scores)

        # --- Vector leg ---
        # Pad to len(self.texts) so documents added without embeddings still
        # participate in scoring (they get vec_score=0 but full BM25 weight).
        if self.embed_fn is not None and self.embeddings:
            qv = self.embed_fn(effective_query)
            vec_scores = [_cosine(qv, ev) for ev in self.embeddings]
        else:
            vec_scores = []
        if len(vec_scores) < len(self.texts):
            vec_scores = vec_scores + [0.0] * (len(self.texts) - len(vec_scores))
        vec_norm = _normalise(vec_scores)

        # --- Fuse --- (lengths are guaranteed equal by the pad above)
        fused = [
            self.bm25_weight * b + self.vector_weight * v
            for b, v in zip(bm25_norm, vec_norm)
        ]

        # Top-N candidates (oversample for the reranker)
        candidate_k = min(len(self.texts), max(k * 3, k))
        idx_sorted = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)[:candidate_k]

        # --- Rerank ---
        if use_reranker and self.reranker_fn is not None:
            cand_texts = [self.texts[i] for i in idx_sorted]
            rerank_scores = self.reranker_fn(effective_query, cand_texts)
            # Build the score map against the ORIGINAL candidate order so each
            # score stays attached to the document that produced it; only then
            # do we reorder ``idx_sorted`` to reflect the new ranking.
            fused_lookup = {
                idx_sorted[j]: float(rerank_scores[j]) for j in range(len(idx_sorted))
            }
            idx_sorted = sorted(idx_sorted, key=lambda i: fused_lookup[i], reverse=True)
        else:
            fused_lookup = {i: fused[i] for i in idx_sorted}

        # --- Build hits ---
        hits: list[RAGHit] = []
        for i in idx_sorted[:k]:
            hits.append(
                RAGHit(
                    text=self.texts[i],
                    source=self.metadatas[i].get("source", "unknown"),
                    score=float(fused_lookup.get(i, fused[i])),
                    metadata=self.metadatas[i],
                )
            )
        return hits


def _normalise(xs: Iterable[float]) -> list[float]:
    xs = list(xs)
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-9:
        return [0.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]
