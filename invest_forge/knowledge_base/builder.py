"""Build the InvestForge knowledge base.

Two paths:
  * ``build_in_memory`` — synchronous, no Qdrant, used by tests + laptop dev.
  * ``build_qdrant`` — server path; uploads embeddings to Qdrant collection.

Chunking uses recursive character splitting with CN-friendly separators.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from invest_forge.common.logging_setup import get_logger
from invest_forge.knowledge_base.pdf_loader import LoadedDocument, iter_directory
from invest_forge.knowledge_base.retriever import HybridRetriever

logger = get_logger(__name__)


@dataclass
class ChunkingConfig:
    chunk_size: int = 800
    chunk_overlap: int = 100
    separators: tuple[str, ...] = ("\n\n", "\n", "。", "；", "，", " ")


def _split_recursive(text: str, cfg: ChunkingConfig) -> list[str]:
    """Recursive separator-aware splitter inspired by LangChain's.

    Falls back to a hard character cut when no separator fits.  This impl
    is intentionally tiny and dependency-free.
    """
    if len(text) <= cfg.chunk_size:
        return [text]

    for sep in cfg.separators:
        if sep and sep in text:
            parts = text.split(sep)
            chunks: list[str] = []
            current = ""
            for part in parts:
                piece = (current + sep + part) if current else part
                if len(piece) <= cfg.chunk_size:
                    current = piece
                    continue
                if current:
                    chunks.append(current)
                if len(part) <= cfg.chunk_size:
                    current = part
                else:
                    chunks.extend(_split_recursive(part, cfg))
                    current = ""
            if current:
                chunks.append(current)
            # Add overlap by re-walking with sliding window
            return _apply_overlap(chunks, cfg.chunk_overlap)

    # No separator helped — slice
    step = cfg.chunk_size - cfg.chunk_overlap
    return [text[i : i + cfg.chunk_size] for i in range(0, len(text), max(step, 1))]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap:] if len(prev) > overlap else prev
        out.append(tail + cur)
    return out


def chunks_from_documents(
    docs: Iterable[LoadedDocument], cfg: ChunkingConfig | None = None
) -> list[tuple[str, dict]]:
    """Flatten loaded documents into (text, metadata) chunks."""
    cfg = cfg or ChunkingConfig()
    out: list[tuple[str, dict]] = []
    for doc in docs:
        for i, chunk in enumerate(_split_recursive(doc.text, cfg)):
            md = {
                "source": doc.source,
                "page": doc.page,
                "chunk": i,
                **doc.metadata,
            }
            out.append((chunk.strip(), md))
    # Drop empty chunks
    return [(t, m) for t, m in out if t]


def build_in_memory(
    root: str | Path,
    *,
    embed_fn: Callable[[str], list[float]] | None = None,
    cfg: ChunkingConfig | None = None,
) -> HybridRetriever:
    """Walk ``root`` (recursively) → chunk → return a ready ``HybridRetriever``."""
    docs = list(iter_directory(root))
    pairs = chunks_from_documents(docs, cfg)
    texts = [t for t, _ in pairs]
    metas = [m for _, m in pairs]
    embeddings = [embed_fn(t) for t in texts] if embed_fn else []
    retriever = HybridRetriever(
        texts=texts, metadatas=metas, embeddings=embeddings, embed_fn=embed_fn
    )
    logger.info("KB built: %d chunks from %s", len(texts), root)
    return retriever


def build_qdrant(
    root: str | Path,
    *,
    url: str,
    collection: str,
    embed_fn: Callable[[str], list[float]],
    cfg: ChunkingConfig | None = None,
    vector_size: int = 512,
    api_key: str | None = None,
) -> int:  # pragma: no cover - requires Qdrant
    """Upload chunks to Qdrant. Returns number of points written."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    docs = list(iter_directory(root))
    pairs = chunks_from_documents(docs, cfg)
    if not pairs:
        logger.warning("no chunks to upload from %s", root)
        return 0

    # ``api_key=None`` is the default for unauthenticated local Qdrant; passing
    # a key is required for Qdrant Cloud / secured instances.
    client = QdrantClient(url=url, api_key=api_key)
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    points = [
        PointStruct(
            id=i,
            vector=embed_fn(text),
            payload={"text": text, **meta},
        )
        for i, (text, meta) in enumerate(pairs)
    ]
    client.upsert(collection_name=collection, points=points, wait=True)
    logger.info("uploaded %d chunks to qdrant://%s/%s", len(points), url, collection)
    return len(points)
