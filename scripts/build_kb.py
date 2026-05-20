#!/usr/bin/env python3
"""CLI: build the InvestForge knowledge base.

Two modes:
  * default: writes an in-memory HybridRetriever snapshot to data/kb/
    (no Qdrant / no GPU needed — works on laptops).
  * ``--target qdrant``: uploads chunks to a running Qdrant collection
    (requires ``qdrant-client`` + a running instance at QDRANT_URL).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from invest_forge.common.config import get_settings  # noqa: E402
from invest_forge.knowledge_base.builder import (  # noqa: E402
    ChunkingConfig,
    build_in_memory,
    build_qdrant,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Source directory (defaults to data/pdfs, falls back to data/sample/kb)",
    )
    parser.add_argument(
        "--target",
        choices=("memory", "qdrant"),
        default="memory",
    )
    parser.add_argument("--collection", default=None, help="Override qdrant collection name")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args(argv)

    settings = get_settings()
    source = args.source
    if source is None:
        # Only count data/pdfs/ as "populated" if it has supported files —
        # ``.gitkeep`` and other dotfiles don't qualify.
        supported = (".pdf", ".md", ".markdown", ".txt")
        pdf_dir = settings.paths.pdf_dir
        has_real_docs = pdf_dir.exists() and any(
            p.is_file() and p.suffix.lower() in supported and not p.name.startswith(".")
            for p in pdf_dir.rglob("*")
        )
        source = pdf_dir if has_real_docs else (settings.paths.sample_dir / "kb")
    if not source.exists():
        print(f"error: source directory does not exist: {source}", file=sys.stderr)
        return 1

    cfg = ChunkingConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    if args.target == "memory":
        retriever = build_in_memory(source, cfg=cfg)
        print(f"OK: built in-memory KB with {len(retriever.texts)} chunks from {source}")
        return 0

    # qdrant path: need embed_fn from sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        print(
            "error: --target=qdrant requires `pip install sentence-transformers`",
            file=sys.stderr,
        )
        return 2

    model = SentenceTransformer(settings.vector_store.embedding_model)
    def embed(text: str) -> list[float]:
        return model.encode(text).tolist()

    n = build_qdrant(
        source,
        url=settings.vector_store.url,
        collection=args.collection or settings.vector_store.collection,
        embed_fn=embed,
        cfg=cfg,
        vector_size=model.get_sentence_embedding_dimension(),
        api_key=settings.vector_store.api_key,
    )
    print(f"OK: uploaded {n} chunks to qdrant://{settings.vector_store.url}/{args.collection or settings.vector_store.collection}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
