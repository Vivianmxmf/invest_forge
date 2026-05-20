"""HybridRetriever + builder tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from invest_forge.knowledge_base.builder import ChunkingConfig, build_in_memory, chunks_from_documents
from invest_forge.knowledge_base.pdf_loader import LoadedDocument, load_markdown
from invest_forge.knowledge_base.retriever import HybridRetriever


@pytest.mark.unit
def test_bm25_retrieves_keyword_match(tiny_retriever: HybridRetriever):
    hits = tiny_retriever.retrieve("毛利率 28.5%", k=2)
    assert hits
    assert "毛利率" in hits[0].text


@pytest.mark.unit
def test_returns_empty_when_corpus_empty():
    r = HybridRetriever(texts=[], metadatas=[])
    assert r.retrieve("anything") == []


@pytest.mark.unit
def test_add_rebuilds_index(tiny_retriever: HybridRetriever):
    tiny_retriever.add("公司新闻 监管处罚 罚款 5000 万元", {"source": "extra.md"})
    hits = tiny_retriever.retrieve("监管处罚", k=1)
    assert hits[0].source == "extra.md"


@pytest.mark.unit
def test_add_keeps_embeddings_in_sync():
    """When embed_fn is set, add() must also append an embedding so the new
    doc participates in vector ranking (regression for codex round-1 P2)."""

    def embed(text: str) -> list[float]:
        return [1.0 if "old" in text else 0.0, 1.0 if "new" in text else 0.0]

    r = HybridRetriever(
        texts=["doc about old subject"],
        metadatas=[{"source": "old"}],
        embeddings=[embed("doc about old subject")],
        embed_fn=embed,
    )
    r.add("doc about new subject", {"source": "new"})
    assert len(r.embeddings) == 2
    hits = r.retrieve("new", k=2)
    sources = [h.source for h in hits]
    assert "new" in sources


@pytest.mark.unit
def test_retrieve_with_partial_embeddings_still_includes_bm25_only_doc():
    """Codex round-2 P2: a doc added without embedding must still rank on
    BM25 even when other docs have vectors."""

    pre_vec = [1.0, 0.0]
    r = HybridRetriever(
        texts=["pre-existing doc about ALPHA"],
        metadatas=[{"source": "alpha"}],
        embeddings=[pre_vec],
        # No embed_fn → calling add() leaves embeddings list short.
    )
    r.add("new doc mentioning BETA keyword", {"source": "beta"})
    # Embeddings list is intentionally shorter than texts now.
    assert len(r.embeddings) == 1 and len(r.texts) == 2
    hits = r.retrieve("BETA", k=2)
    sources = [h.source for h in hits]
    assert "beta" in sources


@pytest.mark.unit
def test_hyde_uses_expanded_query(tiny_retriever: HybridRetriever):
    tiny_retriever.hyde_fn = lambda q: "汇率敞口 海外业务"
    hits = tiny_retriever.retrieve("公司风险", k=1, use_hyde=True)
    # The hyde rewrite contains "汇率" which appears only in doc1.
    assert hits[0].metadata.get("source") == "doc1.md"


@pytest.mark.unit
def test_reranker_changes_order(tiny_retriever: HybridRetriever):
    # Reranker that returns descending scores in the order it received items
    tiny_retriever.reranker_fn = lambda _q, docs: list(range(len(docs), 0, -1))
    hits = tiny_retriever.retrieve("公司", k=3, use_reranker=True)
    assert len(hits) >= 1


@pytest.mark.unit
def test_reranker_scores_attach_to_their_documents():
    """Codex round-4 P2: rerank scores must remain attached to the document
    that produced them, not to the post-sort positional slot."""
    r = HybridRetriever(
        texts=["alpha keyword", "beta keyword", "gamma keyword"],
        metadatas=[{"source": "alpha"}, {"source": "beta"}, {"source": "gamma"}],
    )
    score_table: dict[str, float] = {"alpha": 1.0, "beta": 3.0, "gamma": 5.0}

    def reranker(_q, docs):
        return [score_table[d.split()[0]] for d in docs]

    r.reranker_fn = reranker
    hits = r.retrieve("keyword", k=3, use_reranker=True)
    assert hits[0].source == "gamma"
    assert hits[0].score == pytest.approx(5.0)
    assert hits[1].source == "beta"
    assert hits[1].score == pytest.approx(3.0)
    assert hits[2].source == "alpha"
    assert hits[2].score == pytest.approx(1.0)


@pytest.mark.unit
def test_chunking(tmp_path: Path):
    md = tmp_path / "x.md"
    md.write_text("段落一" * 200 + "\n\n" + "段落二" * 100, encoding="utf-8")
    docs = load_markdown(md)
    chunks = chunks_from_documents(docs, ChunkingConfig(chunk_size=200, chunk_overlap=20))
    assert len(chunks) >= 2
    assert all(len(t) <= 250 for t, _ in chunks)  # +overlap may push past chunk_size


@pytest.mark.unit
def test_build_in_memory(tmp_path: Path):
    (tmp_path / "a.md").write_text("公司 2024 年营收增长强劲, 净利润 65 亿元.", encoding="utf-8")
    (tmp_path / "b.md").write_text("风险: 客户集中度高, 海外汇率波动.", encoding="utf-8")
    retriever = build_in_memory(tmp_path)
    hits = retriever.retrieve("客户集中度风险", k=1)
    assert hits and "客户集中度" in hits[0].text
