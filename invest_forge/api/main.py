"""FastAPI entry-point for InvestForge.

POST /analyze {"ts_code": "688981.SH"} →
    full structured InvestState (recommendation + intermediate artifacts).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from invest_forge.agents.graph import GraphDeps, run_pipeline_inline
from invest_forge.common.config import get_settings
from invest_forge.common.logging_setup import get_logger
from invest_forge.knowledge_base.builder import build_in_memory
from invest_forge.knowledge_base.retriever import HybridRetriever
from invest_forge.llm.client import build_analyst_client, build_client
from invest_forge.tools.data_tools import build_provider
from invest_forge.tools.sentiment import build_sentiment

logger = get_logger(__name__)


class AnalyzeRequest(BaseModel):
    ts_code: str = Field(..., examples=["688981.SH"])


class AnalyzeResponse(BaseModel):
    ts_code: str
    final_recommendation: dict[str, Any]
    research_memo: str
    analyst_report: dict[str, Any]
    risk_assessment: dict[str, Any]
    iteration_count: int


def _select_retriever(settings) -> HybridRetriever | None:
    """Prefer the configured production KB over the synthetic sample.

    Selection order:
      1. Qdrant collection at ``QDRANT_URL`` (production vector store);
      2. ``KB_DIR`` directory if populated with supported docs;
      3. ``PDF_DIR`` if populated;
      4. Sample KB (offline / demo);
      5. None — RAG layer is disabled.
    """
    supported = (".pdf", ".md", ".markdown", ".txt")

    def _populated(d) -> bool:
        return d.exists() and any(
            p.is_file() and p.suffix.lower() in supported and not p.name.startswith(".")
            for p in d.rglob("*")
        )

    qd = _try_qdrant_retriever(settings)
    if qd is not None:
        return qd

    for candidate in (settings.paths.kb_dir, settings.paths.pdf_dir):
        if _populated(candidate):
            logger.info("API retriever: building from production source %s", candidate)
            return build_in_memory(candidate)

    sample = settings.paths.sample_dir / "kb"
    if _populated(sample):
        logger.warning("API retriever: falling back to SAMPLE KB at %s (demo mode)", sample)
        return build_in_memory(sample)
    return None


def _try_qdrant_retriever(settings) -> HybridRetriever | None:
    """Materialise a Qdrant collection into a HybridRetriever.

    Returns None when the qdrant-client library, the URL, or the collection
    is unavailable — callers then fall back to local directories.  We scroll
    the whole collection because the in-process HybridRetriever needs the
    text payloads; for very large collections, switch this to a live-query
    adapter (tracked in CLAUDE.md TODO).
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return None
    if not settings.vector_store.url:
        return None
    try:
        client = QdrantClient(
            url=settings.vector_store.url,
            api_key=settings.vector_store.api_key,
            timeout=5.0,
        )
        if not client.collection_exists(settings.vector_store.collection):
            return None
        texts: list[str] = []
        metas: list[dict] = []
        embeddings: list[list[float]] = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=settings.vector_store.collection,
                with_payload=True,
                with_vectors=True,
                limit=512,
                offset=offset,
            )
            if not points:
                break
            for p in points:
                payload = dict(p.payload or {})
                text = payload.pop("text", "") or ""
                if text:
                    texts.append(text)
                    metas.append(payload)
                    if p.vector is not None:
                        embeddings.append(list(p.vector))
            if offset is None:
                break
        if not texts:
            return None
        logger.info(
            "API retriever: connected to Qdrant collection %s (%d chunks)",
            settings.vector_store.collection,
            len(texts),
        )

        # Attach the same embedding model used to build the collection so the
        # vector leg of HybridRetriever is actually exercised.  Loading the
        # model is best-effort: if sentence-transformers / weights are
        # unavailable, fall back to BM25-only ranking.
        embed_fn = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer(settings.vector_store.embedding_model)

            def embed_fn(text: str) -> list[float]:  # noqa: F811 - intentional
                return model.encode(text).tolist()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "embedding model %s unavailable (%s); Qdrant RAG will use BM25 only",
                settings.vector_store.embedding_model, exc,
            )

        return HybridRetriever(
            texts=texts, metadatas=metas, embeddings=embeddings, embed_fn=embed_fn
        )
    except Exception as exc:  # noqa: BLE001 - network / auth / schema drift
        logger.warning("Qdrant retriever unavailable (%s); falling back to local KB", exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    default_client = build_client()
    # Only attach a distinct analyst_client when there's a real override
    # (provider=local AND LOCAL_ANALYST_MODEL set).  Otherwise leave it None so
    # GraphDeps' ``analyst_client or llm`` fallback keeps every node on the
    # default client — identical behaviour to fake/openai/anthropic today.
    if settings.llm.provider == "local" and settings.llm.local_analyst_model:
        analyst_client = build_analyst_client(settings.llm)
    else:
        analyst_client = None
    app.state.deps = GraphDeps(
        llm=default_client,
        data_provider=build_provider(prefer_real=bool(settings.tushare_token)),
        sentiment=build_sentiment(prefer_real=False),
        retriever=_select_retriever(settings),
        analyst_client=analyst_client,
    )
    logger.info(
        "InvestForge API ready (LLM=%s, analyst=%s)",
        app.state.deps.llm.name,
        app.state.deps.analyst_client.name if app.state.deps.analyst_client else "shared",
    )
    yield


app = FastAPI(title="InvestForge", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "llm": app.state.deps.llm.name}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if not req.ts_code or "." not in req.ts_code:
        raise HTTPException(status_code=422, detail="ts_code must be like '688981.SH'")
    state = run_pipeline_inline(app.state.deps, ts_code=req.ts_code)
    return AnalyzeResponse(
        ts_code=req.ts_code,
        final_recommendation=state.get("final_recommendation", {}),
        research_memo=state.get("research_memo", ""),
        analyst_report=state.get("analyst_report", {}),
        risk_assessment=state.get("risk_assessment", {}),
        iteration_count=int(state.get("iteration_count", 0)),
    )
