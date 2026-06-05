"""Runtime configuration via env vars + .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # pragma: no cover
    pass


def _path(key: str, default: str) -> Path:
    return Path(os.getenv(key, default)).expanduser().resolve()


def _int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"env var {key}={raw!r} is not an integer") from exc


def _bool(key: str, default: bool) -> bool:
    raw = (os.getenv(key, "") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


@dataclass(frozen=True)
class LLMConfig:
    provider: str           # "fake" | "openai" | "anthropic" | "local"
    model: str
    openai_api_key: str | None
    anthropic_api_key: str | None
    gemini_api_key: str | None
    local_base_url: str | None
    local_model: str | None
    # When provider=local, the analyst node targets this named adapter instead
    # of the base model.  Leave as None to use the same model for all nodes.
    local_analyst_model: str | None = None
    temperature: float = 0.1
    # HuggingFace in-process provider settings (provider="hf").
    # device_map passed to AutoModelForCausalLM.from_pretrained.
    hf_device: str = "auto"
    # dtype string: "auto" | "bf16" | "fp16" | "fp32".
    hf_dtype: str = "auto"
    # Vision VLM served at a SEPARATE vLLM endpoint (server-only).
    # When provider=local and local_vision_model is set, build_vision_client()
    # returns an OpenAILLMClient targeting this model; otherwise returns None
    # and the vision node is a no-op.
    local_vision_base_url: str | None = None
    local_vision_model: str | None = None


@dataclass(frozen=True)
class VectorStoreConfig:
    url: str
    api_key: str | None
    collection: str
    embedding_model: str
    reranker_model: str


@dataclass(frozen=True)
class PathConfig:
    data_root: Path
    raw_dir: Path
    pdf_dir: Path
    kb_dir: Path
    sample_dir: Path


@dataclass(frozen=True)
class ObservabilityConfig:
    langsmith_tracing: bool
    langsmith_api_key: str | None
    langsmith_project: str


@dataclass(frozen=True)
class Settings:
    llm: LLMConfig
    vector_store: VectorStoreConfig
    paths: PathConfig
    observability: ObservabilityConfig
    tushare_token: str | None
    log_level: str
    api_host: str
    api_port: int
    dashboard_port: int
    # When True the API uses the compiled LangGraph runtime (``build_invest_graph``)
    # instead of the in-process inline runner.  Default False so the whole
    # pipeline works offline without langgraph installed.
    use_langgraph: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    paths = PathConfig(
        data_root=_path("DATA_ROOT", "./data"),
        raw_dir=_path("RAW_DIR", "./data/raw"),
        pdf_dir=_path("PDF_DIR", "./data/pdfs"),
        kb_dir=_path("KB_DIR", "./data/kb"),
        sample_dir=_path("SAMPLE_DIR", "./data/sample"),
    )
    llm = LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "fake").lower(),
        model=os.getenv("LLM_MODEL", "fake-llm"),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        local_base_url=os.getenv("LOCAL_LLM_BASE_URL") or None,
        local_model=os.getenv("LOCAL_LLM_MODEL") or None,
        local_analyst_model=os.getenv("LOCAL_ANALYST_MODEL") or None,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1") or "0.1"),
        hf_device=os.getenv("HF_DEVICE", "auto") or "auto",
        hf_dtype=os.getenv("HF_DTYPE", "auto") or "auto",
        local_vision_base_url=os.getenv("LOCAL_VISION_BASE_URL") or None,
        local_vision_model=os.getenv("LOCAL_VISION_MODEL") or None,
    )
    vector_store = VectorStoreConfig(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        collection=os.getenv("QDRANT_COLLECTION", "janus_terminal_kb"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
    )
    observability = ObservabilityConfig(
        langsmith_tracing=_bool("LANGCHAIN_TRACING_V2", False),
        langsmith_api_key=os.getenv("LANGCHAIN_API_KEY") or None,
        langsmith_project=os.getenv("LANGCHAIN_PROJECT", "janus_terminal"),
    )
    return Settings(
        llm=llm,
        vector_store=vector_store,
        paths=paths,
        observability=observability,
        tushare_token=os.getenv("TUSHARE_TOKEN") or None,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=_int("API_PORT", 8001),
        dashboard_port=_int("DASHBOARD_PORT", 8501),
        use_langgraph=_bool("USE_LANGGRAPH", False),
    )
