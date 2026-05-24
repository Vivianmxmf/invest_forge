"""LangGraph node implementations.

Each node is a pure function ``InvestState -> partial dict``.  All LLM and
data-provider dependencies are passed in via closures (see ``make_*_node``)
so the test suite can swap in fakes without monkeypatching globals.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Callable

from invest_forge.agents.prompts import (
    ANALYST_PROMPT,
    OUTPUT_PROMPT,
    RESEARCHER_PROMPT,
    RISK_CONTROL_PROMPT,
    VISION_PROMPT,
)
from invest_forge.agents.state import InvestState
from invest_forge.common.logging_setup import get_logger
from invest_forge.common.types import (
    MacroContext,
    NewsItem,
    Rating,
    RAGHit,
    RiskLevel,
)
from invest_forge.knowledge_base.retriever import HybridRetriever
from invest_forge.llm.client import ChatMessage, LLMClient
from invest_forge.tools.data_tools import DataProvider
from invest_forge.tools.sentiment import Sentiment

logger = get_logger(__name__)


# ───────────────────── Helpers ─────────────────────

def _parse_json(text: str) -> dict[str, Any]:
    """Tolerant JSON parse: strips fences, finds the outermost JSON object."""
    if not text:
        return {}
    stripped = text.strip()
    if stripped.startswith("```"):
        # Strip ```json / ``` markdown fences
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    # Find the first balanced { ... } so prose around it is ignored.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned non-JSON payload: %s ; raw=%r", exc, stripped[:200])
        return {}


def _summarise_news(items: list[NewsItem], max_n: int = 5) -> str:
    if not items:
        return "(无相关新闻)"
    lines = [
        f"- {it.published} | {it.label} ({it.sentiment:+.2f}) | {it.title}"
        for it in items[:max_n]
    ]
    return "\n".join(lines)


def _summarise_rag(hits: list[RAGHit], max_n: int = 3) -> str:
    if not hits:
        return "(无 RAG 命中)"
    return "\n\n".join(f"[{h.source}] {h.text[:200].strip()}" for h in hits[:max_n])


# ───────────────────── Nodes ─────────────────────

def make_data_fetcher_node(
    data_provider: DataProvider,
    sentiment: Sentiment,
    retriever: HybridRetriever | None,
) -> Callable[[InvestState], dict]:
    """Pulls fundamentals + macro + news; scores news sentiment; pulls RAG."""

    def node(state: InvestState) -> dict:
        ts_code = state.get("ts_code")
        if not ts_code:
            return {"errors": ["data_fetcher: missing ts_code"]}

        fundamentals = asdict(data_provider.fundamentals(ts_code))
        macro = asdict(data_provider.macro())
        raw_news = data_provider.news(ts_code, limit=10)

        # Re-score sentiment with our backend (provider may return zeros).
        scored: list[NewsItem] = []
        for n in raw_news:
            score, label = sentiment.score(f"{n.title} {n.summary or ''}")
            scored.append(
                NewsItem(
                    title=n.title,
                    published=n.published,
                    source=n.source,
                    sentiment=score,
                    label=label,
                    summary=n.summary,
                )
            )

        # Hit the KB if available.  Retrieve a wide candidate set, then boost
        # documents whose metadata (filename / source / explicit ts_code field)
        # matches the requested ticker so we never return excerpts about the
        # wrong issuer when the corpus only carries ts_codes in filenames.
        rag_query = f"{ts_code} 公司基本面 风险 估值"
        hits: list[dict] = []
        if retriever is not None:
            symbol = ts_code.split(".", 1)[0]
            raw_hits = retriever.retrieve(rag_query, k=15)
            ranked = sorted(
                raw_hits,
                key=lambda h: (
                    1 if _ticker_in_metadata(h, ts_code, symbol) else 0,
                    h.score,
                ),
                reverse=True,
            )[:5]
            hits = [
                {"text": h.text, "source": h.source, "score": h.score, "metadata": h.metadata}
                for h in ranked
            ]

        return {
            "fundamental_data": fundamentals,
            "macro_context": macro,
            "news_items": [_news_to_dict(n) for n in scored],
            "rag_evidence": hits,
        }

    return node


def make_vision_node(client: LLMClient | None) -> Callable[[InvestState], dict]:
    """Return a node that calls the VLM to analyse uploaded report/chart images.

    The node is INERT (returns ``{}``) in two cases:
      * ``client`` is ``None`` — vision VLM not configured (no-op, default).
      * ``state["input_images"]`` is empty — no images in this request.

    When both conditions are met the node calls ``client.complete`` with a
    ``ChatMessage`` whose ``images`` tuple carries the validated data-URLs,
    and writes the response text to ``vision_analysis``.
    """

    def node(state: InvestState) -> dict:
        if client is None:
            return {}
        images: list[str] = state.get("input_images") or []
        if not images:
            return {}
        msg = ChatMessage(
            role="user",
            content=VISION_PROMPT,
            images=tuple(images),
        )
        resp = client.complete([msg])
        return {"vision_analysis": resp.text}

    return node


def make_researcher_node(client: LLMClient) -> Callable[[InvestState], dict]:
    def node(state: InvestState) -> dict:
        items = [_news_from_dict(n) for n in state.get("news_items", [])]
        base_prompt = RESEARCHER_PROMPT.format(
            fundamentals=json.dumps(state.get("fundamental_data", {}), ensure_ascii=False, indent=2),
            macro=json.dumps(state.get("macro_context", {}), ensure_ascii=False, indent=2),
            news_summary=_summarise_news(items),
            rag_evidence=_summarise_rag([_rag_from_dict(r) for r in state.get("rag_evidence", [])]),
        )
        vision_analysis: str = state.get("vision_analysis") or ""
        if vision_analysis:
            # Append vision evidence ONLY when the vision node ran — keeping the
            # no-image path byte-identical to the original RESEARCHER_PROMPT.format(...)
            prompt = base_prompt + f"\n\n# 图像分析证据\n{vision_analysis}"
        else:
            prompt = base_prompt
        resp = client.complete([ChatMessage(role="user", content=prompt)])
        return {"research_memo": resp.text}

    return node


def make_analyst_node(client: LLMClient) -> Callable[[InvestState], dict]:
    def node(state: InvestState) -> dict:
        risk_feedback = ""
        if state.get("iteration_count", 0) > 0 and state.get("risk_assessment"):
            risk_feedback = (
                "上一轮风控意见: " + state["risk_assessment"].get("notes", "")
            )
        prompt = ANALYST_PROMPT.format(
            research_memo=state.get("research_memo", "")[:4000],
            risk_feedback=risk_feedback,
        )
        resp = client.complete(
            [ChatMessage(role="user", content=prompt)],
            response_format="json",
        )
        parsed = _parse_json(resp.text)
        # Normalise the rating into the canonical enum value.
        if "rating" in parsed:
            parsed["rating"] = Rating.parse(parsed["rating"]).value
        # Confidence defensive clamp.
        if isinstance(parsed.get("confidence"), (int, float)):
            parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
        return {
            "analyst_report": parsed,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    return node


def make_risk_control_node(client: LLMClient) -> Callable[[InvestState], dict]:
    def node(state: InvestState) -> dict:
        prompt = RISK_CONTROL_PROMPT.format(
            analyst_report=json.dumps(state.get("analyst_report", {}), ensure_ascii=False, indent=2),
            research_memo=state.get("research_memo", "")[:2000],
        )
        resp = client.complete(
            [ChatMessage(role="user", content=prompt)],
            response_format="json",
        )
        parsed = _parse_json(resp.text)
        # Defensive normalisation
        raw_level = (parsed.get("risk_level") or "MEDIUM").upper()
        try:
            risk_level = RiskLevel(raw_level)
        except ValueError:
            risk_level = RiskLevel.MEDIUM
        parsed["risk_level"] = risk_level.value
        # Coerce booleans early so should_revise() doesn't see "false" strings.
        if "require_revision" in parsed:
            parsed["require_revision"] = _as_bool(parsed["require_revision"])
        else:
            parsed["require_revision"] = risk_level == RiskLevel.HIGH
        if "compliance_ok" in parsed:
            parsed["compliance_ok"] = _as_bool(parsed["compliance_ok"])
        else:
            parsed["compliance_ok"] = risk_level != RiskLevel.HIGH
        return {"risk_assessment": parsed}

    return node


def make_output_node() -> Callable[[InvestState], dict]:
    """Pure deterministic assembly — no LLM needed."""

    def node(state: InvestState) -> dict:
        analyst = state.get("analyst_report", {})
        risk = state.get("risk_assessment", {})
        # Preserve a legitimate confidence=0 (analyst clamp can produce it);
        # only fall back to 0.5 when the value is missing or non-numeric.
        raw_conf = analyst.get("confidence")
        confidence = float(raw_conf) if isinstance(raw_conf, (int, float)) else 0.5
        rec = {
            "rating": analyst.get("rating", Rating.HOLD.value),
            "confidence": confidence,
            "rationale": analyst.get("rationale", ""),
            "target_price": analyst.get("target_price"),
            "risk_factors": analyst.get("risk_factors", []),
            "risk_level": risk.get("risk_level", RiskLevel.MEDIUM.value),
            "compliance_ok": risk.get("compliance_ok", True),
            "risk_notes": risk.get("notes", ""),
            "iteration_count": state.get("iteration_count", 0),
        }
        return {"final_recommendation": rec}

    return node


# ───────────────────── Routing ─────────────────────

def _as_bool(v: Any) -> bool:
    """Lenient boolean parser — handles ``True``, ``"false"``, ``0``, etc."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes", "y", "on"}
    return False


def should_revise(state: InvestState, *, max_iterations: int = 2) -> str:
    risk = state.get("risk_assessment", {}) or {}
    require = _as_bool(risk.get("require_revision", False))
    high = (risk.get("risk_level") or "").upper() == RiskLevel.HIGH.value
    iters = state.get("iteration_count", 0)
    if (require or high) and iters < max_iterations:
        return "revise"
    return "finalize"


# ───────────────────── small dict helpers ─────────────────────

def _news_to_dict(item: NewsItem) -> dict[str, Any]:
    return {
        "title": item.title,
        "published": item.published.isoformat(),
        "source": item.source,
        "sentiment": item.sentiment,
        "label": item.label,
        "summary": item.summary,
    }


def _news_from_dict(raw: dict[str, Any]) -> NewsItem:
    from datetime import date

    return NewsItem(
        title=raw["title"],
        published=date.fromisoformat(raw["published"]),
        source=raw.get("source", ""),
        sentiment=float(raw.get("sentiment", 0.0)),
        label=raw.get("label", "neutral"),
        summary=raw.get("summary"),
    )


def _rag_from_dict(raw: dict[str, Any]) -> RAGHit:
    return RAGHit(
        text=raw.get("text", ""),
        source=raw.get("source", "unknown"),
        score=float(raw.get("score", 0.0)),
        metadata=raw.get("metadata", {}),
    )


def _ticker_in_metadata(hit: RAGHit, ts_code: str, symbol: str) -> bool:
    """True iff the hit's text or metadata (source, filename, ts_code field)
    contains the ticker or its bare 6-digit symbol."""
    blob_parts: list[str] = [hit.source or "", hit.text or ""]
    md = hit.metadata or {}
    for v in md.values():
        if isinstance(v, str):
            blob_parts.append(v)
    blob = " ".join(blob_parts)
    return ts_code in blob or symbol in blob
