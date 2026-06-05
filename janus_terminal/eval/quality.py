"""LLM-as-judge quality scoring for the investment-thesis output.

Used by the self-evolving prompt loop (JD6 §3 Week 3): the optimiser scores
each candidate prompt's output along three axes and feeds the score back
into the iteration.

Two scoring modes:
  * ``HeuristicScorer`` — rule-based, deterministic, used in tests.
  * ``LLMJudgeScorer`` — wraps the configured LLMClient and returns a JSON.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from janus_terminal.common.logging_setup import get_logger
from janus_terminal.llm.client import ChatMessage, LLMClient

logger = get_logger(__name__)


@dataclass(frozen=True)
class QualityScore:
    coherence: float      # 0–1: rationale internally consistent?
    grounding: float      # 0–1: numbers traceable to evidence?
    risk_coverage: float  # 0–1: risks comprehensive + specific?
    overall: float        # weighted average


def _avg(*xs: float) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _tolerant_json(text: str) -> dict | None:
    """Same tolerance as ``agents/nodes._parse_json``: strips fences + finds
    the outermost JSON object so LLM judges wrapping the answer in
    ```json ... ``` markdown still parse."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None


@dataclass
class HeuristicScorer:
    """Rule-based quality scorer for the analyst report dict."""

    def score(self, report: dict[str, Any], *, evidence: str = "") -> QualityScore:
        rationale = (report.get("rationale") or "").strip()
        risk_factors = list(report.get("risk_factors") or [])
        rating = (report.get("rating") or "").upper()
        confidence = float(report.get("confidence") or 0.0)

        # Coherence: rating consistent with confidence?  Decisive rating
        # demands higher confidence; HOLD with any confidence is fine.
        if rating in ("BUY", "SELL"):
            coherence = min(1.0, confidence * 1.3)
        else:
            coherence = 0.6 + 0.4 * (1 - abs(confidence - 0.5) * 2)

        # Grounding: how many numbers from the rationale appear in evidence?
        nums_rationale = set(re.findall(r"\d+(?:\.\d+)?", rationale))
        nums_evidence = set(re.findall(r"\d+(?:\.\d+)?", evidence)) if evidence else set()
        if not nums_rationale:
            grounding = 0.5 if evidence else 0.7
        else:
            grounding = len(nums_rationale & nums_evidence) / len(nums_rationale)

        # Risk coverage: at least three non-duplicate, non-trivial risks.
        unique_risks = {str(r).strip() for r in risk_factors if str(r).strip()}
        if len(unique_risks) >= 3:
            risk_coverage = 1.0
        elif len(unique_risks) == 2:
            risk_coverage = 0.6
        elif len(unique_risks) == 1:
            risk_coverage = 0.3
        else:
            risk_coverage = 0.0

        overall = _avg(coherence, grounding, risk_coverage)
        return QualityScore(
            coherence=coherence,
            grounding=grounding,
            risk_coverage=risk_coverage,
            overall=overall,
        )


@dataclass
class LLMJudgeScorer:
    client: LLMClient

    def score(self, report: dict[str, Any], *, evidence: str = "") -> QualityScore:
        prompt = (
            "你是评审委员会成员, 对下面的投资分析报告做质量评分.\n"
            "请输出 JSON: {coherence, grounding, risk_coverage, overall}, 每项 0-1.\n\n"
            f"# 报告\n{json.dumps(report, ensure_ascii=False, indent=2)}\n\n"
            f"# 可引证据\n{evidence[:2000]}\n"
        )
        resp = self.client.complete([ChatMessage(role="user", content=prompt)])
        data = _tolerant_json(resp.text)
        if not data:
            logger.warning("LLM judge produced non-JSON; falling back to heuristic")
            return HeuristicScorer().score(report, evidence=evidence)
        return QualityScore(
            coherence=float(data.get("coherence", 0.5)),
            grounding=float(data.get("grounding", 0.5)),
            risk_coverage=float(data.get("risk_coverage", 0.5)),
            overall=float(data.get("overall", 0.5)),
        )
