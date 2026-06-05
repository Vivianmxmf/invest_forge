"""Fake LLM client used by tests and the laptop demo path.

The fake routes by the *last system or user message* contents through a
registry of canned responses, so a single fake can serve the researcher,
analyst, risk-control, and HyDE nodes without any model download.

Design goals:
  * deterministic
  * records every call so tests can assert what was sent
  * exposes a register/override API so tests can inject specific responses
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Union

from janus_terminal.llm.client import ChatMessage, ChatResponse

Responder = Union[str, Callable[[list[ChatMessage]], str]]


# Default canned responses keyed by simple substring matchers on the joined prompt.
# These are intentionally small and shape-correct (valid JSON where the agent
# expects JSON) so that the full LangGraph pipeline can exercise its parsing.
# NOTE: patterns are evaluated top-to-bottom and the FIRST match wins.  Order
# them MOST-SPECIFIC FIRST: e.g. the analyst prompt contains both the
# ``[分析师]`` marker and the literal substring "研究员备忘录" — without strict
# ordering the researcher pattern would steal the analyst's response.
_DEFAULT_RESPONSES: list[tuple[re.Pattern[str], str]] = [
    (
        # Risk-control prompt is tagged with these square-bracket markers.
        re.compile(r"\[风控\]|\[Risk Control\]|\[risk assessment\]", re.I),
        json.dumps(
            {
                "risk_level": "MEDIUM",
                "compliance_ok": True,
                "notes": "估值未到极端区域, 但客户集中度风险需在报告中明确披露.",
                "require_revision": False,
            },
            ensure_ascii=False,
        ),
    ),
    (
        re.compile(r"\[分析师\]|\[Analyst\]|\[investment thesis\]", re.I),
        json.dumps(
            {
                "rating": "BUY",
                "confidence": 0.72,
                "rationale": (
                    "综合财务质量、估值分位和宏观环境, 给予买入评级. 主要逻辑: "
                    "盈利能力扩张 + 估值历史中位 + 行业拐点确认."
                ),
                "target_price": 168.5,
                "risk_factors": ["客户集中度", "汇率波动", "下游需求"],
            },
            ensure_ascii=False,
        ),
    ),
    (
        re.compile(r"\[hyde\]|\[hypothetical document\]", re.I),
        "公司 2024 年实现营业收入 432 亿元, 同比增长 22%; 归母净利润 65 亿元, 同比 +28%."
        " 综合毛利率 28.5%, 净利率 15.1%; 经营性现金流净额 89 亿元.",
    ),
    (
        re.compile(r"\[output\]|\[final report\]|\[最终报告\]", re.I),
        "InvestForge 输出报告: 建议买入, 置信度 0.72, 目标价 168.5.",
    ),
    (
        # Vision node prompt is tagged with these markers.
        re.compile(r"\[视觉\]|\[Vision\]", re.I),
        (
            "**图像分析摘要**\n\n"
            "关键财务数据: 营收 432 亿元(+22%), 净利率 15.1%, ROE 17%.\n"
            "图表趋势: K 线呈震荡上行态势, 成交量温和放大, 均线多头排列.\n"
            "风险提示: 报告中标注了客户集中度风险及海外汇率敞口风险."
        ),
    ),
    (
        # Researcher catch-all comes LAST so it doesn't eat the analyst /
        # risk-control prompts that incidentally quote "研究员".
        re.compile(r"\[研究员\]|\[Researcher\]|research memo", re.I),
        (
            "## 盈利质量\nROE 17%, ROIC 13%, 净利率扩张, 经营性现金流为正.\n\n"
            "## 成长性\n营收增速 22%, 净利润增速 28%, 高于行业中位.\n\n"
            "## 估值\nPE 22 处历史 40 分位, 相对 PEG 0.78, 估值合理.\n\n"
            "## 核心风险\n1. 客户集中度高于 60%\n2. 海外销售敞口暴露汇率波动\n3. 行业景气度仍有回落风险."
        ),
    ),
]


@dataclass
class FakeLLMClient:
    name: str = "fake-llm"
    calls: list[list[ChatMessage]] = field(default_factory=list)
    responses: list[tuple[re.Pattern[str], Responder]] = field(
        default_factory=lambda: [(p, b) for p, b in _DEFAULT_RESPONSES]
    )
    fallback: Responder = '{"rating": "HOLD", "confidence": 0.5, "rationale": "fake default"}'

    def register(self, pattern: str | re.Pattern[str], response: Responder) -> None:
        """Inject a custom response (highest priority, prepended).

        ``response`` may be a string or a callable taking the message list.
        """
        if isinstance(pattern, str):
            pattern = re.compile(pattern, re.I)
        self.responses.insert(0, (pattern, response))

    def set_fallback(self, response: Responder) -> None:
        self.fallback = response

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,  # noqa: ARG002 - parity with real clients
        max_tokens: int | None = None,      # noqa: ARG002
        response_format: str | None = None,  # noqa: ARG002
    ) -> ChatResponse:
        self.calls.append(list(messages))
        joined = "\n".join(m.content for m in messages)
        for pattern, body in self.responses:
            if pattern.search(joined):
                text = body(messages) if callable(body) else body
                return ChatResponse(text=text, usage={"prompt": len(joined), "completion": len(text)})
        text = self.fallback(messages) if callable(self.fallback) else self.fallback
        return ChatResponse(text=text, usage={"prompt": len(joined), "completion": len(text)})

    def reset(self) -> None:
        self.calls.clear()


def with_handlers(handlers: dict[str, Responder]) -> FakeLLMClient:
    """Build a FakeLLMClient seeded with extra regex→response handlers."""
    fake = FakeLLMClient()
    for pat, body in handlers.items():
        fake.register(pat, body)
    return fake
