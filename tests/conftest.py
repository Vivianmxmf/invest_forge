"""Shared pytest fixtures for InvestForge."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from janus_terminal.agents.graph import GraphDeps  # noqa: E402
from janus_terminal.common.types import MacroContext, NewsItem  # noqa: E402
from janus_terminal.knowledge_base.retriever import HybridRetriever  # noqa: E402
from janus_terminal.llm.fake import FakeLLMClient  # noqa: E402
from janus_terminal.tools.data_tools import FakeDataProvider  # noqa: E402
from janus_terminal.tools.sentiment import LexiconSentiment  # noqa: E402


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def fake_data(tmp_path: Path) -> FakeDataProvider:
    """Provider backed by a tmp sample directory we control per-test."""
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "fundamentals.json").write_text(
        json.dumps(
            {
                "TEST.SH": {
                    "fiscal_year": 2024,
                    "revenue": 1.2e10,
                    "net_profit": 2e9,
                    "gross_margin": 0.4,
                    "net_margin": 0.167,
                    "roe": 0.18,
                    "roic": 0.14,
                    "debt_ratio": 0.35,
                    "free_cash_flow": 1.5e9,
                    "pe_ratio": 18.0,
                    "pb_ratio": 2.5,
                    "market_cap": 5.5e10,
                }
            }
        ),
        encoding="utf-8",
    )
    (sample_dir / "news.json").write_text(
        json.dumps(
            {
                "TEST.SH": [
                    {
                        "title": "Q1 业绩超预期 净利润同比增长 30%",
                        "published": "2025-04-15",
                        "source": "test",
                        "sentiment": 0.0,
                        "label": "neutral",
                        "summary": "盈利增长强劲, 突破市场预期",
                    },
                    {
                        "title": "海外业务下滑 汇率风险加剧",
                        "published": "2025-04-20",
                        "source": "test",
                        "sentiment": 0.0,
                        "label": "neutral",
                        "summary": "出口承压, 业务收缩",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (sample_dir / "macro.json").write_text(
        json.dumps(
            {
                "cpi_yoy": 0.5,
                "ppi_yoy": -1.0,
                "m1_yoy": 2.0,
                "m2_yoy": 8.0,
                "pmi": 51.2,
                "us10y_yield": 4.2,
                "cny_dxy": 104.0,
                "summary": "test macro snapshot",
            }
        ),
        encoding="utf-8",
    )
    return FakeDataProvider(sample_dir=sample_dir)


@pytest.fixture
def lexicon_sentiment() -> LexiconSentiment:
    return LexiconSentiment()


@pytest.fixture
def tiny_retriever() -> HybridRetriever:
    texts = [
        "公司 2024 年营业收入 432 亿元, 同比 +22%; 毛利率 28.5%, 净利率 15.1%.",
        "公司面临的核心风险包括客户集中度高、海外销售汇率敞口、行业景气度回落.",
        "宏观环境: 制造业 PMI 重回扩张, 国内通胀温和, 海外美元利率高位.",
        "Annual report excerpt: revenue grew 22% YoY, net margin expanded to 15.1%.",
    ]
    metas = [{"source": f"doc{i}.md", "page": 0} for i in range(len(texts))]
    return HybridRetriever(texts=texts, metadatas=metas)


@pytest.fixture
def deps(fake_llm, fake_data, lexicon_sentiment, tiny_retriever) -> GraphDeps:
    return GraphDeps(
        llm=fake_llm,
        data_provider=fake_data,
        sentiment=lexicon_sentiment,
        retriever=tiny_retriever,
    )


@pytest.fixture
def sample_news() -> list[NewsItem]:
    return [
        NewsItem(
            title="Q1 业绩超预期",
            published=date(2025, 4, 15),
            source="t",
            sentiment=0.4,
            label="positive",
            summary="盈利改善",
        ),
        NewsItem(
            title="海外业务下滑 汇率风险加剧",
            published=date(2025, 4, 20),
            source="t",
            sentiment=-0.3,
            label="negative",
            summary="承压",
        ),
    ]


@pytest.fixture
def sample_macro() -> MacroContext:
    return MacroContext(
        cpi_yoy=0.4, ppi_yoy=-2.7, m1_yoy=1.2, m2_yoy=7.0, pmi=50.4,
        us10y_yield=4.21, cny_dxy=104.3, summary="test",
    )
