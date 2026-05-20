"""Streamlit dashboard — paste a stock code, get a multi-agent investment view."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from invest_forge.agents.graph import GraphDeps, run_pipeline_inline  # noqa: E402
from invest_forge.api.main import _select_retriever  # noqa: E402 - reuse production logic
from invest_forge.common.config import get_settings  # noqa: E402
from invest_forge.llm.client import build_client  # noqa: E402
from invest_forge.tools.data_tools import build_provider  # noqa: E402
from invest_forge.tools.sentiment import build_sentiment  # noqa: E402

st.set_page_config(page_title="InvestForge", page_icon="🔨", layout="wide")
st.title("🔨 InvestForge — AI 主观投研助手")
st.caption("LangGraph Multi-Agent · 混合 RAG · 类 DPO 自进化 Prompt · LangSmith 可观测")


@st.cache_resource(show_spinner=False)
def _build_deps() -> GraphDeps:
    settings = get_settings()
    return GraphDeps(
        llm=build_client(),
        data_provider=build_provider(prefer_real=bool(settings.tushare_token)),
        sentiment=build_sentiment(prefer_real=False),
        retriever=_select_retriever(settings),
    )


with st.sidebar:
    st.header("配置")
    st.write("LLM 后端:", _build_deps().llm.name)
    st.write("数据提供商:", _build_deps().data_provider.name)
    st.write("情感模型:", _build_deps().sentiment.name)
    st.divider()
    st.markdown(
        """
        ### 用法
        1. 在主面板输入 A 股代码 (如 `688981.SH`).
        2. 点击 *开始分析* 触发研究员 → 分析师 → 风控 → 输出.
        3. 在底部展开看到中间产物 (memo / report / risk).
        """
    )

ts_code = st.text_input("股票代码", value="688981.SH").strip()
go = st.button("开始分析", type="primary", use_container_width=False)

if go and ts_code:
    with st.spinner(f"AI 研究团队正在分析 {ts_code} ..."):
        state = run_pipeline_inline(_build_deps(), ts_code=ts_code)

    rec = state.get("final_recommendation", {})
    col1, col2, col3 = st.columns(3)
    color = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(rec.get("rating", "HOLD"), "⚪")
    col1.metric("评级", f"{color} {rec.get('rating', 'HOLD')}")
    col2.metric("置信度", f"{rec.get('confidence', 0):.0%}")
    col3.metric("迭代轮数", f"{rec.get('iteration_count', 0)}")

    st.subheader("核心逻辑")
    st.write(rec.get("rationale", "(空)"))

    if rec.get("risk_factors"):
        st.subheader("风险因子")
        for r in rec["risk_factors"]:
            st.markdown(f"- {r}")

    with st.expander("研究员备忘录"):
        st.markdown(state.get("research_memo", "(空)"))
    with st.expander("分析师 JSON"):
        st.json(state.get("analyst_report", {}))
    with st.expander("风控评估 JSON"):
        st.json(state.get("risk_assessment", {}))
    with st.expander("RAG 命中 (Top 5)"):
        for hit in state.get("rag_evidence", []):
            st.markdown(f"**[{hit['source']}]** (score={hit['score']:.3f})")
            st.code(hit["text"][:500])
