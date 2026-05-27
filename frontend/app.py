"""Streamlit dashboard — paste a stock code, get a multi-agent investment view."""
from __future__ import annotations

import base64
import dataclasses
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from invest_forge.agents.graph import GraphDeps, run_pipeline_inline  # noqa: E402
from invest_forge.api.main import _select_retriever  # noqa: E402 - reuse production logic
from invest_forge.common.config import get_settings  # noqa: E402
from invest_forge.llm.client import build_client, build_vision_client  # noqa: E402
from invest_forge.tools.backtest_tools import (  # noqa: E402
    load_price_panel,
    make_lookahead_safe_signal,
    portfolio_long_short,
)
from invest_forge.tools.data_tools import build_provider  # noqa: E402
from invest_forge.tools.image_input import (  # noqa: E402
    ImagePolicy,
    ImageRef,
    ImageValidationError,
    load_images,
)
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
        vision_client=build_vision_client(settings.llm),
    )


with st.sidebar:
    st.header("配置")
    st.write("LLM 后端:", _build_deps().llm.name)
    st.write("数据提供商:", _build_deps().data_provider.name)
    st.write("情感模型:", _build_deps().sentiment.name)
    _vision_client = _build_deps().vision_client
    if _vision_client is not None:
        st.write("视觉后端:", _vision_client.name)
    else:
        st.write("视觉后端:", "disabled (set LLM_PROVIDER=local + LOCAL_VISION_MODEL)")
    st.divider()
    st.markdown(
        """
        ### 用法
        1. 在主面板输入 A 股代码 (如 `688981.SH`).
        2. (可选) 上传研报或图表图片以启用视觉分析.
        3. 点击 *开始分析* 触发研究员 → 分析师 → 风控 → 输出.
        4. 在底部展开看到中间产物 (memo / report / risk).
        """
    )

ts_code = st.text_input("股票代码", value="688981.SH").strip()

uploaded_files = st.file_uploader(
    "上传研报/图表 (可选)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

go = st.button("开始分析", type="primary", use_container_width=False)

if go and ts_code:
    # ── Build image data-URLs from uploaded files ──────────────────────────
    data_urls: list[str] = []
    if uploaded_files:
        policy = ImagePolicy.from_env()
        # Enforce count + per-file size BEFORE reading bytes into memory, so a
        # large or numerous batch is rejected up front rather than after we
        # have already read + base64-encoded every file.
        if len(uploaded_files) > policy.max_images:
            st.error(f"最多上传 {policy.max_images} 张图片，当前 {len(uploaded_files)} 张。")
        elif any((uf.size or 0) > policy.max_bytes for uf in uploaded_files):
            st.error(f"单张图片不得超过 {policy.max_bytes // 1_000_000} MB。")
        else:
            refs = [
                ImageRef(kind="base64", value=base64.b64encode(uf.read()).decode("ascii"))
                for uf in uploaded_files
            ]
            try:
                validated = load_images(refs, policy)
                data_urls = [v.data_url for v in validated]
            except ImageValidationError:
                st.error("图片校验失败: 请检查图片格式、大小或数量是否符合要求。")
                # Skip the image batch; data_urls stays empty — don't crash.

    # ── Resolve effective deps (offline demo fallback) ─────────────────────
    deps = _build_deps()
    settings = get_settings()
    if deps.vision_client is None and data_urls and settings.llm.provider == "fake":
        # No real VLM configured, but images were uploaded and we are in fake
        # mode.  Use the fake LLM client as a stand-in so the vision node runs
        # and returns its canned [视觉] placeholder instead of silently dropping
        # the images.
        deps = dataclasses.replace(deps, vision_client=deps.llm)
        st.info(
            "视觉后端未配置真实 VLM — 离线演示使用 fake 客户端的占位响应；"
            "生产环境设置 LOCAL_VISION_MODEL。"
        )

    # ── Show thumbnails ONLY for images that passed validation ─────────────
    # Gating on data_urls avoids feeding rejected/oversized bytes to Streamlit's
    # image decoder (load_images is all-or-nothing, so a non-empty data_urls
    # means every uploaded file validated).
    if data_urls:
        thumb_cols = st.columns(min(len(uploaded_files), 4))
        for col, uf in zip(thumb_cols, uploaded_files):
            uf.seek(0)  # reset after earlier read()
            col.image(uf, width=160)

    # ── Run pipeline ───────────────────────────────────────────────────────
    with st.spinner(f"AI 研究团队正在分析 {ts_code} ..."):
        state = run_pipeline_inline(
            deps,
            ts_code=ts_code,
            input_images=data_urls or None,
        )

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

    vision_analysis = state.get("vision_analysis")
    if vision_analysis:
        with st.expander("视觉分析 (VLM)"):
            st.markdown(vision_analysis)

# ── Gated backtest section (pure pandas, no alphalens) ────────────────────────
st.divider()
with st.expander("📈 5-股票回测 (sample)"):
    st.caption(
        "基于 data/sample/prices.csv 的5支股票样本，使用纯 pandas 截面 long-short 回测。"
        "无需 alphalens 或 API 密钥。"
    )
    bt_window = st.slider("MA 窗口 (天)", min_value=5, max_value=60, value=20, step=5)
    bt_top_n = st.slider("每侧持仓数 (top-N)", min_value=1, max_value=2, value=2, step=1)
    run_bt = st.button("运行回测", key="run_backtest_btn")

    if run_bt:
        try:
            _settings = get_settings()
            _prices_csv = _settings.paths.sample_dir / "prices.csv"
            _panel = load_price_panel(_prices_csv)
            _factor_wide = _panel.apply(
                lambda col: make_lookahead_safe_signal(col, bt_window)
            )
            _bt_result = portfolio_long_short(_panel, _factor_wide, top_n=bt_top_n)

            # ── Metrics ──────────────────────────────────────────────────────
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("IC Mean", f"{_bt_result.ic_mean:+.4f}")
            m2.metric("IC IR", f"{_bt_result.ic_ir:+.4f}")
            m3.metric("年化收益", f"{_bt_result.annualised_return:+.2%}")
            m4.metric("Sharpe", f"{_bt_result.sharpe:+.4f}")
            m5.metric("最大回撤", f"{_bt_result.max_drawdown:+.2%}")

            # ── Equity curve ────────────────────────────────────────────────
            _weights = (
                _factor_wide.shift(1)
                .rank(axis=1, ascending=False)
                .le(min(bt_top_n, max(1, len(_panel.columns) // 2)))
                .astype(float)
            )
            _short_mask = (
                _factor_wide.shift(1)
                .rank(axis=1, ascending=True)
                .le(min(bt_top_n, max(1, len(_panel.columns) // 2)))
            )
            _w = _weights - _short_mask.astype(float)
            _w_sum = _w.abs().sum(axis=1).replace(0, float("nan"))
            _w = _w.div(_w_sum, axis=0).fillna(0)
            _daily_ret = (_panel.pct_change() * _w).sum(axis=1).dropna()
            _equity = (1.0 + _daily_ret).cumprod()
            _equity.name = "净值"

            st.subheader("策略净值曲线")
            st.line_chart(_equity)
            st.caption(
                f"股票池: {', '.join(_panel.columns.tolist())}  |  "
                f"日期: {_panel.index[0].date()} → {_panel.index[-1].date()}  |  "
                f"N obs: {_bt_result.n_obs}"
            )
        except Exception as _bt_exc:
            st.warning(f"回测出错: {_bt_exc}")
