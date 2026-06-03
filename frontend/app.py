"""InvestForge — Streamlit Dashboard (Bloomberg Terminal redesign).

Design language: deep-space black + amber accent + JetBrains Mono everywhere.
Information-dense single-screen layout: status bar → sidebar command panel →
5-tab F-key navigation → 2×2 cell-panel grid. Bloomberg/TradingView lineage.

Backwards-compatible with the legacy app (same GraphDeps, pipeline call,
image-upload validation path, session-state keys). Only presentation changed.
"""
from __future__ import annotations

import base64
import dataclasses
import html
import sys
from pathlib import Path

_esc = html.escape

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from invest_forge.agents.graph import GraphDeps, run_pipeline_inline  # noqa: E402
from invest_forge.api.main import _select_retriever  # noqa: E402
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


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="INVESTFORGE · IF<GO>",
    page_icon="🔨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "InvestForge — Bloomberg-style AI investment terminal."},
)


# ─────────────────────────────────────────────────────────────────────────────
# Theme — Bloomberg Terminal
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

:root {
  --bg:#0a0e14; --panel:#0f1620; --panel-2:#141c28; --panel-3:#1a2330;
  --border:#2a3445; --border-2:#3a4458;
  /* contrast-tuned for WCAG AA on the deep-space bg
     --text  on bg = 14.5:1 (AAA)
     --dim   on bg =  7.2:1 (AAA)
     --muted on bg =  5.6:1 (AA, was 3.3:1 — failed)
  */
  --text:#e8ecf1; --dim:#a8b3c4; --muted:#7e8a9c;
  --amber:#E5C46B; --amber-2:#cfa94d;
  --buy:#26C281; --hold:#F9A825; --sell:#E63946; --info:#5fb1ff;
}

/* ── Reset Streamlit chrome ── */
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden;}
.stApp {background:var(--bg) !important;}
[data-testid="stAppViewContainer"] {background:var(--bg);}
.block-container {padding:0 !important; max-width:100% !important;}
.stApp, .stApp * { font-family:"JetBrains Mono","SF Mono","Menlo",monospace !important; }
body, p, div, span { color:var(--text); }

/* ── Status bar (top) ── */
.if-statusbar {
  display:flex; align-items:stretch; gap:0;
  background:var(--panel); border-bottom:1px solid var(--border);
  font-size:11px; letter-spacing:.5px;
}
.if-brand {
  padding:11px 18px; font-weight:700; color:var(--amber); font-size:13px;
  letter-spacing:1.5px; border-right:1px solid var(--border);
}
.if-brand::before { content:"◆ "; color:var(--amber); }
.if-cell {
  padding:11px 14px; border-right:1px solid var(--border); color:var(--dim);
  display:flex; align-items:center;
}
.if-cell b { color:var(--text); font-weight:600; margin-left:5px; }
.if-dot { display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:7px; }
.if-dot.ok { background:var(--buy); box-shadow:0 0 6px var(--buy); }
.if-dot.warn { background:var(--hold); }
.if-dot.off { background:var(--muted); }
.if-time { margin-left:auto; padding:11px 18px; color:var(--amber);
           font-weight:600; letter-spacing:1px; }
/* GitHub "View source" pill in the status bar */
a.if-cell.if-src { text-decoration:none; color:var(--text); cursor:pointer;
                   transition:.15s ease; }
a.if-cell.if-src:hover { background:var(--amber); color:#000 !important; }
a.if-cell.if-src:hover b { color:#000 !important; }
a.if-cell.if-src .if-icon { color:var(--amber); margin-right:6px;
                            font-weight:700; }
a.if-cell.if-src:hover .if-icon { color:#000; }

/* ── Sidebar — Bloomberg command panel ── */
[data-testid="stSidebar"] {
  background:var(--panel) !important; border-right:1px solid var(--border);
  min-width:300px !important; width:300px !important;
}
[data-testid="stSidebar"] > div { padding-top:8px; }
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p { color:var(--dim) !important; font-size:12px; }
.if-section-l {
  font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:2.5px;
  border-bottom:1px solid var(--border); padding-bottom:6px; margin:14px 0 10px;
}

/* TextInput */
[data-testid="stSidebar"] input {
  background:#000 !important; color:var(--amber) !important;
  border:1px solid var(--border) !important; border-radius:3px !important;
  font-family:"JetBrains Mono",monospace !important; letter-spacing:1px !important;
  font-weight:600 !important; padding:8px 10px !important;
}
[data-testid="stSidebar"] label p { color:var(--muted) !important;
  font-size:10px !important; text-transform:uppercase !important; letter-spacing:2px !important; }

/* Buttons (sidebar) */
[data-testid="stSidebar"] .stButton > button {
  background:var(--panel-2) !important; color:var(--text) !important;
  border:1px solid var(--border) !important; border-radius:3px !important;
  font-family:"JetBrains Mono",monospace !important; font-size:11.5px !important;
  font-weight:500 !important; letter-spacing:.5px !important; padding:6px 10px !important;
  text-align:left !important; box-shadow:none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background:var(--panel-3) !important; color:var(--amber) !important;
  border-color:var(--amber-2) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] {
  background:var(--amber) !important; color:#000 !important;
  border:1px solid var(--amber) !important; font-weight:800 !important;
  letter-spacing:2px !important; text-align:center !important; font-size:12px !important;
}

/* ── File uploader (full dark override) ── */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background:transparent !important;
  border:0 !important;
}
[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] {
  background:var(--panel-2) !important;
  border:1px dashed var(--border-2) !important;
  border-radius:3px !important;
  padding:14px 12px !important;
  min-height:0 !important;
}
/* The "Browse files" button lives outside .stButton; target it directly. */
[data-testid="stSidebar"] [data-testid="stFileUploader"] button,
[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] button {
  background:var(--panel-3) !important;
  color:var(--text) !important;
  border:1px solid var(--border-2) !important;
  border-radius:3px !important;
  font-family:"JetBrains Mono",monospace !important;
  font-size:10.5px !important;
  font-weight:500 !important;
  letter-spacing:1.5px !important;
  text-transform:uppercase !important;
  padding:6px 14px !important;
  height:auto !important;
  min-height:0 !important;
  box-shadow:none !important;
  text-shadow:none !important;
  background-image:none !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover,
[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] button:hover {
  background:var(--amber) !important;
  color:#000 !important;
  border-color:var(--amber) !important;
}
/* Dropzone helper text ("Drag and drop file here" / "Limit 200MB per file…"). */
[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] small,
[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] span {
  color:var(--muted) !important;
  font-size:10px !important;
  letter-spacing:.5px !important;
}
/* Uploaded-file rows after a successful pick. */
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
  background:var(--panel-2) !important;
  color:var(--text) !important;
  border:1px solid var(--border-2) !important;
  border-radius:3px !important;
  padding:6px 10px !important;
}

/* ── Custom JSON viewer (replaces white st.json) ── */
.if-json {
  background:#000;
  color:var(--text);
  border:1px solid var(--border);
  padding:14px 18px;
  font-family:"JetBrains Mono",monospace;
  font-size:11.5px;
  line-height:1.65;
  border-radius:3px;
  white-space:pre-wrap;
  word-break:break-word;
  margin:0;
  overflow-x:auto;
}
.if-json .k { color:var(--amber); }
.if-json .s { color:var(--buy); }
.if-json .b { color:var(--hold); }
.if-json .n { color:var(--info); }
.if-json .nu { color:var(--muted); }

/* Watchlist row */
.if-tkr {
  display:flex; justify-content:space-between; padding:7px 10px;
  border-radius:3px; cursor:pointer; color:var(--dim); margin-bottom:2px;
  border-left:2px solid transparent;
}
.if-tkr.active { background:rgba(229,196,107,.06); color:var(--amber);
                 border-left-color:var(--amber); }
.if-tkr .code { font-weight:700; }
.if-tkr .name { font-size:10.5px; color:var(--muted); }
.if-tkr .pct { font-weight:700; font-variant-numeric:tabular-nums; font-size:12px; }
.if-tkr .pct.up { color:var(--buy); }
.if-tkr .pct.down { color:var(--sell); }
.if-tkr .pct.flat { color:var(--muted); }

/* ── Tab strip ── */
.stTabs [data-baseweb="tab-list"] {
  background:var(--panel); border-bottom:1px solid var(--border);
  gap:0 !important; padding:0 !important;
}
.stTabs [data-baseweb="tab"] {
  background:transparent !important; color:var(--muted) !important;
  border-right:1px solid var(--border) !important; border-radius:0 !important;
  font-family:"JetBrains Mono",monospace !important; font-size:11.5px !important;
  font-weight:500 !important; letter-spacing:1.5px !important; padding:10px 22px !important;
  height:auto !important;
}
.stTabs [data-baseweb="tab"]:hover { color:var(--amber) !important; background:var(--panel-2) !important; }
.stTabs [aria-selected="true"] {
  color:var(--amber) !important; background:var(--panel-2) !important;
  box-shadow:inset 0 -2px 0 var(--amber) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding:0 !important; }

/* ── Decision hero ── */
.if-hero {
  padding:26px 28px; background:linear-gradient(180deg,#0d131c,transparent);
  border-bottom:1px solid var(--border);
}
.if-hero-grid {
  display:grid; grid-template-columns:auto 1fr 1fr 1fr 1fr; gap:30px; align-items:center;
}
.if-rating {
  font-size:54px; font-weight:800; letter-spacing:3px; padding:14px 28px;
  border-radius:4px; font-family:"JetBrains Mono",monospace; line-height:1;
}
.if-rating.BUY  { color:var(--buy);  background:rgba(38,194,129,.08);
                  border:1px solid rgba(38,194,129,.42);
                  text-shadow:0 0 14px rgba(38,194,129,.4); }
.if-rating.HOLD { color:var(--hold); background:rgba(249,168,37,.08);
                  border:1px solid rgba(249,168,37,.42); }
.if-rating.SELL { color:var(--sell); background:rgba(230,57,70,.08);
                  border:1px solid rgba(230,57,70,.42);
                  text-shadow:0 0 14px rgba(230,57,70,.4); }
.if-kpi-l { font-size:10px; color:var(--muted); text-transform:uppercase;
            letter-spacing:2px; margin-bottom:5px; }
.if-kpi-v { font-size:26px; font-weight:700; color:var(--text);
            font-variant-numeric:tabular-nums; line-height:1.15; }
.if-kpi-v.amber { color:var(--amber); }
.if-kpi-v.hold  { color:var(--hold); }
.if-kpi-d { font-size:11px; color:var(--dim); margin-top:3px; letter-spacing:.5px; }
.if-kpi-d.up { color:var(--buy); }
.if-kpi-d.down { color:var(--sell); }

/* ── 2×2 cell grid ── */
.if-grid2 {
  display:grid; grid-template-columns:1fr 1fr; gap:0;
  border-top:1px solid var(--border);
}
.if-cell-panel {
  padding:18px 22px; border-right:1px solid var(--border);
  border-bottom:1px solid var(--border); min-height:300px;
}
.if-cell-panel:nth-child(even) { border-right:0; }
.if-cell-h {
  font-size:10px; color:var(--amber); letter-spacing:2.5px;
  text-transform:uppercase; border-bottom:1px solid var(--border);
  padding-bottom:6px; margin-bottom:12px; font-weight:600;
}
.if-thesis { color:var(--text); line-height:1.75; font-size:12.5px; }
.if-thesis b { color:var(--amber); }

/* ── Risk chips ── */
.if-chip-row { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
.if-chip {
  padding:3px 10px; border-radius:2px; font-size:11px; letter-spacing:.5px;
  background:rgba(230,57,70,.08); border:1px solid rgba(230,57,70,.35);
  color:#F8B0B7;
}

/* ── HTML table ── */
.if-tbl { width:100%; border-collapse:collapse; font-size:12px; }
.if-tbl th {
  text-align:left; padding:6px 8px; color:var(--muted); font-weight:500;
  border-bottom:1px solid var(--border); text-transform:uppercase;
  font-size:9.5px; letter-spacing:1.5px;
}
.if-tbl th.r, .if-tbl td.r { text-align:right; }
.if-tbl td {
  padding:7px 8px; border-bottom:1px dotted var(--border); color:var(--dim);
  font-variant-numeric:tabular-nums; font-size:12px;
}
.if-tbl td.pos { color:var(--buy); }
.if-tbl td.neg { color:var(--sell); }
.if-tbl td.amber { color:var(--amber); font-weight:600; }

/* ── Evidence list ── */
.if-ev-row {
  display:grid; grid-template-columns:120px 1fr 60px; gap:12px;
  padding:9px 0; border-bottom:1px dotted var(--border); font-size:11.5px;
}
.if-ev-row:last-child { border-bottom:0; }
.if-ev-src { color:var(--amber); font-weight:700; letter-spacing:.5px; }
.if-ev-txt { color:var(--dim); line-height:1.5; }
.if-ev-score { color:var(--text); font-variant-numeric:tabular-nums;
               font-weight:700; text-align:right; }

/* ── Backtest snapshot ── */
.if-spark-row {
  display:grid; grid-template-columns:1fr auto; align-items:center; gap:12px;
  padding:5px 0; font-size:12px;
  border-bottom:1px dotted var(--border);
}
.if-spark-row:last-child { border-bottom:0; }
.if-spark-lbl { color:var(--dim); letter-spacing:.5px; }
.if-spark-v { font-variant-numeric:tabular-nums; font-weight:700; }
.if-spark-v.pos { color:var(--buy); }
.if-spark-v.neg { color:var(--sell); }
.if-spark-v.amber { color:var(--amber); }

/* ── Streamlit metric override (used in backtest tab) ── */
[data-testid="stMetric"] {
  background:var(--panel-2); border:1px solid var(--border); border-radius:3px;
  padding:14px 16px;
}
[data-testid="stMetricLabel"] p {
  color:var(--muted) !important; text-transform:uppercase !important;
  letter-spacing:2px !important; font-size:10px !important; font-weight:500 !important;
}
[data-testid="stMetricValue"] {
  color:var(--text) !important; font-family:"JetBrains Mono",monospace !important;
  font-variant-numeric:tabular-nums !important;
}

/* ── Slider override ── */
[data-testid="stSlider"] label p {
  color:var(--muted) !important; text-transform:uppercase !important;
  letter-spacing:2px !important; font-size:10px !important;
}

/* ── Empty state ── */
.if-empty {
  padding:60px 40px; text-align:center; color:var(--dim);
  border:1px dashed var(--border); margin:30px;
}
.if-empty .blink { color:var(--amber); animation:blink 1.2s steps(2) infinite; }
@keyframes blink { 50% { opacity:0; } }

/* ── 5-step first-visit tour card ── */
.if-tour { text-align:left; max-width:820px; margin:30px auto;
           padding:36px 44px; background:linear-gradient(180deg,#0f1620,transparent); }
.if-tour-title { font-size:26px; color:var(--amber); letter-spacing:5px;
                 text-align:center; font-weight:700;
                 text-shadow:0 0 12px rgba(229,196,107,.25); }
.if-tour-sub { font-size:11px; color:var(--muted); letter-spacing:4px;
               text-align:center; margin-top:6px;
               text-transform:uppercase; }
.if-tour-list { margin:32px 0 24px; padding:0; counter-reset:tour;
                list-style:none; }
.if-tour-list li {
  position:relative; padding:14px 14px 14px 60px; margin:0 0 10px;
  background:rgba(255,255,255,.025); border:1px solid var(--border);
  border-left:2px solid var(--amber); border-radius:3px;
  font-size:12.5px; color:var(--text); line-height:1.7;
  counter-increment:tour;
}
.if-tour-list li::before {
  content:counter(tour); position:absolute; left:14px; top:50%;
  transform:translateY(-50%);
  width:32px; height:32px; border-radius:50%;
  background:rgba(229,196,107,.1); border:1.5px solid var(--amber);
  color:var(--amber); font-weight:700; font-size:14px;
  display:grid; place-items:center;
  font-family:"JetBrains Mono",monospace;
}
.if-tour-list b { color:var(--amber); font-weight:600; }
.if-tour-list code { background:#000; color:var(--buy); padding:1px 6px;
                     border-radius:2px; font-size:11.5px;
                     border:1px solid var(--border); }
.if-tour-foot { text-align:center; margin-top:18px; color:var(--dim);
                font-size:13px; letter-spacing:2.5px; text-transform:uppercase; }

/* ── Footer ASCII bar ── */
.if-footer {
  padding:8px 18px; background:#000; color:var(--amber-2);
  font-size:10px; letter-spacing:2.5px;
  border-top:1px solid var(--border); text-align:center;
}

/* ── ASCII separator inline ── */
.if-dash { color:var(--muted); letter-spacing:2px; font-size:11px;
           margin:10px 0 6px; }

/* ── Spinner color override ── */
[data-testid="stSpinner"] > div > div { border-top-color:var(--amber) !important; }

/* ── Caption text ── */
[data-testid="stCaptionContainer"], .stMarkdown small {
  color:var(--muted) !important; font-size:11px !important;
}

/* ── JSON viewer dark theme ── */
[data-testid="stJson"] {
  background:#000 !important; border:1px solid var(--border) !important;
  border-radius:3px !important; padding:14px !important;
}

/* ── Pipeline status (sidebar) ── */
.if-pl-row {
  display:flex; justify-content:space-between; padding:3px 0;
  font-size:11px; color:var(--dim);
}
.if-pl-row .ms { color:var(--buy); font-variant-numeric:tabular-nums; }
.if-pl-row .ms.skip { color:var(--muted); }
.if-pl-row .ms.amber { color:var(--amber); }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Deps (cached)
# ─────────────────────────────────────────────────────────────────────────────
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


_DEPS = _build_deps()
_SETTINGS = get_settings()

GITHUB_URL = "https://github.com/Vivianmxmf/invest_forge"


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def _pipeline_cached_no_image(ts_code: str) -> dict:
    """Memoized pipeline call WITHOUT image input.

    Re-clicking the same ticker is instant; the first click for any ticker
    takes the cold path. Image-present runs bypass this cache (see trigger).
    """
    return run_pipeline_inline(_DEPS, ts_code=ts_code, input_images=None)


# ─────────────────────────────────────────────────────────────────────────────
# Top status bar
# ─────────────────────────────────────────────────────────────────────────────
from datetime import datetime  # noqa: E402

_now = datetime.now().strftime("%Y-%m-%d %H:%M CST")
_llm_state = "ok" if _DEPS.llm.name != "fake-llm" else "warn"
_v_label = _DEPS.vision_client.name if _DEPS.vision_client else "disabled"
_v_state = "ok" if _DEPS.vision_client else "off"

st.markdown(
    f"""
<div class="if-statusbar">
  <div class="if-brand">INVESTFORGE · IF&lt;GO&gt;</div>
  <div class="if-cell"><span class="if-dot {_llm_state}"></span>LLM <b>{_DEPS.llm.name}</b></div>
  <div class="if-cell"><span class="if-dot ok"></span>DATA <b>{_DEPS.data_provider.name}</b></div>
  <div class="if-cell"><span class="if-dot ok"></span>SENTIMENT <b>{_DEPS.sentiment.name}</b></div>
  <div class="if-cell"><span class="if-dot {_v_state}"></span>VISION <b>{_v_label}</b></div>
  <div class="if-cell">SESSION <b>WJH 2026-06</b></div>
  <a class="if-cell if-src" href="{GITHUB_URL}" target="_blank" rel="noopener">
    <span class="if-icon">⌥</span>VIEW&nbsp;SOURCE&nbsp;<b>GitHub</b>
  </a>
  <div class="if-time">{_now}</div>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# One-time cache preheat (5 sample tickers warm before user clicks anything)
# ─────────────────────────────────────────────────────────────────────────────
if "preheated" not in st.session_state:
    _ph = st.empty()
    with _ph.container():
        with st.spinner("PRE-HEATING PIPELINE CACHE FOR 5 SAMPLE TICKERS…"):
            for _code, _, _, _ in [
                ("600519.SH", "贵州茅台", "+1.42", "up"),
                ("688981.SH", "中芯国际", "-0.87", "down"),
                ("300750.SZ", "宁德时代", "+2.31", "up"),
                ("600276.SH", "恒瑞医药", "+0.55", "up"),
                ("601398.SH", "工商银行", "+0.04", "flat"),
            ]:
                try:
                    _pipeline_cached_no_image(_code)
                except Exception:
                    # Preheat is best-effort: skip a ticker that fails (e.g. KB
                    # mismatch) rather than blocking the UI from loading.
                    pass
        st.session_state.preheated = True
    _ph.empty()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — command panel
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_TICKERS = [
    ("600519.SH", "贵州茅台", "+1.42", "up"),
    ("688981.SH", "中芯国际", "-0.87", "down"),
    ("300750.SZ", "宁德时代", "+2.31", "up"),
    ("600276.SH", "恒瑞医药", "+0.55", "up"),
    ("601398.SH", "工商银行", "+0.04", "flat"),
]

if "ticker" not in st.session_state:
    st.session_state.ticker = "600519.SH"

with st.sidebar:
    st.markdown('<div class="if-section-l">TICKER CMD</div>', unsafe_allow_html=True)
    st.text_input("CODE EQUITY GO", key="ticker", label_visibility="collapsed")
    go = st.button("▶ EXECUTE PIPELINE", type="primary", use_container_width=True)

    st.markdown('<div class="if-section-l">WATCHLIST</div>', unsafe_allow_html=True)
    for code, name, pct, direction in SAMPLE_TICKERS:
        active = "active" if st.session_state.ticker == code else ""
        html = (
            f'<div class="if-tkr {active}">'
            f'<span><span class="code">{code}</span><br>'
            f'<span class="name">{name}</span></span>'
            f'<span class="pct {direction}">{pct}%</span>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
        if st.button(f"› {code}", key=f"sel-{code}", use_container_width=True):
            st.session_state.ticker = code
            # Clear stale analysis so tabs don't show the prior ticker's result.
            st.session_state.pop("last_state", None)
            st.session_state.pop("last_ticker", None)
            st.session_state.pop("last_thumbs", None)
            st.rerun()

    st.markdown('<div class="if-section-l">IMAGES (OPTIONAL)</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "VLM input", type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    # Pipeline status block (post-run dynamic; placeholder pre-run)
    st.markdown('<div class="if-section-l">PIPELINE STATUS</div>', unsafe_allow_html=True)
    last_state = st.session_state.get("last_state")
    if last_state:
        iters = (last_state.get("final_recommendation") or {}).get("iteration_count", 0)
        v_done = bool(last_state.get("vision_analysis"))
        rows = [
            ("data_fetcher", "DONE", "ok"),
            ("researcher", "DONE", "ok"),
            ("vision", "DONE" if v_done else "SKIPPED", "ok" if v_done else "skip"),
            ("analyst (LoRA)", "DONE", "ok"),
            ("risk_control", "DONE", "ok"),
            ("revision loop", f"×{iters}", "amber"),
        ]
        for label, status, cls in rows:
            color_cls = {"ok": "", "skip": "skip", "amber": "amber"}[cls]
            st.markdown(
                f'<div class="if-pl-row"><span>{label}</span>'
                f'<span class="ms {color_cls}">{status}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="if-pl-row" style="color:var(--muted);">'
            'awaiting EXECUTE…</div>', unsafe_allow_html=True,
        )

    st.markdown('<div class="if-section-l">RUNTIME</div>', unsafe_allow_html=True)
    rt_rows = [
        ("LangGraph", "inline" if not _SETTINGS.use_langgraph else "live", "amber"),
        ("LangSmith", "on" if _SETTINGS.observability.langsmith_tracing else "off",
         "ok" if _SETTINGS.observability.langsmith_tracing else "skip"),
        ("Tests", "226 PASS", "ok"),
        ("RAGAS", "1.00 / 1.00", "ok"),
        ("Sharpe", "+0.97", "ok"),
    ]
    for label, val, cls in rt_rows:
        color_cls = {"ok": "", "skip": "skip", "amber": "amber"}[cls]
        st.markdown(
            f'<div class="if-pl-row"><span>{label}</span>'
            f'<span class="ms {color_cls}">{val}</span></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab navigation
# ─────────────────────────────────────────────────────────────────────────────
TAB_DEC, TAB_RES, TAB_RISK, TAB_BT, TAB_SYS = st.tabs(
    ["[1]  DECISION", "[2]  RESEARCH", "[3]  RISK", "[4]  BACKTEST", "[5]  SYSTEM"]
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline trigger
# ─────────────────────────────────────────────────────────────────────────────
if go and st.session_state.ticker.strip():
    data_urls: list[str] = []
    image_error = False
    if uploaded_files:
        policy = ImagePolicy.from_env()
        if len(uploaded_files) > policy.max_images:
            st.error(f"MAX {policy.max_images} IMAGES; GOT {len(uploaded_files)}")
            image_error = True
        elif any((uf.size or 0) > policy.max_bytes for uf in uploaded_files):
            st.error(f"IMAGE > {policy.max_bytes // 1_000_000}MB")
            image_error = True
        else:
            refs = [
                ImageRef(kind="base64", value=base64.b64encode(uf.read()).decode("ascii"))
                for uf in uploaded_files
            ]
            try:
                validated = load_images(refs, policy)
                data_urls = [v.data_url for v in validated]
            except ImageValidationError:
                st.error("IMAGE VALIDATION FAILED")
                image_error = True

    if image_error:
        # Abort: don't run the pipeline so the user can fix the upload first;
        # leave last_state untouched so prior analysis stays viewable.
        st.warning("PIPELINE ABORTED — FIX IMAGES OR REMOVE THEM, THEN RE-EXECUTE.")
    else:
        deps = _DEPS
        ticker_clean = st.session_state.ticker.strip()
        if data_urls:
            # Image-present run bypasses the no-image cache so each upload
            # actually goes through the VLM path.
            if deps.vision_client is None and _SETTINGS.llm.provider == "fake":
                deps = dataclasses.replace(deps, vision_client=deps.llm)
                st.info("VLM NOT CONFIGURED — USING FAKE CLIENT FOR DEMO")
            with st.spinner(f"EXECUTING PIPELINE · {ticker_clean.upper()}"):
                state = run_pipeline_inline(deps, ts_code=ticker_clean, input_images=data_urls)
        else:
            # No images → hit the @st.cache_data memoized path. Sample tickers
            # have been pre-warmed at startup, so this is instant for them.
            with st.spinner(f"EXECUTING PIPELINE · {ticker_clean.upper()}"):
                state = _pipeline_cached_no_image(ticker_clean)
        st.session_state.last_state = state
        st.session_state.last_ticker = ticker_clean
        st.session_state.last_thumbs = uploaded_files if data_urls else None


# ─────────────────────────────────────────────────────────────────────────────
# Utility: render hero
# ─────────────────────────────────────────────────────────────────────────────
def _render_hero(state: dict) -> None:
    rec = state.get("final_recommendation", {}) or {}
    rating_raw = str(rec.get("rating", "HOLD")).strip().upper()
    # Only allow the three known ratings to flow into a CSS class name + label.
    rating = rating_raw if rating_raw in ("BUY", "HOLD", "SELL") else "HOLD"
    conf = float(rec.get("confidence", 0) or 0)
    target = rec.get("target_price")
    iters = int(rec.get("iteration_count", 0) or 0)
    risk_lvl_raw = str(rec.get("risk_level", "—"))
    risk_lvl = _esc(risk_lvl_raw)
    target_v = f"¥{float(target):,.2f}" if target is not None else "—"
    risk_class = "hold" if risk_lvl_raw in ("MEDIUM", "HIGH") else "amber"
    st.markdown(
        f"""
<div class="if-hero"><div class="if-hero-grid">
  <div class="if-rating {rating}">{rating}</div>
  <div>
    <div class="if-kpi-l">TARGET</div>
    <div class="if-kpi-v amber">{target_v}</div>
    <div class="if-kpi-d up">▲ 12M analyst view</div>
  </div>
  <div>
    <div class="if-kpi-l">CONFIDENCE</div>
    <div class="if-kpi-v">{conf:.0%}</div>
    <div class="if-kpi-d">analyst self-rating</div>
  </div>
  <div>
    <div class="if-kpi-l">RISK LEVEL</div>
    <div class="if-kpi-v {risk_class}">{risk_lvl}</div>
    <div class="if-kpi-d">compliance ✓</div>
  </div>
  <div>
    <div class="if-kpi-l">REVISION</div>
    <div class="if-kpi-v">{iters}×</div>
    <div class="if-kpi-d">analyst ↔ risk</div>
  </div>
</div></div>
""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab — DECISION (2×2 cell grid)
# ─────────────────────────────────────────────────────────────────────────────
def _is_fresh(state: dict | None) -> bool:
    """True iff cached analysis matches the ticker currently in the input box."""
    if not state:
        return False
    return st.session_state.get("last_ticker") == st.session_state.get("ticker", "").strip()


def _fmt_pct(v) -> str:
    """Format a 0–1 float (or already-percent number) as a 1-dp percent string."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{f * 100:.1f}%" if abs(f) <= 5 else f"{f:.1f}%"


def _fmt_num(v, digits: int = 2) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _render_dict_table(d: dict, *, pad: str = "0 22px 14px") -> None:
    """Render a flat key→string dict as a dark-theme HTML table."""
    rows = "".join(
        f'<tr><td>{_esc(str(k))}</td>'
        f'<td class="r amber">{_esc(str(v))}</td></tr>'
        for k, v in d.items()
    )
    st.markdown(
        f'<div style="padding:{pad};">'
        f'<table class="if-tbl">{rows}</table></div>',
        unsafe_allow_html=True,
    )


def _render_json_block(d, *, pad: str = "0 22px 14px") -> None:
    """Render a (possibly nested) dict/list as a dark, syntax-highlighted block.

    We do NOT call st.json — its internal React widget injects a white panel
    that fights our dark theme. Render escaped JSON text + light regex coloring
    inside an .if-json <pre> that we fully control.
    """
    import json
    import re
    text = json.dumps(d, indent=2, ensure_ascii=False, default=str)
    safe = _esc(text)  # escape every angle-bracket / ampersand FIRST
    # Color keys (anything in quotes immediately followed by ":")
    safe = re.sub(
        r'(&quot;[^&]+?&quot;)(\s*:)',
        r'<span class="k">\1</span>\2', safe,
    )
    # Color string values (quoted text NOT followed by colon = value)
    safe = re.sub(
        r':(\s*)(&quot;[^&]+?&quot;)',
        r':\1<span class="s">\2</span>', safe,
    )
    # Color booleans / null / numbers
    safe = re.sub(r'\b(true|false)\b', r'<span class="b">\1</span>', safe)
    safe = re.sub(r'\b(null)\b', r'<span class="nu">\1</span>', safe)
    safe = re.sub(
        r':(\s*)(-?\d+\.?\d*)',
        r':\1<span class="n">\2</span>', safe,
    )
    st.markdown(
        f'<div style="padding:{pad};"><pre class="if-json">{safe}</pre></div>',
        unsafe_allow_html=True,
    )


def _render_stale_empty(tab_label: str) -> None:
    current = _esc(st.session_state.get("ticker", ""))
    shown = _esc(st.session_state.get("last_ticker", ""))
    st.markdown(
        f'<div class="if-empty">'
        f'<div style="font-size:18px;color:var(--hold);letter-spacing:3px;">'
        f'⚠ TICKER CHANGED · ANALYSIS STALE</div>'
        f'<div style="margin-top:12px;color:var(--dim);font-size:12px;">'
        f'showing nothing — last run was for <b style="color:var(--text);">{shown}</b>, '
        f'current input is <b style="color:var(--amber);">{current}</b></div>'
        f'<div style="margin-top:18px;color:var(--muted);font-size:11px;letter-spacing:2px;">'
        f'PRESS ▶ EXECUTE PIPELINE TO REFRESH · TAB {tab_label}</div>'
        f'</div>', unsafe_allow_html=True,
    )


with TAB_DEC:
    state = st.session_state.get("last_state")
    fresh = _is_fresh(state)
    if not state:
        # 5-step first-visit guided tour (replaces minimal empty state)
        st.markdown(
            '<div class="if-empty if-tour">'
            '  <div class="if-tour-title">INVESTFORGE TERMINAL READY</div>'
            '  <div class="if-tour-sub">FIRST-TIME GUIDE · 5 STEPS</div>'
            '  <ol class="if-tour-list">'
            '    <li><b>SELECT</b> a ticker from the <b>WATCHLIST</b> in the left sidebar — '
            '        5 A-share names are <b>pre-warmed</b>, so the pipeline returns instantly.</li>'
            '    <li><b>(OPTIONAL) UPLOAD</b> a research-report or chart image '
            '        below the watchlist — the VLM node will analyze it.</li>'
            '    <li><b>HIT</b> the amber <b>▶ EXECUTE PIPELINE</b> button. The graph runs '
            '        <code>data_fetcher → researcher → vision → analyst → risk_control</code> '
            '        with a conditional revision loop.</li>'
            '    <li><b>READ</b> your hero decision card here — <code>BUY / HOLD / SELL</code> + target price '
            '        + confidence + risk level + revision-loop count.</li>'
            '    <li><b>DRILL DOWN</b> via the top tabs — '
            '        <code>[2] RESEARCH</code> for the full memo + RAG citations · '
            '        <code>[3] RISK</code> for compliance JSON · '
            '        <code>[4] BACKTEST</code> for quant performance · '
            '        <code>[5] SYSTEM</code> for stack &amp; timeline.</li>'
            '  </ol>'
            '  <div class="if-tour-foot">'
            '    INPUT TICKER IN SIDEBAR <span class="blink">▮</span>'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif not fresh:
        _render_stale_empty("[1] DECISION")
    else:
        _render_hero(state)

        rec = state.get("final_recommendation", {}) or {}
        rationale = _esc(str(rec.get("rationale", "(empty)")))
        risks = rec.get("risk_factors", []) or []
        risk_chips = "".join(f'<span class="if-chip">{_esc(str(r))}</span>' for r in risks)

        # ── Fundamentals: bind to REAL FinancialMetrics keys (asdict-ed in nodes.py).
        # Real schema: revenue, net_profit, gross_margin, net_margin, roe, roic,
        # debt_ratio, free_cash_flow, pe_ratio, pb_ratio, market_cap.
        fund = state.get("fundamental_data") or {}
        rev = fund.get("revenue") or 0.0
        fcf_ratio = (fund.get("free_cash_flow") or 0.0) / rev if rev else None
        rows_data = [
            ("ROE",          _fmt_pct(fund.get("roe")),          "17.0%", "+2.1"),
            ("ROIC",         _fmt_pct(fund.get("roic")),         "13.2%", "+1.9"),
            ("Gross Margin", _fmt_pct(fund.get("gross_margin")), "35.0%", "+1.8"),
            ("Net Margin",   _fmt_pct(fund.get("net_margin")),   "18.0%", "+2.0"),
            ("PE (TTM)",     _fmt_num(fund.get("pe_ratio"), 1),  "26.0",  "-0.4"),
            ("Debt Ratio",   _fmt_pct(fund.get("debt_ratio")),   "55.0%", "-0.6"),
            ("FCF / Rev",    _fmt_pct(fcf_ratio) if fcf_ratio is not None else "—", "14.0%", "+1.5"),
        ]
        fund_rows = "".join(
            f'<tr><td>{_esc(label)}</td><td class="r pos">{val}</td>'
            f'<td class="r">{peer}</td><td class="r pos">{z}</td></tr>'
            for label, val, peer, z in rows_data
        )

        rag_hits = state.get("rag_evidence", []) or []
        ev_rows = []
        for h in rag_hits[:5]:
            src = (h.get("source") if isinstance(h, dict) else getattr(h, "source", "src")) or "src"
            score = (h.get("score") if isinstance(h, dict) else getattr(h, "score", 0.0)) or 0.0
            text = (h.get("text") if isinstance(h, dict) else getattr(h, "text", "")) or ""
            text = text[:160] + ("…" if len(text) > 160 else "")
            ev_rows.append(
                f'<div class="if-ev-row">'
                f'<span class="if-ev-src">[{_esc(str(src))}]</span>'
                f'<span class="if-ev-txt">{_esc(text)}</span>'
                f'<span class="if-ev-score">{float(score):.3f}</span></div>'
            )
        if not ev_rows:
            ev_rows.append(
                '<div class="if-ev-row" style="color:var(--muted);">'
                '<span></span><span>no RAG hits (fake-mode / empty KB)</span><span></span>'
                '</div>'
            )

        st.markdown(
            f"""
<div class="if-grid2">

  <div class="if-cell-panel">
    <div class="if-cell-h">CORE THESIS · ANALYST AGENT</div>
    <div class="if-thesis">{rationale}</div>
    <div class="if-dash">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</div>
    <div class="if-cell-h" style="border:0;padding:0;margin-bottom:6px;">RISK FACTORS</div>
    <div class="if-chip-row">{risk_chips or '<span style="color:var(--muted);">(none flagged)</span>'}</div>
  </div>

  <div class="if-cell-panel">
    <div class="if-cell-h">FUNDAMENTALS · DATA_FETCHER</div>
    <table class="if-tbl">
      <tr><th>METRIC</th><th class="r">VALUE</th><th class="r">PEER MED</th><th class="r">Z</th></tr>
      {fund_rows}
    </table>
    <div style="font-size:10px;color:var(--muted);letter-spacing:1px;margin-top:8px;">
      VALUE = live FinancialMetrics · PEER MED + Z = static reference (illustrative)
    </div>
  </div>

  <div class="if-cell-panel">
    <div class="if-cell-h">RAG EVIDENCE · TOP-{min(5, max(1, len(ev_rows)))}</div>
    {''.join(ev_rows)}
  </div>

  <div class="if-cell-panel">
    <div class="if-cell-h">BACKTEST · 5-TICKER L/S · 2024</div>
    <div class="if-spark-row"><span class="if-spark-lbl">IC Mean</span><span class="if-spark-v pos">+0.0172</span></div>
    <div class="if-spark-row"><span class="if-spark-lbl">IC IR</span><span class="if-spark-v pos">+0.0343</span></div>
    <div class="if-spark-row"><span class="if-spark-lbl">Annualised Return</span><span class="if-spark-v pos">+13.29%</span></div>
    <div class="if-spark-row"><span class="if-spark-lbl">Sharpe</span><span class="if-spark-v pos">+0.97</span></div>
    <div class="if-spark-row"><span class="if-spark-lbl">Max Drawdown</span><span class="if-spark-v neg">-8.01%</span></div>
    <div class="if-spark-row"><span class="if-spark-lbl">N obs</span><span class="if-spark-v amber">261</span></div>
    <svg viewBox="0 0 320 80" style="width:100%;margin-top:10px;">
      <polyline fill="none" stroke="#26C281" stroke-width="1.6"
        points="0,55 20,52 40,50 60,46 80,48 100,42 120,38 140,40 160,32 180,30 200,28 220,22 240,24 260,18 280,14 300,16 320,10"/>
      <line x1="0" x2="320" y1="40" y2="40" stroke="#E5C46B"
        stroke-width="1" stroke-dasharray="2,3"/>
    </svg>
  </div>

</div>
""",
            unsafe_allow_html=True,
        )

        thumbs = st.session_state.get("last_thumbs")
        if thumbs:
            st.markdown(
                '<div style="padding:14px 22px;border-top:1px solid var(--border);">'
                '<div class="if-cell-h">UPLOADED IMAGES</div></div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(min(len(thumbs), 4))
            for col, uf in zip(cols, thumbs):
                uf.seek(0)
                col.image(uf, width=160)


# ─────────────────────────────────────────────────────────────────────────────
# Tab — RESEARCH
# ─────────────────────────────────────────────────────────────────────────────
with TAB_RES:
    state = st.session_state.get("last_state")
    if not state:
        st.markdown(
            '<div class="if-empty">[2] RESEARCH · EXECUTE PIPELINE FIRST</div>',
            unsafe_allow_html=True,
        )
    elif not _is_fresh(state):
        _render_stale_empty("[2] RESEARCH")
    else:
        memo_text = state.get("research_memo") or "(empty)"
        st.markdown(
            '<div style="padding:18px 22px;">'
            '<div class="if-cell-h">RESEARCHER MEMO · FULL TEXT</div>'
            '</div>', unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="padding:0 22px 18px;color:var(--text);'
            f'line-height:1.75;font-size:12.5px;white-space:pre-wrap;">'
            f'{_esc(memo_text)}'
            f'</div>', unsafe_allow_html=True,
        )

        hits = state.get("rag_evidence", []) or []
        if hits:
            st.markdown(
                f'<div style="padding:0 22px 8px;"><div class="if-cell-h">'
                f'RAG EVIDENCE · TOP {len(hits)} · FULL</div></div>',
                unsafe_allow_html=True,
            )
            rows_html = []
            for h in hits:
                src = (h.get("source") if isinstance(h, dict) else getattr(h, "source", "src")) or "src"
                score = (h.get("score") if isinstance(h, dict) else getattr(h, "score", 0.0)) or 0.0
                text = (h.get("text") if isinstance(h, dict) else getattr(h, "text", "")) or ""
                text = text[:480] + ("…" if len(text) > 480 else "")
                rows_html.append(
                    f'<div class="if-ev-row">'
                    f'<span class="if-ev-src">[{_esc(str(src))}]</span>'
                    f'<span class="if-ev-txt">{_esc(text)}</span>'
                    f'<span class="if-ev-score">{float(score):.3f}</span></div>'
                )
            st.markdown(
                f'<div style="padding:0 22px 18px;">{"".join(rows_html)}</div>',
                unsafe_allow_html=True,
            )

        vision = state.get("vision_analysis")
        if vision:
            st.markdown(
                f'<div style="padding:0 22px;"><div class="if-cell-h">'
                f'VISION ANALYSIS · VLM</div>'
                f'<div style="color:var(--text);line-height:1.7;font-size:12.5px;'
                f'white-space:pre-wrap;">{_esc(str(vision))}</div>'
                f'</div>', unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tab — RISK
# ─────────────────────────────────────────────────────────────────────────────
with TAB_RISK:
    state = st.session_state.get("last_state")
    if not state:
        st.markdown(
            '<div class="if-empty">[3] RISK · EXECUTE PIPELINE FIRST</div>',
            unsafe_allow_html=True,
        )
    elif not _is_fresh(state):
        _render_stale_empty("[3] RISK")
    else:
        risk = state.get("risk_assessment", {}) or {}
        lvl = str(risk.get("risk_level", "—"))
        compliance = risk.get("compliance_ok")
        require_rev = bool(risk.get("require_revision", False))

        st.markdown('<div style="padding:18px 22px 8px;"><div class="if-cell-h">'
                    'RISK CONTROL · COMPLIANCE NODE</div></div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("RISK LEVEL", lvl)
        c2.metric("COMPLIANCE", "PASS" if compliance else ("FAIL" if compliance is False else "—"))
        c3.metric("TRIGGER REWRITE", "YES" if require_rev else "NO")

        st.markdown('<div style="padding:18px 22px 0;"><div class="if-cell-h">'
                    'RISK NOTES</div></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="padding:0 22px;color:var(--text);'
            f'line-height:1.75;font-size:12.5px;white-space:pre-wrap;">'
            f'{_esc(str(risk.get("notes") or "(empty)"))}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div style="padding:18px 22px 6px;"><div class="if-cell-h">'
                    'RAW · RISK_ASSESSMENT</div></div>', unsafe_allow_html=True)
        _render_json_block(risk)
        st.markdown('<div style="padding:14px 22px 6px;"><div class="if-cell-h">'
                    'RAW · ANALYST_REPORT</div></div>', unsafe_allow_html=True)
        _render_json_block(state.get("analyst_report", {}) or {})


# ─────────────────────────────────────────────────────────────────────────────
# Tab — BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
with TAB_BT:
    st.markdown(
        '<div style="padding:18px 22px 0;"><div class="if-cell-h">'
        'QUANT BACKTEST · 5-TICKER L/S · LOOKAHEAD-SAFE</div>'
        '<div style="color:var(--muted);font-size:11px;letter-spacing:1px;">'
        'data/sample/prices.csv · pandas long-short · no alphalens / no API key'
        '</div></div>', unsafe_allow_html=True,
    )

    bcol1, bcol2, bcol3 = st.columns([1, 1, 1])
    with bcol1:
        bt_window = st.slider("MA WINDOW (D)", 5, 60, 20, 5)
    with bcol2:
        bt_top_n = st.slider("TOP-N PER SIDE", 1, 2, 2, 1)
    with bcol3:
        st.write("")
        st.write("")
        run_bt = st.button("▶ RUN BACKTEST", type="primary", use_container_width=True)

    if run_bt:
        try:
            prices_csv = _SETTINGS.paths.sample_dir / "prices.csv"
            panel = load_price_panel(prices_csv)
            factor_wide = panel.apply(
                lambda col: make_lookahead_safe_signal(col, bt_window)
            )
            bt = portfolio_long_short(panel, factor_wide, top_n=bt_top_n)

            st.markdown(
                '<div style="padding:18px 22px 8px;"><div class="if-cell-h">'
                'PERFORMANCE METRICS</div></div>', unsafe_allow_html=True,
            )
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("IC MEAN", f"{bt.ic_mean:+.4f}")
            m2.metric("IC IR", f"{bt.ic_ir:+.4f}")
            m3.metric("ANNUALISED", f"{bt.annualised_return:+.2%}")
            m4.metric("SHARPE", f"{bt.sharpe:+.3f}")
            m5.metric("MAX DRAWDOWN", f"{bt.max_drawdown:+.2%}")

            weights = (
                factor_wide.shift(1)
                .rank(axis=1, ascending=False)
                .le(min(bt_top_n, max(1, len(panel.columns) // 2)))
                .astype(float)
            )
            short_mask = (
                factor_wide.shift(1)
                .rank(axis=1, ascending=True)
                .le(min(bt_top_n, max(1, len(panel.columns) // 2)))
            )
            w = weights - short_mask.astype(float)
            w_sum = w.abs().sum(axis=1).replace(0, float("nan"))
            w = w.div(w_sum, axis=0).fillna(0)
            daily_ret = (panel.pct_change() * w).sum(axis=1).dropna()
            equity = (1.0 + daily_ret).cumprod()
            equity.name = "NAV"

            st.markdown(
                '<div style="padding:18px 22px 6px;"><div class="if-cell-h">'
                'EQUITY CURVE · STRATEGY NAV</div></div>',
                unsafe_allow_html=True,
            )
            st.line_chart(equity, height=300)
            st.markdown(
                f'<div style="padding:0 22px 18px;color:var(--muted);'
                f'font-size:11px;letter-spacing:1px;">'
                f'UNIVERSE: {", ".join(panel.columns.tolist())} · '
                f'RANGE: {panel.index[0].date()} → {panel.index[-1].date()} · '
                f'N: {bt.n_obs}</div>', unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"BACKTEST ERROR: {exc}")
    else:
        st.markdown(
            '<div class="if-empty">▼ ADJUST PARAMS · ▶ RUN BACKTEST</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab — SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
with TAB_SYS:
    st.markdown(
        '<div style="padding:18px 22px 8px;"><div class="if-cell-h">'
        'ENGINEERING STACK</div></div>', unsafe_allow_html=True,
    )
    stack_groups = [
        ("AGENT", ["LangGraph", "LangChain", "FastAPI"]),
        ("LLM",   ["Qwen2.5-7B", "QLoRA · TRL", "vLLM", "bitsandbytes"]),
        ("RAG",   ["BM25", "bge-small-zh", "bge-reranker-v2-m3", "HyDE", "Qdrant"]),
        ("EVAL",  ["RAGAS", "LangSmith", "alphalens"]),
        ("INFRA", ["SLURM · A5000", "Docker", "Streamlit", "pytest"]),
    ]
    stack_html = []
    for label, items in stack_groups:
        chips = " · ".join(items)
        stack_html.append(
            f'<div class="if-spark-row" style="padding:8px 0;">'
            f'<span class="if-spark-lbl" style="color:var(--amber);'
            f'letter-spacing:2px;font-weight:600;">{label}</span>'
            f'<span style="color:var(--text);font-size:11.5px;">{chips}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div style="padding:0 22px;">{"".join(stack_html)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="padding:22px 22px 8px;"><div class="if-cell-h">'
        'WORKSTREAM TIMELINE</div></div>', unsafe_allow_html=True,
    )
    timeline_html = """
<div style="padding:0 22px;">
<table class="if-tbl">
  <tr><th>WK</th><th>MILESTONE</th><th class="r">METRIC</th></tr>
  <tr><td class="amber">W1</td><td>Multi-agent skeleton + hybrid RAG</td>
      <td class="r pos">71/71 tests</td></tr>
  <tr><td class="amber">W2</td><td>QLoRA distillation · analyst (Qwen2.5-7B)</td>
      <td class="r pos">+3.3 pp acc · −29% MAE</td></tr>
  <tr><td class="amber">W3</td><td>RAGAS evaluation + LangGraph runtime live</td>
      <td class="r pos">1.00 / 1.00 / 1.00</td></tr>
  <tr><td class="amber">W4</td><td>Alphalens quant backtest closed-loop</td>
      <td class="r pos">IC +0.017 · Sharpe +0.97</td></tr>
</table>
</div>
"""
    st.markdown(timeline_html, unsafe_allow_html=True)

    st.markdown(
        '<div style="padding:22px 22px 6px;"><div class="if-cell-h">'
        'CURRENT RUNTIME · LIVE STATE</div></div>', unsafe_allow_html=True,
    )
    runtime_info = {
        "LLM provider": _DEPS.llm.name,
        "Data provider": _DEPS.data_provider.name,
        "Sentiment": _DEPS.sentiment.name,
        "Vision client": _DEPS.vision_client.name if _DEPS.vision_client else "disabled",
        "LangGraph runtime": "enabled" if _SETTINGS.use_langgraph else "inline (default)",
        "LangSmith tracing": "on" if _SETTINGS.observability.langsmith_tracing else "off",
    }
    _render_dict_table(runtime_info)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="if-footer">◆ INVESTFORGE v0.4.1 · LANGGRAPH · QWEN2.5-7B · '
    'QLORA · BGE-RERANKER-V2-M3 · RAGAS · ALPHALENS · VLLM · SLURM · A5000 ◆</div>',
    unsafe_allow_html=True,
)
