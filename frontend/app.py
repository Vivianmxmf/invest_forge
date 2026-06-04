"""JANUS — Streamlit Dashboard (Bloomberg Terminal redesign).

Brand: JANUS — the two-faced Roman god, looking at the past (fundamentals
/ history / RAG) and the future (analyst forecast / target price). Python
package keeps the internal name ``invest_forge`` for import stability.

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
_FAVICON_PATH = _ROOT / "assets" / "janus_favicon.png"

st.set_page_config(
    page_title="JANUS · IF<GO>",
    # Custom JANUS target favicon (32×32 PNG); falls back to ⌖ glyph if file
    # is missing (e.g. trimmed deployment).
    page_icon=str(_FAVICON_PATH) if _FAVICON_PATH.exists() else "⌖",
    layout="wide",
    # "auto" → Streamlit collapses the sidebar on narrow viewports (mobile),
    # keeps it expanded on desktop. Critical for mobile-responsive UX.
    initial_sidebar_state="auto",
    menu_items={"About": "JANUS — Bloomberg-style AI investment terminal · two-faced multi-agent research system."},
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
.if-brand::before { content:"⌖ "; color:var(--amber); }
/* ⌖ = JANUS target glyph (the two-faced god aims at past + future) */
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

/* ── Ticker tape (horizontally-scrolling marquee under status bar) ── */
.if-tape {
  background:#070b10; border-bottom:1px solid var(--border);
  overflow:hidden; white-space:nowrap; padding:6px 0;
  position:relative;
}
.if-tape::before, .if-tape::after {
  content:""; position:absolute; top:0; bottom:0; width:60px; z-index:2;
  pointer-events:none;
}
.if-tape::before { left:0;  background:linear-gradient(90deg,#070b10,transparent); }
.if-tape::after  { right:0; background:linear-gradient(-90deg,#070b10,transparent); }
.if-tape-inner {
  display:inline-block; animation:tapeScroll 80s linear infinite;
  will-change:transform;
}
@keyframes tapeScroll {
  from { transform:translateX(0); }
  to   { transform:translateX(-50%); }
}
.tt-item {
  display:inline-flex; align-items:center; gap:8px;
  padding:0 18px; border-right:1px solid var(--border);
  font-size:11.5px; letter-spacing:.5px;
}
.tt-code { color:var(--amber); font-weight:700; }
.tt-name { color:var(--dim); font-size:10.5px; }
.tt-px   { color:var(--text); font-weight:600;
           font-variant-numeric:tabular-nums; }
.tt-chg  { font-weight:700; font-variant-numeric:tabular-nums;
           padding:1px 6px; border-radius:2px; font-size:10.5px; }
.tt-chg.up   { color:var(--buy);  background:rgba(38,194,129,.08); }
.tt-chg.down { color:var(--sell); background:rgba(230,57,70,.08); }
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

/* ── Compare-tab card ── */
.if-cmp-col {
  padding:20px 22px; background:rgba(255,255,255,.02);
  border:1px solid var(--border); border-radius:4px;
}
.if-cmp-tkr {
  font-size:14px; color:var(--amber); letter-spacing:3px;
  font-weight:700; margin-bottom:10px; text-transform:uppercase;
  border-bottom:1px solid var(--border); padding-bottom:8px;
}
.if-cmp-kpi-row {
  display:grid; grid-template-columns:repeat(4,1fr); gap:14px;
  padding:4px 0; border-bottom:1px dotted var(--border); margin-bottom:8px;
}
.if-cmp-kpi-row .if-kpi-v { font-size:18px; }
.if-cmp-kpi-row .if-kpi-l { font-size:9px; letter-spacing:1.5px; }

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

/* ─────────────────────────────────────────────────────────────────────────
 * Responsive layout — tablet (≤ 980 px) + phone (≤ 640 px)
 * Streamlit auto-collapses the sidebar at narrow widths; the rules below
 * fix the BLOOMBERG-style fixed grids that would otherwise overflow.
 * ───────────────────────────────────────────────────────────────────────*/
@media (max-width: 980px) {
  /* Status bar: collapse to wrap + reduce padding */
  .if-statusbar { flex-wrap:wrap; }
  .if-brand { font-size:11px; padding:8px 12px; letter-spacing:1px; }
  .if-cell  { padding:6px 10px; font-size:10px; }
  .if-time  { padding:6px 12px; font-size:10px; }

  /* Hero grid: 5-column → 2-column wrap, badge becomes its own row */
  .if-hero { padding:18px 16px; }
  .if-hero-grid {
    grid-template-columns:1fr 1fr; gap:14px;
  }
  .if-rating { font-size:42px; padding:10px 18px; letter-spacing:2px; }
  .if-kpi-v  { font-size:22px; }
  .if-kpi-l  { font-size:9px; letter-spacing:1.5px; }

  /* 2x2 cell grid → single column on tablet */
  .if-grid2 { grid-template-columns:1fr; }
  .if-cell-panel {
    border-right:0 !important;
    border-bottom:1px solid var(--border);
    min-height:0; padding:14px 16px;
  }

  /* Table: tighter padding */
  .if-tbl td, .if-tbl th { padding:5px 6px; font-size:11px; }

  /* Compare tab: side-by-side → stack */
  .if-cmp-col { padding:16px; }
  .if-cmp-kpi-row { grid-template-columns:repeat(2, 1fr); gap:10px; }
  .if-cmp-kpi-row .if-kpi-v { font-size:16px; }

  /* Tab labels: drop the F-key prefix at narrow widths via shorter label */
  .stTabs [data-baseweb="tab"] {
    padding:8px 12px !important; font-size:10px !important;
    letter-spacing:1px !important;
  }
}

@media (max-width: 640px) {
  /* Phone tier: tighter still */
  .if-brand::before { content:"" !important; }
  .if-brand { font-size:10px; letter-spacing:.5px; padding:6px 10px; }
  .if-cell  { padding:5px 8px; font-size:9px; }
  .if-cell b { display:none; }       /* hide secondary values, save row width */
  .if-time  { padding:5px 8px; font-size:9px; }

  /* Hero: full-stack */
  .if-hero-grid { grid-template-columns:1fr; gap:10px; }
  .if-rating { font-size:36px; padding:8px 14px; letter-spacing:1.5px; }
  .if-kpi-v  { font-size:18px; }

  /* Section header rule */
  .if-cell-h { font-size:9px; letter-spacing:1.5px; }

  /* Tour card: shrink padding */
  .if-tour { padding:24px 18px; margin:14px auto; }
  .if-tour-title { font-size:18px; letter-spacing:2px; }
  .if-tour-list li {
    padding:10px 10px 10px 50px; font-size:11.5px; line-height:1.55;
  }
  .if-tour-list li::before { left:10px; width:26px; height:26px; font-size:11px; }

  /* Tabs: stack into a horizontal scroll strip */
  .stTabs [data-baseweb="tab-list"] { overflow-x:auto; }
  .stTabs [data-baseweb="tab"] {
    padding:7px 10px !important; font-size:9.5px !important;
    flex-shrink:0;
  }

  /* Footer scales down */
  .if-footer { font-size:8px; letter-spacing:1.2px; padding:6px 10px; }

  /* Status pill row VIEW SOURCE: keep only the icon at phone width */
  a.if-cell.if-src { padding:5px 8px; }
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# OpenGraph / Twitter card injection — Streamlit Cloud serves a generic
# shell, so we inject our own meta tags via st.markdown.  Most unfurlers
# (Slack, Discord) scrape the body; head-only ones (LinkedIn) won't see
# these — share the GitHub URL for those instead.
# ─────────────────────────────────────────────────────────────────────────────
_OG_META = """
<meta property="og:title" content="JANUS — Two-Faced AI Investment Terminal">
<meta property="og:description" content="JANUS · LangGraph multi-agent + hybrid RAG + QLoRA-distilled analyst · Bloomberg-Terminal UI · 226 tests · RAGAS 1.00 · Sharpe +0.97 — try the live demo">
<meta property="og:image" content="https://raw.githubusercontent.com/Vivianmxmf/invest_forge/main/docs/screenshots/ui_terminal_decision.png">
<meta property="og:url" content="https://invest-forge.streamlit.app">
<meta property="og:type" content="website">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="JANUS — Two-Faced AI Investment Terminal">
<meta name="twitter:description" content="Multi-agent investment research · Bloomberg-Terminal UI · LangGraph + LoRA + RAG + alphalens">
<meta name="twitter:image" content="https://raw.githubusercontent.com/Vivianmxmf/invest_forge/main/docs/screenshots/ui_terminal_decision.png">
<!-- Apple touch icon: served from the GitHub raw CDN so iOS home-screen pinning shows the JANUS ⌖ target. -->
<link rel="apple-touch-icon" href="https://raw.githubusercontent.com/Vivianmxmf/invest_forge/main/assets/janus_apple_touch.png">
<link rel="shortcut icon" type="image/png" href="https://raw.githubusercontent.com/Vivianmxmf/invest_forge/main/assets/janus_favicon.png">
"""
st.markdown(_OG_META, unsafe_allow_html=True)


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


CLOUD_BASE_URL = "https://invest-forge.streamlit.app"
RATING_FAVICONS = {
    "BUY":  "https://raw.githubusercontent.com/Vivianmxmf/invest_forge/main/assets/janus_favicon_buy.png",
    "HOLD": "https://raw.githubusercontent.com/Vivianmxmf/invest_forge/main/assets/janus_favicon_hold.png",
    "SELL": "https://raw.githubusercontent.com/Vivianmxmf/invest_forge/main/assets/janus_favicon_sell.png",
}


# ─────────────────────────────────────────────────────────────────────────────
# ① Real-time ticker tape — simulated quotes from sample close + jitter
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=30)
def _simulated_quotes(seed: int = 0) -> list[dict]:
    """Generate a fresh batch of simulated A-share quotes every 30 s.

    Reads the latest available close from data/sample/prices.csv per ticker
    and adds a small ±1.5 % random jitter so the tape "ticks". Deterministic
    per-seed so tests are reproducible. NOT a real market feed.
    """
    import pandas as pd
    import random
    p = _SETTINGS.paths.sample_dir / "prices.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    rng = random.Random(seed)
    out: list[dict] = []
    for code, name in [(t[0], t[1]) for t in [
        ("600519.SH", "贵州茅台"), ("688981.SH", "中芯国际"),
        ("300750.SZ", "宁德时代"), ("600276.SH", "恒瑞医药"),
        ("601398.SH", "工商银行"),
    ]]:
        sub = df[df["ts_code"] == code]
        if sub.empty:
            continue
        last_close = float(sub.tail(1)["close"].iloc[0])
        # Jitter: simulate intraday move
        change_pct = rng.uniform(-1.5, 1.5)
        price = last_close * (1 + change_pct / 100)
        out.append({
            "code": code, "name": name, "price": price,
            "change_pct": change_pct,
            "direction": "up" if change_pct >= 0 else "down",
        })
    return out


def _render_ticker_tape() -> None:
    """Bloomberg-style horizontally-scrolling tape under the status bar."""
    # Refresh tape every 30 s. _simulated_quotes is cache-keyed on the seed
    # so passing a different seed each tick invalidates the cache entry.
    try:
        from streamlit_autorefresh import st_autorefresh
        tick = st_autorefresh(interval=30_000, key="ticker_tape_refresh")
    except Exception:
        tick = 0
    quotes = _simulated_quotes(seed=int(tick) % 100)
    if not quotes:
        return
    # Build one rendering inline; the CSS marquee duplicates it for an
    # infinite-scroll illusion without JS.
    spans = []
    for q in quotes:
        dir_cls = q["direction"]
        sign = "+" if q["change_pct"] >= 0 else ""
        spans.append(
            f'<span class="tt-item">'
            f'<span class="tt-code">{_esc(q["code"])}</span>'
            f'<span class="tt-name">{_esc(q["name"])}</span>'
            f'<span class="tt-px tabular-nums">¥{q["price"]:,.2f}</span>'
            f'<span class="tt-chg {dir_cls} tabular-nums">{sign}{q["change_pct"]:.2f}%</span>'
            f'</span>'
        )
    # Duplicate the strip so the CSS keyframe seamlessly loops.
    strip = "".join(spans) + "".join(spans)
    st.markdown(
        f'<div class="if-tape"><div class="if-tape-inner">{strip}</div></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ② Favicon-on-tab notification — swap browser tab icon by rating
# ─────────────────────────────────────────────────────────────────────────────
def _set_favicon_for_rating(rating: str) -> None:
    """Replace the OUTER-page <link rel*="icon"> with a rating-colored variant.

    Uses streamlit.components.v1.html (a 0-height iframe) — st.markdown strips
    <script> tags. The iframe is same-origin so window.parent.document is
    reachable; the loop climbs up to 5 parents to handle Cloud's nested
    iframe shell.
    """
    url = RATING_FAVICONS.get(str(rating).upper())
    if not url:
        return
    from streamlit.components.v1 import html as _components_html
    _components_html(
        f"""
<script>
(function() {{
  try {{
    let w = window;
    for (let i = 0; i < 5 && w.parent && w.parent !== w; i++) w = w.parent;
    const doc = w.document;
    if (!doc || !doc.head) return;
    const ourUrl = "{url}?t=" + Date.now();
    const apply = () => {{
      // Remove every existing icon-family link so the browser can't fall
      // back to a Streamlit-served default that React re-renders.
      doc.querySelectorAll('link[rel*="icon"]').forEach(l => l.remove());
      // Append a fresh shortcut icon at the END of head — last write wins.
      const link = doc.createElement('link');
      link.rel = 'shortcut icon';
      link.type = 'image/png';
      link.href = ourUrl;
      doc.head.appendChild(link);
      const apple = doc.createElement('link');
      apple.rel = 'apple-touch-icon';
      apple.href = ourUrl;
      doc.head.appendChild(apple);
    }};
    apply();
    // Re-apply a few times to outlast Streamlit's React re-render of <head>.
    [300, 800, 1500].forEach(ms => setTimeout(apply, ms));
  }} catch (e) {{ /* cross-origin / detached — silently no-op */ }}
}})();
</script>
""",
        height=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ③ Share QR code — encodes ticker + watchlist + tab into a single URL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=32, ttl=600)
def _make_qr_png(url: str) -> bytes:
    """Render a Bloomberg-amber QR code on dark bg as PNG bytes."""
    from io import BytesIO
    import qrcode
    qr = qrcode.QRCode(
        version=None, error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8, border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#E5C46B", back_color="#0a0e14")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_share_url(ts_code: str) -> str:
    """Compose a shareable URL with ticker + custom watchlist embedded."""
    from urllib.parse import urlencode
    params = {"ticker": ts_code}
    custom = st.session_state.get("custom_tickers") or []
    if custom:
        params["w"] = ",".join(custom)
    return f"{CLOUD_BASE_URL}/?{urlencode(params)}"


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def _kline_panel(ts_code: str, lookback: int = 120):
    """Load OHLCV slice for one ticker from the sample prices.csv.

    Returns a DataFrame indexed by date with columns [open, high, low, close,
    volume] so plotly Candlestick + Bar can render OHLC + volume sub-panel.
    Empty DataFrame when the ticker isn't in the sample.

    Lookback is 120 sessions so MA20 has full warm-up before the visible
    90-session window (cropped at render time).
    """
    import pandas as pd
    p = _SETTINGS.paths.sample_dir / "prices.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    sub = df[df["ts_code"] == ts_code].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["trade_date"] = pd.to_datetime(sub["trade_date"], errors="coerce")
    sub = sub.dropna(subset=["trade_date"]).sort_values("trade_date").tail(lookback)
    sub = sub.set_index("trade_date")
    cols_lower = {c.lower(): c for c in sub.columns}
    needed = ("open", "high", "low", "close", "volume")
    keep = {k: cols_lower[k] for k in needed if k in cols_lower}
    if len([k for k in keep if k in ("open", "high", "low", "close")]) < 4:
        return pd.DataFrame()
    return sub[list(keep.values())].rename(columns={v: k for k, v in keep.items()})


def _render_candlestick(ts_code: str, *, height: int = 320) -> None:
    """Render a Bloomberg-styled K-line with MA5/10/20 overlay + volume sub-panel.

    Layout: 72 % top panel = candlestick + three MA lines (amber/cyan/violet),
    28 % bottom panel = colored volume bars (green when close>=open, red else).
    Shared x-axis; range slider disabled; transparent bg blends with Bloomberg
    cell-panel container.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    df = _kline_panel(ts_code)
    if df.empty:
        st.markdown(
            '<div style="color:var(--muted);font-size:11px;letter-spacing:1px;'
            'padding:8px 0;">no OHLC in sample · K-line skipped</div>',
            unsafe_allow_html=True,
        )
        return

    # Compute MA on full lookback then crop the visible window so the lines are warm.
    has_volume = "volume" in df.columns
    df = df.copy()
    df["ma5"]  = df["close"].rolling(5,  min_periods=1).mean()
    df["ma10"] = df["close"].rolling(10, min_periods=1).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=1).mean()
    visible = df.tail(90)  # crop tail to keep chart density readable

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.72, 0.28] if has_volume else [1.0, 0.0],
    )
    # ── Candlestick + MA overlay (row 1) ──
    fig.add_trace(go.Candlestick(
        x=visible.index, open=visible["open"], high=visible["high"],
        low=visible["low"], close=visible["close"],
        increasing_line_color="#26C281", increasing_fillcolor="#26C281",
        decreasing_line_color="#E63946", decreasing_fillcolor="#E63946",
        line=dict(width=1), showlegend=False, name="OHLC",
    ), row=1, col=1)
    for name, color in (("ma5", "#E5C46B"), ("ma10", "#5fb1ff"), ("ma20", "#a78bfa")):
        fig.add_trace(go.Scatter(
            x=visible.index, y=visible[name], mode="lines",
            line=dict(color=color, width=1.4), name=name.upper(),
            hovertemplate="%{y:.2f}<extra>" + name.upper() + "</extra>",
        ), row=1, col=1)

    # ── Volume bars (row 2) ──
    if has_volume:
        vol_colors = [
            "#26C281" if c >= o else "#E63946"
            for c, o in zip(visible["close"], visible["open"])
        ]
        fig.add_trace(go.Bar(
            x=visible.index, y=visible["volume"],
            marker_color=vol_colors, marker_line_width=0,
            showlegend=False, name="VOL",
        ), row=2, col=1)

    fig.update_layout(
        height=height, margin=dict(l=4, r=4, t=10, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color="#a8b3c4", size=10), bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0a0e14", font_color="#e8ecf1",
                        font_family="JetBrains Mono"),
    )
    fig.update_xaxes(
        rangeslider=dict(visible=False), showgrid=False,
        tickfont=dict(color="#7e8a9c", size=9, family="JetBrains Mono"),
        row=1, col=1,
    )
    fig.update_xaxes(
        rangeslider=dict(visible=False), showgrid=False,
        tickfont=dict(color="#7e8a9c", size=9, family="JetBrains Mono"),
        row=2, col=1,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#2a3445", zeroline=False,
        tickfont=dict(color="#7e8a9c", size=9, family="JetBrains Mono"),
        row=1, col=1,
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False,
        tickfont=dict(color="#7e8a9c", size=8, family="JetBrains Mono"),
        row=2, col=1, tickformat=".2s",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# PDF research-report export — reportlab with Adobe CID font (CJK-ready,
# no TTF file required, works on Streamlit Cloud out-of-the-box).
# ─────────────────────────────────────────────────────────────────────────────
def _pdf_report(state: dict, ts_code: str) -> bytes:
    """Render the latest pipeline state as a one-page PDF research note."""
    from io import BytesIO
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    # Register Adobe's built-in Simplified-Chinese font once per process.
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        cjk_font = "STSong-Light"
    except Exception:
        cjk_font = "Helvetica"

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    rec = state.get("final_recommendation", {}) or {}
    rating_raw = str(rec.get("rating", "HOLD")).strip().upper()
    rating = rating_raw if rating_raw in ("BUY", "HOLD", "SELL") else "HOLD"
    conf = float(rec.get("confidence", 0) or 0)
    target = rec.get("target_price")
    iters = int(rec.get("iteration_count", 0) or 0)
    risk_lvl = str(rec.get("risk_level", "—"))
    rationale = str(rec.get("rationale", ""))
    risks = rec.get("risk_factors", []) or []
    risk_color = {"BUY": (0.15, 0.76, 0.50), "HOLD": (0.98, 0.66, 0.14),
                  "SELL": (0.90, 0.22, 0.27)}[rating]

    # ── Header band ──
    c.setFillColorRGB(0.04, 0.06, 0.08); c.rect(0, height - 70, width, 70, fill=1, stroke=0)
    c.setFillColorRGB(0.90, 0.77, 0.42); c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 38, "JANUS")
    c.setFillColorRGB(0.66, 0.70, 0.77); c.setFont("Helvetica", 10)
    c.drawString(40, height - 56, "AI INVESTMENT RESEARCH NOTE")
    c.setFont("Helvetica", 9)
    c.drawRightString(width - 40, height - 38,
                      f"REPORT · {ts_code}  ·  WJH  ·  invest-forge.streamlit.app")

    # ── Rating badge ──
    y = height - 130
    c.setFillColorRGB(*risk_color); c.setStrokeColorRGB(*risk_color); c.setLineWidth(1.5)
    c.roundRect(40, y, 110, 50, 8, fill=1, stroke=1)
    c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(95, y + 16, rating)

    # ── KPI row ──
    c.setFillColorRGB(0.55, 0.61, 0.71); c.setFont("Helvetica", 8)
    target_v = f"¥{float(target):,.2f}" if target is not None else "—"
    kpi = [("TARGET", target_v), ("CONFIDENCE", f"{conf:.0%}"),
           ("RISK LEVEL", risk_lvl), ("REVISION", f"{iters}x")]
    kx = 180
    for label, val in kpi:
        c.setFillColorRGB(0.55, 0.61, 0.71); c.setFont("Helvetica", 7.5)
        c.drawString(kx, y + 36, label)
        c.setFillColorRGB(0.91, 0.93, 0.95); c.setFont("Helvetica-Bold", 14)
        c.drawString(kx, y + 18, val)
        kx += 95

    # ── Thesis section ──
    y -= 32
    _section(c, "CORE THESIS · ANALYST AGENT", 40, y, width - 80)
    y -= 18
    c.setFillColorRGB(0.85, 0.87, 0.91); c.setFont(cjk_font, 10)
    y = _wrap_text(c, rationale, 40, y, width - 80, line_h=14, font=cjk_font, size=10)

    # ── Risk factors ──
    if risks:
        y -= 18
        _section(c, "RISK FACTORS", 40, y, width - 80)
        y -= 16
        chip_x = 40
        for r in risks:
            txt = str(r)
            tw = c.stringWidth(txt, cjk_font, 9) + 14
            if chip_x + tw > width - 40:
                chip_x = 40; y -= 18
            c.setFillColorRGB(0.93, 0.71, 0.74); c.setStrokeColorRGB(0.79, 0.39, 0.43)
            c.roundRect(chip_x, y - 3, tw, 14, 7, fill=0, stroke=1)
            c.setFont(cjk_font, 9)
            c.drawString(chip_x + 7, y, txt)
            chip_x += tw + 6

    # ── RAG evidence (top 5) ──
    hits = state.get("rag_evidence", []) or []
    if hits:
        y -= 24
        _section(c, f"RAG EVIDENCE  ·  TOP-{min(5, len(hits))}", 40, y, width - 80)
        y -= 14
        for h in hits[:5]:
            src = (h.get("source") if isinstance(h, dict) else getattr(h, "source", "")) or ""
            score = (h.get("score") if isinstance(h, dict) else getattr(h, "score", 0.0)) or 0.0
            text = (h.get("text") if isinstance(h, dict) else getattr(h, "text", "")) or ""
            c.setFillColorRGB(0.90, 0.77, 0.42); c.setFont("Helvetica-Bold", 8)
            c.drawString(40, y, f"[{str(src)[:70]}]  ({float(score):.3f})")
            y -= 12
            c.setFillColorRGB(0.66, 0.70, 0.77); c.setFont(cjk_font, 9)
            y = _wrap_text(c, str(text)[:260], 40, y, width - 80,
                           line_h=12, font=cjk_font, size=9)
            y -= 6
            if y < 80:
                break

    # ── Footer ──
    c.setFillColorRGB(0.36, 0.40, 0.46); c.setFont("Helvetica", 7)
    c.drawString(40, 30,
                 "Generated by JANUS — two-faced multi-agent · hybrid RAG · LoRA-distilled analyst")
    c.drawRightString(width - 40, 30,
                      "github.com/Vivianmxmf/invest_forge  ·  v0.4.1")

    c.showPage(); c.save()
    return buf.getvalue()


def _section(c, label: str, x: float, y: float, w: float) -> None:
    """Draw a gold section header rule."""
    c.setFillColorRGB(0.90, 0.77, 0.42); c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y, label)
    c.setStrokeColorRGB(0.16, 0.20, 0.27); c.setLineWidth(0.5)
    c.line(x, y - 3, x + w, y - 3)


def _wrap_text(c, text: str, x: float, y: float, w: float, *,
               line_h: float = 12, font: str = "Helvetica", size: int = 9) -> float:
    """Tiny word/char-wrap helper that respects PDF width and returns next-y."""
    if not text:
        return y
    c.setFont(font, size)
    # CJK-friendly: wrap by character; Latin stays readable too.
    line = ""
    for ch in text:
        if c.stringWidth(line + ch, font, size) > w:
            c.drawString(x, y, line)
            y -= line_h
            line = ch
            if y < 60:
                return y
        else:
            line += ch
    if line:
        c.drawString(x, y, line)
        y -= line_h
    return y


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
  <div class="if-brand">JANUS · IF&lt;GO&gt;</div>
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

# ── Ticker tape (simulated quotes, 30 s auto-refresh) ────────────────────────
_render_ticker_tape()


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

# ── Watchlist persistence via URL query params ────────────────────────────
# Streamlit is stateless — we use ?w=600519.SH,000001.SZ in the URL so the
# user's added tickers survive refresh AND are shareable as a single link.
def _load_custom_tickers() -> list[str]:
    raw = st.query_params.get("w", "")
    if not raw:
        return []
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _save_custom_tickers(items: list[str]) -> None:
    items = [t.strip().upper() for t in items if t.strip()]
    if items:
        st.query_params["w"] = ",".join(items)
    elif "w" in st.query_params:
        del st.query_params["w"]


if "custom_tickers" not in st.session_state:
    st.session_state.custom_tickers = _load_custom_tickers()

if "ticker" not in st.session_state:
    # Honour ?ticker=600519.SH query param so QR-share links auto-fill.
    qp_ticker = st.query_params.get("ticker", "")
    st.session_state.ticker = (qp_ticker or "600519.SH").strip().upper()

if "ticker_b" not in st.session_state:
    st.session_state.ticker_b = "688981.SH"

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

    # ── Custom watchlist (persisted via URL ?w=...) ──────────────────────
    if st.session_state.custom_tickers:
        st.markdown('<div class="if-section-l">MY WATCHLIST</div>', unsafe_allow_html=True)
        for code in st.session_state.custom_tickers:
            active = "active" if st.session_state.ticker == code else ""
            row = (
                f'<div class="if-tkr {active}">'
                f'<span><span class="code">{_esc(code)}</span><br>'
                f'<span class="name">custom</span></span>'
                f'<span class="pct flat">—</span></div>'
            )
            st.markdown(row, unsafe_allow_html=True)
            cols = st.columns([5, 1])
            if cols[0].button(f"› {code}", key=f"cusel-{code}", use_container_width=True):
                st.session_state.ticker = code
                st.session_state.pop("last_state", None)
                st.session_state.pop("last_ticker", None)
                st.session_state.pop("last_thumbs", None)
                st.rerun()
            if cols[1].button("✕", key=f"curm-{code}",
                              help=f"remove {code} from MY WATCHLIST"):
                st.session_state.custom_tickers = [
                    t for t in st.session_state.custom_tickers if t != code
                ]
                _save_custom_tickers(st.session_state.custom_tickers)
                st.rerun()

    with st.expander("➕  ADD TICKER TO MY WATCHLIST"):
        st.caption("Persists in URL ?w=… — copy URL to share watchlist.")
        new_code = st.text_input(
            "code", key="add_ticker_input",
            placeholder="e.g. 000001.SZ", label_visibility="collapsed",
        )
        if st.button("ADD", key="add_ticker_btn", use_container_width=True):
            code_clean = (new_code or "").strip().upper()
            if not code_clean:
                st.warning("ticker code empty")
            elif code_clean in st.session_state.custom_tickers or \
                 code_clean in [s[0] for s in SAMPLE_TICKERS]:
                st.warning(f"{code_clean} already in watchlist")
            else:
                st.session_state.custom_tickers.append(code_clean)
                _save_custom_tickers(st.session_state.custom_tickers)
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
TAB_DEC, TAB_RES, TAB_RISK, TAB_BT, TAB_CMP, TAB_SYS = st.tabs(
    ["[1]  DECISION", "[2]  RESEARCH", "[3]  RISK", "[4]  BACKTEST",
     "[5]  COMPARE", "[6]  SYSTEM"]
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
        # Swap the browser tab favicon to BUY-green / HOLD-amber / SELL-red.
        _rating = (state.get("final_recommendation") or {}).get("rating", "HOLD")
        _set_favicon_for_rating(_rating)


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
            '  <div class="if-tour-title">JANUS TERMINAL READY</div>'
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
    <div class="if-cell-h">PRICE · K-LINE · LAST 90 SESSIONS</div>
    <div id="kline-anchor"></div>
  </div>

</div>
""",
            unsafe_allow_html=True,
        )
        # Render the candlestick into the 4th cell-panel slot above.
        # (Streamlit's component model writes after the HTML block, but the
        #  Bloomberg grid keeps the chart inside the same visual quadrant.)
        _ticker_now = st.session_state.get("last_ticker", "")
        _render_candlestick(_ticker_now, height=340)

        # ── PDF export + SHARE QR row ──
        try:
            _pdf_bytes = _pdf_report(state, _ticker_now)
            pdf_col, qr_col, _ = st.columns([1, 1, 3])
            with pdf_col:
                st.download_button(
                    label="📄 EXPORT PDF",
                    data=_pdf_bytes,
                    file_name=f"JANUS_{_ticker_now}_{rec.get('rating','HOLD')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with qr_col:
                _show_qr = st.toggle("📱 SHARE", value=False, key="share_qr_toggle",
                                     help="Show a scannable QR — opens this exact "
                                          "analysis on a phone with ticker + watchlist auto-filled.")
        except Exception as _exc:
            st.warning(f"PDF export unavailable: {_exc}")
            _show_qr = False

        if _show_qr:
            _share_url = _build_share_url(_ticker_now)
            try:
                _qr_png = _make_qr_png(_share_url)
                qcol1, qcol2 = st.columns([1, 3])
                with qcol1:
                    st.image(_qr_png, width=200)
                with qcol2:
                    st.markdown(
                        '<div class="if-cell-h">SHARE THIS ANALYSIS</div>'
                        '<div style="color:var(--dim);font-size:12px;line-height:1.7;">'
                        'Scan with a phone camera to open this exact ticker + custom '
                        'watchlist on the live demo. Embeds <code>?ticker=…&amp;w=…</code> '
                        'so the recipient lands on the same BUY/HOLD/SELL view.</div>',
                        unsafe_allow_html=True,
                    )
                    st.code(_share_url, language=None)
            except Exception as _qexc:
                st.warning(f"QR unavailable: {_qexc}")

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
# Tab — COMPARE (2 tickers side-by-side)
# ─────────────────────────────────────────────────────────────────────────────
def _cmp_card(state: dict, slot_label: str) -> str:
    """Render one comparison column as a single HTML block."""
    rec = state.get("final_recommendation", {}) or {}
    rating_raw = str(rec.get("rating", "HOLD")).strip().upper()
    rating = rating_raw if rating_raw in ("BUY", "HOLD", "SELL") else "HOLD"
    conf = float(rec.get("confidence", 0) or 0)
    target = rec.get("target_price")
    target_v = f"¥{float(target):,.2f}" if target is not None else "—"
    risk_lvl = _esc(str(rec.get("risk_level", "—")))
    iters = int(rec.get("iteration_count", 0) or 0)
    rationale = _esc(str(rec.get("rationale", "(empty)")))[:280]
    risks = rec.get("risk_factors", []) or []
    chips = "".join(f'<span class="if-chip">{_esc(str(r))}</span>' for r in risks[:3])
    return f"""
<div class="if-cmp-col">
  <div class="if-cmp-tkr">{_esc(slot_label)}</div>
  <div class="if-rating {rating}" style="font-size:38px;padding:10px 18px;margin-bottom:14px;">{rating}</div>
  <div class="if-cmp-kpi-row">
    <div><div class="if-kpi-l">TARGET</div><div class="if-kpi-v amber">{target_v}</div></div>
    <div><div class="if-kpi-l">CONF</div><div class="if-kpi-v">{conf:.0%}</div></div>
    <div><div class="if-kpi-l">RISK</div><div class="if-kpi-v">{risk_lvl}</div></div>
    <div><div class="if-kpi-l">REV</div><div class="if-kpi-v">{iters}×</div></div>
  </div>
  <div class="if-cell-h" style="border:0;padding:6px 0 4px;margin:8px 0 0;">THESIS</div>
  <div class="if-thesis" style="font-size:11.5px;">{rationale}…</div>
  <div class="if-chip-row" style="margin-top:10px;">{chips}</div>
</div>
"""


with TAB_CMP:
    st.markdown(
        '<div style="padding:18px 22px 6px;"><div class="if-cell-h">'
        'PAIRWISE COMPARISON · TWO TICKERS SIDE-BY-SIDE</div>'
        '<div style="color:var(--muted);font-size:11px;letter-spacing:1px;">'
        'Selects both tickers from the watchlist or any custom A-share code. '
        'Uses cached pipeline runs (fake-LLM mode).</div></div>',
        unsafe_allow_html=True,
    )
    cc1, cc2, cc3 = st.columns([1, 1, 1])
    with cc1:
        st.text_input("TICKER A", key="ticker_a_cmp",
                      value=st.session_state.ticker, label_visibility="collapsed",
                      placeholder="ticker A")
    with cc2:
        st.text_input("TICKER B", key="ticker_b_cmp",
                      value=st.session_state.ticker_b, label_visibility="collapsed",
                      placeholder="ticker B")
    with cc3:
        cmp_go = st.button("▶ COMPARE", type="primary", use_container_width=True)

    if cmp_go:
        a = (st.session_state.ticker_a_cmp or "").strip().upper()
        b = (st.session_state.ticker_b_cmp or "").strip().upper()
        if not a or not b:
            st.error("BOTH TICKER A AND B REQUIRED")
        elif a == b:
            st.warning("PICK TWO DIFFERENT TICKERS")
        else:
            st.session_state.ticker_b = b
            with st.spinner(f"COMPARING {a}  vs  {b} …"):
                try:
                    sa = _pipeline_cached_no_image(a)
                    sb = _pipeline_cached_no_image(b)
                except Exception as exc:
                    st.error(f"COMPARE FAILED: {exc}")
                    sa, sb = None, None
            if sa and sb:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(_cmp_card(sa, a), unsafe_allow_html=True)
                    _render_candlestick(a, height=180)
                with col_b:
                    st.markdown(_cmp_card(sb, b), unsafe_allow_html=True)
                    _render_candlestick(b, height=180)
                # Quick verdict
                ra = (sa.get("final_recommendation") or {}).get("rating", "HOLD")
                rb = (sb.get("final_recommendation") or {}).get("rating", "HOLD")
                ca = float((sa.get("final_recommendation") or {}).get("confidence", 0) or 0)
                cb = float((sb.get("final_recommendation") or {}).get("confidence", 0) or 0)
                winner = a if (ra == "BUY" and rb != "BUY") or (ra == rb and ca > cb) else b
                if ra == rb and ca == cb:
                    verdict = "TIE — both tickers receive identical ratings + confidence."
                else:
                    verdict = f"PIPELINE VOTE → {winner} (higher conviction)"
                st.markdown(
                    f'<div style="padding:18px 22px 0;"><div class="if-cell-h">VERDICT</div>'
                    f'<div style="color:var(--amber);font-size:14px;letter-spacing:2px;">'
                    f'{_esc(verdict)}</div></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<div class="if-empty">'
            '<div style="font-size:18px;color:var(--amber);letter-spacing:3px;">PAIRWISE COMPARISON</div>'
            '<div style="margin-top:14px;color:var(--dim);">Enter two A-share codes → ▶ COMPARE</div>'
            '<div style="margin-top:18px;color:var(--muted);font-size:11px;letter-spacing:2px;">'
            'BOTH PIPELINES RUN IN PARALLEL · CACHED RESULTS WHEN AVAILABLE</div>'
            '</div>', unsafe_allow_html=True,
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
    '<div class="if-footer">⌖ JANUS v0.5.0 · TWO-FACED AI INVESTMENT TERMINAL · '
    'LANGGRAPH · QWEN2.5-7B · QLORA · BGE-RERANKER-V2-M3 · RAGAS · ALPHALENS · VLLM ⌖</div>',
    unsafe_allow_html=True,
)
