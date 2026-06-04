# Deploy InvestForge to Streamlit Community Cloud

> **Goal:** publish the Bloomberg-Terminal dashboard at a permanent public URL like
> `https://invest-forge.streamlit.app` — free tier, ~5 min from this checklist
> to a live link recruiters can click directly from your CV.

---

## Pre-flight (already done in the repo)

| File | Why |
|---|---|
| `frontend/app.py` | Entry point for Streamlit Cloud |
| `requirements.txt` | Streamlit + LangGraph + RAG stack (CPU-only paths) |
| `.streamlit/config.toml` | Bloomberg-Terminal dark theme baked in, no light-flash |
| `runtime.txt` | Pins `python-3.12` so cold-starts are reproducible |
| `data/sample/` | Synthetic prices/news/RAGAS eval set committed (141 KB) |
| `.env.example` | Reference for which env vars the app reads — **no secrets shipped** |

The app boots in `LLM_PROVIDER=fake` by default — **no API keys required** for
the public demo to work. Real-model paths (`local` / `openai`) are env-gated.

---

## Manual deploy — 3 clicks at share.streamlit.io

### 1 · Sign in
- Open <https://share.streamlit.io>
- **"Continue with GitHub"** → authorize Streamlit's GitHub app on
  `Vivianmxmf/janus-terminal`.

### 2 · "New app" wizard

| Field | Value |
|---|---|
| **Repository** | `Vivianmxmf/janus-terminal` |
| **Branch** | `main` |
| **Main file path** | `frontend/app.py` |
| **App URL (subdomain)** | `invest-forge` &nbsp;⇒&nbsp; final URL = `invest-forge.streamlit.app` |
| **Python version** | leave default — `runtime.txt` already pins 3.12 |

Click **Advanced settings** → **Secrets** (TOML format):

```toml
# Minimum for a fake-mode public demo — nothing sensitive.
LLM_PROVIDER = "fake"
LOG_LEVEL = "INFO"

# If you want REAL Tushare data later, uncomment + paste your token here.
# TUSHARE_TOKEN = "xxxxxxxxxxxx"
```

### 3 · "Deploy"
- First cold-build takes **3–6 minutes** (installs ~50 deps; `sentence-transformers`
  is the heaviest at ~250 MB).
- Streamlit Cloud auto-rebuilds on every `git push` to `main`.

---

## Once it's live

- **Smoke-test:** open the URL → status bar shows `● LLM fake-llm · DATA fake` →
  click any sample ticker → hit **▶ EXECUTE PIPELINE** → expect a BUY hero card
  in under 2 seconds (cache pre-warmed on cold-boot).
- **Update README badge:** replace the placeholder Live Demo badge target with
  your actual URL — the badge already lives at the top of `README.md`.

---

## If the cold build fails (most common causes)

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: streamlit` | Streamlit Cloud couldn't read `requirements.txt` | Confirm file is at repo root, not in a subdir |
| `Killed` mid-install | OOM on free tier (1 GB RAM) | Drop `sentence-transformers` from requirements; the fake-LLM demo doesn't need it |
| `ImportError: torch` | `sentence-transformers` pulled it but the CPU wheel timed out | Add `--extra-index-url https://download.pytorch.org/whl/cpu` to a `requirements-cloud.txt` and point Cloud to it |
| Slow first-paint (10 s+) | Cold-boot cache preheat running 5 tickers | Expected — only first visitor pays; subsequent visitors get cached results |

---

## Optional polish (after live URL is up)

1. Add the **live URL to your CV / LinkedIn** — recruiters click → 5-step tour →
   they understand the project in 30 s without cloning.
2. Pin a **GitHub Topic** `streamlit` + `langgraph` + `quant` on the repo so it
   shows up in Streamlit's gallery.
3. Add `analytics.streamlit.io` link in the app footer if you want usage stats.

---

> Maintained alongside the codebase — keep this file synced when you change
> `requirements.txt`, `.streamlit/config.toml`, or the entry-point path.
