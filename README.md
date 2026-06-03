# InvestForge — AI 主观投研助手

> LangGraph Multi-Agent · Hybrid RAG · QLoRA-distilled analyst (vLLM-served) · Multimodal vision · LangSmith observable

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Bloomberg%20Terminal-E5C46B?style=for-the-badge&logo=streamlit&logoColor=black)](https://github.com/Vivianmxmf/invest_forge)
[![Tests](https://img.shields.io/badge/tests-226%20passed%20%2F%202%20skipped-26C281?style=for-the-badge&logo=pytest&logoColor=white)](#testing-validation)
[![RAGAS](https://img.shields.io/badge/RAGAS-1.00%20%2F%201.00%20%2F%201.00-26C281?style=for-the-badge)](#w3--ragas-evaluation)
[![Sharpe](https://img.shields.io/badge/Sharpe-+0.97-26C281?style=for-the-badge)](#w4--alphalens-backtest)
[![LoRA](https://img.shields.io/badge/LoRA-+3.3pp%20acc%20%2F%20−29%25%20MAE-635bff?style=for-the-badge&logo=huggingface&logoColor=white)](#w2--qlora-distillation)
[![Python](https://img.shields.io/badge/python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)

---

## ✨ Live UI Preview

<p align="center">
  <img src="docs/screenshots/ui_terminal_decision.png" alt="InvestForge Bloomberg Terminal — Decision view" width="100%">
  <br><sub><b>Decision view</b> — 54px mono BUY/HOLD/SELL hero · 4 KPIs · 2×2 cell-panel grid (thesis / fundamentals / RAG / backtest snapshot)</sub>
</p>

<p align="center">
  <img src="docs/screenshots/ui_terminal_landing.png" alt="InvestForge — first-visit guided tour" width="100%">
  <br><sub><b>First-time guided tour</b> — 5-step welcome card · pre-warmed cache for 5 A-share sample tickers · instant Watchlist clicks</sub>
</p>

> **Run it locally:** `LLM_PROVIDER=fake python -m streamlit run frontend/app.py` → open `http://localhost:8501`
>
> Bloomberg-terminal aesthetic (JetBrains Mono + amber accent), 5-tab F-key navigation, codex-audited (3 rounds), WCAG-AA contrast verified.

InvestForge is the capstone project of the *AI 主观投研* sprint
([`../JD6_AI主观投研实习生_AI_Agent方向.md`](../JD6_AI主观投研实习生_AI_Agent方向.md)).
Input a stock ticker → the system pulls fundamentals + macro + news, runs a
researcher → analyst → risk-control multi-agent pipeline with conditional
revision loops, and outputs a structured investment recommendation
(BUY / HOLD / SELL + confidence + logic + risk factors).

---

## Architecture

```mermaid
flowchart TD
    U[/"User<br/>ts_code (+ optional images)"/] -->|POST /analyze| API[FastAPI<br/>analyze endpoint]

    subgraph DataLayer["📊 Data layer (DataProvider protocol)"]
        TS[Tushare<br/>fundamentals]
        AK[AKShare<br/>news]
        SAMP[(data/sample/<br/>synthetic 141 KB)]
        SENT[FinBERT / lexicon<br/>sentiment]
    end

    subgraph RAG["🔍 Hybrid RAG"]
        BM25[BM25] --> RER
        BGE[bge-embeddings] --> RER[bge-reranker-v2-m3]
        HYDE[HyDE expand] -.optional.-> BGE
        QDR[(Qdrant /<br/>in-memory)] --> BM25
        QDR --> BGE
    end

    subgraph Vision["🖼️ Multimodal (gated, opt-in)"]
        IMG[/uploaded images/] --> VAL[SSRF-hardened<br/>validator]
        VAL --> VLM[Qwen2.5-VL-7B<br/>vLLM 8002]
    end

    subgraph Graph["🤖 LangGraph state machine (USE_LANGGRAPH=1) · inline fallback"]
        direction TB
        DF[data_fetcher]
        VN["vision-node<br/>(no-op if no images)"]
        R[researcher]
        A[analyst]
        RC[risk_control]
        OUT[output]
        DF --> VN --> R --> A --> RC
        RC -->|HIGH · iter < max_iter| A
        RC -->|LOW / MED| OUT
    end

    subgraph LLM["🧠 LLM layer (LLMClient protocol)"]
        FAKE[Fake LLM<br/>tests / offline]
        CLOUD[OpenAI / Anthropic]
        VLLM[Local vLLM<br/>Qwen2.5-7B 8000]
        ADAPTER[["investforge-analyst<br/>QLoRA r=16 adapter"]]
        VLLM --- ADAPTER
    end

    subgraph Eval["📐 Offline metrics"]
        RAGAS["RAGAS judge<br/>Faithfulness ≥ 0.8"]
        ALPHA["alphalens tear-sheet<br/>IC · IR · Sharpe · MDD"]
    end

    LS[(LangSmith tracing)]

    API --> DF
    DataLayer --> DF
    SENT --> DF
    DF -. evidence .-> RAG
    RAG --> R
    Vision --> VN
    R --> LLM
    RC --> LLM
    A -. build_analyst_client .-> ADAPTER
    OUT --> RESP[/"Recommendation<br/>rating · confidence · rationale ·<br/>risk_factors · target_price"/]
    Graph -. LANGCHAIN_TRACING_V2 .-> LS
    OUT -. measured by .-> RAGAS
    RAG -. measured by .-> RAGAS
    OUT -. signal feeds .-> ALPHA
```

Every external dependency (LLM, vector store, data provider, sentiment
model) sits behind a small Protocol so the *entire* pipeline runs offline
against deterministic fakes — see `tests/` and `invest_forge/llm/fake.py`.
`USE_LANGGRAPH=true` swaps the in-process inline runner for the compiled
LangGraph runtime (which is also what lets LangSmith capture traces).

---

## Layout

```
invest_forge/
├── invest_forge/
│   ├── agents/                    # state · nodes · graph · prompts
│   ├── common/                    # config · logging · types
│   ├── eval/                      # ragas_eval · quality
│   ├── knowledge_base/            # builder · retriever · pdf_loader
│   ├── llm/                       # client protocol · fake · openai · anthropic · hf (in-process)
│   └── tools/                     # data · sentiment · rag · backtest · image_input (vision security)
├── finetune/                      # QLoRA: sft_format · train_lora · eval_lora
├── frontend/                      # Streamlit dashboard (with image upload)
├── api/                           # (inside invest_forge/api/) FastAPI
├── notebooks/                     # Week 1 / 3 / 4 walkthroughs
├── configs/lora.yaml              # QLoRA hyperparameters
├── scripts/                       # generate_sample_data · build_kb · build_sft_dataset · serve_vllm.sbatch · analyze_client.sh
├── docs/                          # W2_LORA_RUNBOOK · VISION_RUNBOOK
├── data/sample/                   # synthetic 5-stock dataset (141 KB) — committed
├── tests/                         # 226 unit tests, network-free
├── docker-compose.yml             # qdrant + vllm + vllm-vision (for Docker hosts)
├── requirements-gpu.txt           # cu121 GPU stack (driver 535 / CUDA 12.2)
├── pyproject.toml · requirements.txt · Makefile
└── README.md
```

---

## Quick start — laptop / dev (no GPU, no API keys)

```bash
bash scripts/setup_local.sh         # venv + deps + sample + pytest
.venv/bin/streamlit run frontend/app.py
```

The dashboard runs against ``LLM_PROVIDER=fake`` (deterministic canned
responses) and reads the synthetic sample dataset.  Drop in real keys
later by editing ``.env``.

## Quick start — server (8× A5000 24G, CUDA 12.2)

> **Hardware:** node1.athena — 8× NVIDIA RTX A5000 24 GB, Driver 535.154.05,
> CUDA 12.2.  Use `cu121` wheels (not `cu124`).

```bash
bash scripts/server_migrate.sh      # idempotent; ~5 min on a fast link

# Install GPU stack (cu121 — required for driver 535):
pip install -r requirements-gpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# After it finishes:
#   • conda env 'invest_forge' (Py 3.11)
#   • core deps + GPU stack (gated on nvidia-smi) installed
#   • Qdrant running on :6333
#   • .env scaffolded — fill OPENAI_API_KEY / ANTHROPIC_API_KEY / TUSHARE_TOKEN

make api          # FastAPI on :8001
make dashboard    # Streamlit on :8501
```

> **No model weights are downloaded by ``server_migrate.sh``.**
> Pull them manually after editing ``.env``:
> ```bash
> huggingface-cli login
> huggingface-cli download Qwen/Qwen2.5-7B-Instruct
> ```

### Self-hosted vLLM serving (Qwen2.5-7B + `investforge-analyst` LoRA)

On a **Docker host**: `docker compose --profile vllm up -d`.

On the **athena SLURM cluster** (no Docker / no sudo), serve via the bundled
job script — run vLLM in its **own conda env** (it pins its own torch /
transformers, so it must not share the `invest_forge` training env):

```bash
# one-time: a dedicated serving env (vllm 0.6.3.post1 is cu121 / driver-535 safe)
conda create -n vllm python=3.11 -y && conda activate vllm
pip install "vllm==0.6.3.post1" "transformers==4.46.3"

# submit the server (handles chat-template + lm-format-enforcer backend):
sbatch scripts/serve_vllm.sbatch                 # tail -f vllm_<jobid>.log
# run the full agent on the SAME node (LoRA analyst + base researcher/risk):
srun --jobid=<jobid> --overlap --pty bash
bash scripts/analyze_client.sh 688981.SH
```

**GPU / port pitfalls (learned the hard way).** `serve_vllm.sbatch` ships with
`--gres=gpu:1` commented out, so SLURM reserves no specific card and vLLM
defaults to **GPU 0**. If GPU 0 is already busy (a stale server, a training
job), startup dies with `CUDA out of memory`; if its port is taken you get
`Address already in use`. Pin a free card and/or an open port at submit time —
both env vars are honoured by the script:

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv   # find an idle card
CUDA_VISIBLE_DEVICES=6 PORT=8001 sbatch scripts/serve_vllm.sbatch
VLLM_PORT=8001 bash scripts/analyze_client.sh 688981.SH  # client must match the port
```

A leftover server squats both the card and the port — `kill <pid>` it (the OOM
trace prints the offending PID) or just dodge to a free card + port as above.

> See [`docs/W2_LORA_RUNBOOK.md`](docs/W2_LORA_RUNBOOK.md) for the full
> fine-tuning workflow and [`scripts/serve_vllm.sbatch`](scripts/serve_vllm.sbatch)
> for the SLURM serving recipe — validated end-to-end on node4: the
> **temperature-augmented, retrained** `investforge-analyst` adapter serves the
> analyst node and `/analyze` returns a full BUY/HOLD/SELL recommendation
> (rating · confidence · rationale · risk_factors · target_price).

---

## Multimodal vision (report / chart images)

InvestForge can analyse research-report screenshots, K-line charts, and
financial diagrams by routing them through a dedicated vision VLM
(`Qwen/Qwen2.5-VL-7B-Instruct`) served at a separate vLLM endpoint.

**The feature is INERT by default** — set `LOCAL_VISION_MODEL` in `.env` to
activate it.  When unset, the pipeline behaves exactly as before.

```bash
# POST a base64-encoded PNG alongside the ticker:
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"ts_code":"688981.SH","images":[{"kind":"base64","value":"<B64>"}]}'
```

Images are validated by a SSRF-hardened security layer before any processing:
https-only URLs, IP-pinned fetching (no DNS rebinding), decompression-bomb
guards, and a format allowlist (PNG / JPEG / WEBP).  Local file paths are
**disabled by default** (`VISION_ALLOW_LOCAL_PATHS=false`).

See [`docs/VISION_RUNBOOK.md`](docs/VISION_RUNBOOK.md) for the full setup,
boot commands, and security model.

---

## Environment variables (`.env.example`)

| Key                              | Purpose                                            |
|----------------------------------|----------------------------------------------------|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | Pick one for the LLM layer |
| `LOCAL_LLM_BASE_URL` / `LOCAL_LLM_MODEL` | Use a self-hosted vLLM endpoint (`Qwen/Qwen2.5-7B-Instruct`) |
| `LOCAL_ANALYST_MODEL`            | LoRA adapter name for the analyst node (`investforge-analyst`); other nodes use the base model |
| `LOCAL_VISION_BASE_URL` / `LOCAL_VISION_MODEL` | Vision VLM endpoint + model (`Qwen/Qwen2.5-VL-7B-Instruct`); leave blank to disable vision |
| `VISION_ALLOW_LOCAL_PATHS`       | `false` (safe default) — set `true` only in trusted environments |
| `VISION_MAX_IMAGES` / `VISION_MAX_BYTES` / `VISION_MAX_PIXELS` | Image input policy |
| `LORA_ADAPTERS_DIR` / `LORA_ADAPTER_PATH` | Host adapter dir + in-container adapter path for docker-compose |
| `LLM_PROVIDER`                   | `fake` (default) / `openai` / `anthropic` / `local` |
| `TEACHER_LLM_PROVIDER` / `TEACHER_OPENAI_API_KEY` | Teacher model for SFT dataset distillation |
| `TUSHARE_TOKEN`                  | Real fundamental + news data on the server         |
| `QDRANT_URL` / `QDRANT_COLLECTION` | Production knowledge-base location              |
| `EMBEDDING_MODEL` / `RERANKER_MODEL` | bge-small-zh + bge-reranker-v2-m3              |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` | LangSmith observability             |

`.env` is git-ignored; only `.env.example` is checked in.

---

## Tests

226 pytest unit tests, fully network-free, cover:

* domain enums + lenient rating parsing
* config loader edge cases (missing env, bad int)
* lexicon sentiment scorer
* fake LLM dispatch + handler-priority ordering · in-process HF provider wiring
* hybrid retriever (BM25, HyDE, reranker), recursive chunking
* every agent node (researcher / analyst / risk-control / vision / output)
* end-to-end pipeline including conditional revision loop + iteration cap
* lookahead-bias-safe signal + simple / cross-sectional backtest + 5-ticker panel loader / MultiIndex factor builder (alphalens-ready)
* heuristic quality scorer + RAGAS (stub + real-judge runner, eval-set loader, faithfulness gate)
* SFT dataset distillation + train/serve prompt parity
* temperature-augmented distillation (sample ladder, full-identity dedup, prompt-parity)
* LangGraph runtime end-to-end + graph/inline parity (runs on server; skipped when langgraph absent)
* SSRF-hardened image-input validator (IP pinning, path traversal, bombs)
* multimodal vision wiring (ChatMessage images, vision node gating)

```bash
pytest -q
```

---

## Roadmap (4-week sprint)

| Week | Deliverable                                                          | Status |
|------|----------------------------------------------------------------------|--------|
| W1   | Fundamental analysis report on a single A-share name                 | ⏳ on user |
| W2   | LoRA distill of Qwen2.5-7B analyst + vLLM adapter serving + routing | ✅ **shipped live** (trained → eval → vLLM-served → `/analyze` end-to-end on SLURM) → measured non-ceiling delta on a 200-example temperature-augmented distill set: rating-accuracy 0.867→0.900, confidence-MAE 0.0283→0.0200 (30-ex val); see [docs/W2_LORA_RUNBOOK.md](docs/W2_LORA_RUNBOOK.md) |
| W3   | Hybrid RAG + RAGAS Faithfulness ≥ 0.8                                | ✅ **measured** on node4 (Qwen2.5-7B judge, 6-row eval set): **Faithfulness 1.00, context_precision 1.00, context_recall 1.00** → gate PASS; real RAGAS runner + judge wiring committed |
| W4   | 5-ticker end-to-end + alphalens backtest dashboard                   | ✅ **measured** on the sample (2024, 261 trading days, 5 tickers): IC 0.0172 / IC-IR 0.0343 / annualised return +13.29% / Sharpe 0.97 / MDD −8.01%; alphalens tear-sheet wired (`scripts/run_backtest.py --alphalens`) + dashboard expander shipped |

---

## License

MIT.  © 2026 Weijia Han.
