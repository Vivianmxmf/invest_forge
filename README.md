# InvestForge — AI 主观投研助手

> LangGraph Multi-Agent · Hybrid RAG · LoRA-ready · LangSmith observable

InvestForge is the capstone project of the *AI 主观投研* sprint
([`../JD6_AI主观投研实习生_AI_Agent方向.md`](../JD6_AI主观投研实习生_AI_Agent方向.md)).
Input a stock ticker → the system pulls fundamentals + macro + news, runs a
researcher → analyst → risk-control multi-agent pipeline with conditional
revision loops, and outputs a structured investment recommendation
(BUY / HOLD / SELL + confidence + logic + risk factors).

---

## Architecture

```
                ┌───────────────────────────────┐
                │   Data layer                  │
   ts_code  ──▶ │   Tushare · AKShare · news    │  ──▶ FinBERT / lexicon sentiment
                └───────────┬───────────────────┘
                            ▼
                ┌───────────────────────────────┐
                │   Hybrid RAG                  │   BM25 + bge-embeddings + reranker
                │   (Qdrant / in-memory)        │   ± HyDE query expansion
                └───────────┬───────────────────┘
                            ▼
       ┌──────────────────────────────────────────────────────┐
       │ LangGraph state machine                              │
       │   data_fetcher → researcher → analyst → risk_control │
       │                                  ▲          │       │
       │                                  └── revise ┘       │
       │   risk_control:HIGH ⇒ analyst again (≤ max_iter)     │
       │                                                      │
       │   risk_control:LOW/MED ⇒ output                      │
       └────────────────────────┬─────────────────────────────┘
                                 ▼
                ┌────────────────────────────┐
                │ Recommendation             │ rating · confidence · rationale ·
                │ (BUY / HOLD / SELL)        │ risk_factors · target_price
                └────────────────────────────┘
```

Every external dependency (LLM, vector store, data provider, sentiment
model) is hidden behind a small Protocol so the *entire* pipeline runs
offline against deterministic fakes — see ``tests/`` and
``invest_forge/llm/fake.py``.

---

## Layout

```
invest_forge/
├── invest_forge/
│   ├── agents/                    # state · nodes · graph · prompts
│   ├── common/                    # config · logging · types
│   ├── eval/                      # ragas_eval · quality
│   ├── knowledge_base/            # builder · retriever · pdf_loader
│   ├── llm/                       # client protocol · fake · openai · anthropic
│   └── tools/                     # data · sentiment · rag · backtest
├── frontend/                      # Streamlit dashboard
├── api/                           # (inside invest_forge/api/) FastAPI
├── notebooks/                     # Week 1 / 3 / 4 walkthroughs
├── scripts/                       # generate_sample_data · build_kb · setup · server_migrate
├── data/sample/                   # synthetic 5-stock dataset (141 KB) — committed
├── tests/                         # 64 unit tests, network-free
├── docker-compose.yml             # qdrant + (optional) vllm
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

# Optional vLLM (self-hosted Qwen2.5-7B + investforge-analyst LoRA adapter):
docker compose --profile vllm up -d
```

> **No model weights are downloaded by ``server_migrate.sh``.**
> Pull them manually after editing ``.env``:
> ```bash
> huggingface-cli login
> huggingface-cli download Qwen/Qwen2.5-7B-Instruct
> ```
>
> See [`docs/W2_LORA_RUNBOOK.md`](docs/W2_LORA_RUNBOOK.md) for the full
> LoRA fine-tuning + adapter serving workflow.

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

64 pytest unit tests cover:

* domain enums + lenient rating parsing
* config loader edge cases (missing env, bad int)
* lexicon sentiment scorer
* fake LLM dispatch + handler-priority ordering
* hybrid retriever (BM25, HyDE, reranker), recursive chunking
* every agent node (researcher / analyst / risk-control / output)
* end-to-end pipeline including conditional revision loop + iteration cap
* lookahead-bias-safe signal + simple / cross-sectional backtest
* heuristic quality scorer + RAGAS stub

```bash
pytest -q
```

---

## Roadmap (4-week sprint)

| Week | Deliverable                                                          | Status |
|------|----------------------------------------------------------------------|--------|
| W1   | Fundamental analysis report on a single A-share name                 | ⏳ on user |
| W2   | LoRA distill of Qwen2.5-7B analyst + vLLM adapter serving + routing | ✅ see [docs/W2_LORA_RUNBOOK.md](docs/W2_LORA_RUNBOOK.md) |
| W3   | Hybrid RAG + RAGAS Faithfulness ≥ 0.8                                | ✅ stub eval ready |
| W4   | 5-ticker end-to-end + alphalens backtest dashboard                   | ✅ skeleton ready |

---

## License

MIT.  © 2026 Weijia Han.
