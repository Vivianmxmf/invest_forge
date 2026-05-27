# InvestForge — AI 主观投研助手

> LangGraph Multi-Agent · Hybrid RAG · QLoRA-distilled analyst (vLLM-served) · Multimodal vision · LangSmith observable

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
├── tests/                         # 221 unit tests, network-free
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

221 pytest unit tests, fully network-free, cover:

* domain enums + lenient rating parsing
* config loader edge cases (missing env, bad int)
* lexicon sentiment scorer
* fake LLM dispatch + handler-priority ordering · in-process HF provider wiring
* hybrid retriever (BM25, HyDE, reranker), recursive chunking
* every agent node (researcher / analyst / risk-control / vision / output)
* end-to-end pipeline including conditional revision loop + iteration cap
* lookahead-bias-safe signal + simple / cross-sectional backtest
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
| W4   | 5-ticker end-to-end + alphalens backtest dashboard                   | ✅ skeleton ready |

---

## License

MIT.  © 2026 Weijia Han.
