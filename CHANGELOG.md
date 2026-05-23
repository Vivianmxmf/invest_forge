# Changelog

All version-level changes to InvestForge are documented here.

---

## [0.2.0] - 2026-05-23

### Features

- **Analyst LoRA distillation pipeline (Week-2 deliverable):** SFT dataset
  builder (`scripts/build_sft_dataset.py`) uses a teacher LLM to generate
  analyst reasoning traces; `finetune/train_lora.py` fine-tunes
  `Qwen/Qwen2.5-7B-Instruct` with QLoRA (rank 16) on the distilled data and
  writes the adapter to `models/adapters/investforge-analyst/`.

- **vLLM LoRA adapter serving:** `docker-compose.yml` bumped to
  `vllm/vllm-openai:v0.6.3` with `--enable-lora`, `--max-lora-rank 16`, and
  `--lora-modules investforge-analyst=...`.  A new volume mount exposes the
  host adapter directory (`LORA_ADAPTERS_DIR`) inside the container at
  `/adapters`.  Default base model updated to `Qwen/Qwen2.5-7B-Instruct`.

- **Per-node analyst routing:** `LLMConfig` gained a `local_analyst_model`
  field (env `LOCAL_ANALYST_MODEL`).  `build_analyst_client()` in
  `invest_forge/llm/client.py` returns an `OpenAILLMClient` targeting the
  adapter name when `provider=local` and `local_analyst_model` is set;
  otherwise falls back to `build_client()` (no-op for `fake`/`openai`/
  `anthropic`).  `GraphDeps` gained an optional `analyst_client` field; when
  set, `make_analyst_node` receives the adapter-targeted client while
  researcher and risk-control keep the base-model client.  All existing
  behaviour is unchanged when `LOCAL_ANALYST_MODEL` is not configured.

- **cu121 server fix:** `scripts/server_migrate.sh` corrected to target
  node1.athena's actual hardware (8× A5000, Driver 535.154.05, CUDA 12.2).
  GPU stack now installs from `requirements-gpu.txt` with
  `--extra-index-url .../cu121`; hard `cu124` / `torch==2.4.*` pins removed.

- **W2 runbook:** `docs/W2_LORA_RUNBOOK.md` documents the complete
  server-side workflow: migrate → configure → build SFT data → train →
  evaluate → serve → verify.

### Design Rationale

- **Distill only the analyst node** (not researcher or risk-control): the
  analyst turn is the most token-heavy and stylistically distinct step, making
  it the best leverage point for a small domain-specific adapter.  Researcher
  and risk-control benefit less from style fine-tuning and more from RAG
  context, so they stay on the capable base model to minimise risk of
  capability regression.

- **cu121 forced by driver 535:** NVIDIA driver 535.x ships with a CUDA 12.2
  runtime.  PyTorch's `cu124` wheels link against `libcuda.so.12.4` which is
  absent; they will silently install but crash at first `import torch`.  Using
  cu121 wheels avoids this mismatch without requiring a driver upgrade.

- **`analyst_client` as an optional field on `GraphDeps`** (rather than a
  mandatory second client): keeps the constructor backward-compatible for all
  test fixtures and third-party callers that only pass `llm`.  The fallback
  `deps.analyst_client or deps.llm` pattern is a two-line change in both
  `build_invest_graph` and `run_pipeline_inline`.

### Notes & Caveats

- GPU steps (Steps 3–6 in the runbook) run on node1.athena only.  Running
  `train_lora.py` or the vLLM compose service on a CPU-only machine will fail.
- The vLLM container will exit immediately if the adapter directory
  (`models/adapters/investforge-analyst/`) does not exist or is empty when the
  container starts.  Train the adapter (Step 4) before bringing up vLLM.
- `data/sft/` is git-ignored — it is regenerable and may contain outputs
  derived from proprietary API calls.

---

## [0.1.0] - 2026-05-20

### Features

- Initial scaffold: LangGraph multi-agent pipeline (data_fetcher → researcher
  → analyst → risk_control → output) with conditional revision loop.
- Hybrid RAG: BM25 + bge-embeddings + optional reranker; Qdrant + in-memory
  backends.
- Backtest engine with lookahead-bias guard and cross-sectional normalisation.
- Heuristic quality scorer + RAGAS stub eval.
- Streamlit dashboard + FastAPI `/analyze` endpoint.
- 71 offline unit tests (no API keys, no GPU required).
- Codex review × 7; 17+ P1/P2 findings addressed.
