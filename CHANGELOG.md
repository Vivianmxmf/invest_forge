# Changelog

All version-level changes to InvestForge are documented here.

---

## [0.4.1] - 2026-05-27

### Features

- **W4 — alphalens tear-sheet + 5-ticker backtest end-to-end:** `load_price_panel`
  (long CSV → wide DatetimeIndex × ticker close) and `build_factor_panel`
  (lookahead-safe MultiIndex `(date, asset)` momentum factor) feed both the
  existing pandas `portfolio_long_short` and a rewired `alphalens_report`
  (data dict: per-period mean IC, mean return by quantile, n_obs; lazy
  alphalens + matplotlib-Agg; PNG tear-sheet save in try/except so a tiny-
  universe failure doesn't crash). New `scripts/run_backtest.py` (laptop
  pandas path always; `--alphalens` opt-in, server). Streamlit dashboard
  gains a gated "📈 5-股票回测" expander with equity curve.

### Design Rationale

- 5-asset universe makes the alphalens default `quantiles=5` degenerate
  (duplicate bin edges); pinned default to 2 and exposed via flag.
- Pure-pandas path keeps the dashboard + default runner laptop-runnable;
  alphalens stays a server-only opt-in to preserve laptop import-safety.
- Factor at t uses prices through t−1 (signal is `price/MA` shifted by 1) and
  predicts forward returns from t → conservative; no same-day leak.

### Notes & Caveats

- **Measured on the sample** (2024-01-02 → 2024-12-31, 5 A-share tickers):
  IC 0.0172, IC-IR 0.0343, annualised return +13.29%, Sharpe 0.97, MDD −8.01%
  (261 obs). Small, synthetic universe — these are smoke numbers, not alpha.
- alphalens tear-sheet (`--alphalens --out-png`) only on server.

---

## [0.4.0] - 2026-05-27

### Features

- **LangGraph runtime runs live + API routes through it:** `build_invest_graph`
  is now executable via `run_pipeline_graph`; the FastAPI `/analyze` compiles
  and invokes the real compiled graph when `USE_LANGGRAPH=true` (default false
  keeps the in-process inline runner). Enables LangSmith auto-tracing of the
  agent graph (`LANGCHAIN_TRACING_V2`). `scripts/smoke_langgraph.py` proves the
  runtime end-to-end offline.
- **Real RAGAS evaluation (W3):** `evaluate_ragas` gains judge LLM/embeddings
  wiring (`build_ragas_judge`, OpenAI or local vLLM via base_url) and metric
  selection (default = the 3 LLM-only metrics; `answer_relevancy` opt-in).
  `scripts/run_ragas_eval.py` loads a committed 6-row eval set and gates on
  Faithfulness ≥ 0.8.

### Design Rationale

- LangGraph runtime is opt-in so the proven offline inline path (and all tests)
  stay byte-identical; the graph path is what makes LangSmith tracing capture.
- RAGAS results are reduced NaN-safely (ragas returns per-row score lists);
  metric selection avoids embeddings calls a local vLLM chat server can't serve.

### Notes & Caveats

- **Measured on node4** (Qwen2.5-7B judge, vLLM :8000, 6-row grounded eval set):
  **Faithfulness 1.00, context_precision 1.00, context_recall 1.00 → W3 gate PASS.**
- The eval set is a small synthetic fixture; its reference answers were tightened
  to be strictly context-grounded (an earlier run scored faithfulness 0.61 because
  answers carried ungrounded editorial claims — RAGAS correctly flagged it).
- Faithfulness is judge-dependent; a GPT-4-class judge is the stronger lever for
  larger/real eval sets. `answer_relevancy` needs an embeddings endpoint
  (`RAGAS_EMBEDDINGS_BASE_URL`).

---

## [0.3.1] - 2026-05-27

### Features

- **Temperature-ladder augmentation for the SFT distill set:**
  `scripts/build_sft_dataset.py` gains `--samples-per-ticker N` plus
  `--min-temp` / `--max-temp`, drawing N teacher samples per prompt across an
  evenly-spaced temperature ladder (validated `0 ≤ min ≤ max ≤ 2.0`). `N=1`
  is byte-identical to the prior single-sample behavior.

- **Full-identity dedup:** Generated examples are deduplicated on the full
  `(user, assistant)` identity rather than assistant content alone, so a
  baseline example and its revision counterpart are never collapsed even when
  the teacher returns identical analyst JSON for both.

- **+13 tests (197 → 210):** temperature-ladder bounds, full-identity dedup,
  prompt-parity recording, and invalid-argument validation.

### Design Rationale

- **Why scale up:** the original ~10-example POC gave a delta of 0 (a ceiling
  artifact — only 2 val examples, no resolution). Augmenting to 200 examples
  yields enough val samples (30) to measure a real signal.

- **Memo built once per ticker:** only the analyst *sampling* temperature
  varies between samples; the research memo and thus the user prompt stay
  byte-identical, preserving train/serve prompt parity.

- **Dedup on full identity:** keying on `(user, assistant)` ensures
  baseline ≠ revision examples both survive even with a same-answer
  (deterministic) teacher, since their user prompts differ.

### Notes & Caveats

- **Measured delta** (200-example set, 30-example val):
  `rating_accuracy` 0.8667 → 0.9000 (+3.3pp), `confidence_mae`
  0.0283 → 0.0200 (−0.0083); `json_validity_rate` and `risk_factor_coverage`
  both sit at the 1.0 ceiling. The 20-sample run produced 200 examples
  (170 train / 30 val), 40 unique per ticker, in ~23 min on a single A5000.

- **Self-distillation bounds the gain:** the live run used
  `teacher == student == Qwen2.5-7B`, so the improvement is format/schema
  fidelity, not new alpha. Val labels are teacher-generated (fidelity to the
  teacher, not ground truth); a stronger teacher (`gpt-4o` / `claude`) gives
  real headroom on `rating_accuracy`.

- **Fake teacher = zero diversity:** the fake/deterministic teacher produces
  identical samples that all dedup to 1 — augmentation requires a real teacher
  at `temp > 0`.

---

## [0.3.0] - 2026-05-24

### Features

- **Multimodal vision via uploaded report images:** The `/analyze` endpoint
  now accepts an optional `images` list (`kind: "url" | "base64" | "path"`,
  `value`, `mime`).  Supplied images are validated by the SSRF-hardened
  `image_input.py` security layer and forwarded to a dedicated vision VLM
  for analysis before the researcher node runs.

- **SSRF-hardened image validator:** `invest_forge/tools/image_input.py`
  provides `load_images(refs, policy) -> list[ValidatedImage]` with full
  threat mitigation: https-only scheme allowlist, IP-pinned fetching (no
  DNS-rebinding TOCTOU window), decompression-bomb guards (`max_bytes` before
  decode, `max_pixels` before raster), format allowlist (PNG/JPEG/WEBP only),
  and local-path containment (`VISION_ALLOW_LOCAL_PATHS=false` by default).

- **Vision vLLM serving:** New `vllm-vision` service in `docker-compose.yml`
  (profiles: `vllm`, `all`) runs `Qwen/Qwen2.5-VL-7B-Instruct` at host port
  8002 (`VLLM_VISION_PORT`), with `--limit-mm-per-prompt image=4` and
  `--max-model-len 8192`.  Text vLLM (:8000) and vision vLLM (:8002) each
  occupy one of the 4× A5000 cards.

- **Gated vision node:** `make_vision_node(client)` in `nodes.py` is a
  complete no-op when `client is None` or when `state["input_images"]` is
  empty, so the graph is unchanged for all non-vision requests.  Vision
  analysis is appended to the researcher prompt only when non-empty.

- **`ChatMessage.images` tuple:** `ChatMessage` gains an `images: tuple[str, ...]`
  field (default `()`) carrying data-URL strings for the vision payload.
  Existing code constructing `ChatMessage` without `images` is unchanged.

- **`build_vision_client`:** New factory in `llm/client.py`; returns an
  `OpenAILLMClient` targeting the vision endpoint only when
  `provider=local` and `local_vision_model` is set; otherwise `None`.

### Design Rationale

- **Vision isolated to one node + a dedicated VLM client:** The vision
  analysis step is cleanly separated from the text pipeline.  The vision
  client is a distinct `OpenAILLMClient` pointing at the separate vLLM
  endpoint so model routing, GPU allocation, and failure modes are
  independent from the text LLM.

- **`ChatMessage` extended backward-compatibly:** The `images` field has
  `default=()` so every existing `ChatMessage(role=..., content=...)` call
  compiles and behaves identically.  `_to_openai_messages` in
  `openai_client.py` emits a plain string `content` when `images` is empty —
  byte-identical to the pre-vision wire format.

- **Security-first input validation:** All image input flows through
  `image_input.py` before any data reaches the VLM.  The `load_images` API
  is the single choke point; error messages are generic to prevent
  information leakage.

- **Graph topology preserved for non-vision runs:** The vision node is only
  wired into `build_invest_graph` / `run_pipeline_inline` when
  `deps.vision_client is not None`.  All existing graph tests pass unchanged
  because their `GraphDeps` fixtures do not supply a `vision_client`.

### Notes & Caveats

- The vision VLM (`Qwen/Qwen2.5-VL-7B-Instruct`) runs server-side only.
  Laptop/offline mode (`LLM_PROVIDER=fake`) uses a canned response for
  the `[视觉] [Vision]` prompt marker so the full pipeline can be exercised
  without a GPU or API key.
- `VISION_ALLOW_LOCAL_PATHS=false` is the safe default.  Enable only in a
  trusted, isolated environment with a strictly controlled upload directory.
- The vision feature is completely inert when `LOCAL_VISION_MODEL` is unset:
  no extra latency, no graph topology change, no new error paths.

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
