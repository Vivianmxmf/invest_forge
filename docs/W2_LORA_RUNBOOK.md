# Week-2 LoRA Runbook — InvestForge Analyst Adapter

End-to-end guide for training and serving the `investforge-analyst` LoRA adapter
on the project server, then wiring it into the InvestForge pipeline so only the
analyst node uses the fine-tuned adapter while researcher and risk-control keep
the base model.

---

## (0) Hardware & CUDA note

**Server:** node1.athena — 8× NVIDIA RTX A5000 24 GB, Driver 535.154.05.

Driver 535.x caps the supported CUDA runtime at **12.2**.  PyTorch `cu124`
wheels will fail to import.  All GPU packages must be installed with the
`--extra-index-url https://download.pytorch.org/whl/cu121` flag.  The
`requirements-gpu.txt` file (owned by the training agent) pins compatible
versions.

One A5000 (24 GB) is sufficient to:
- Fine-tune Qwen2.5-7B with QLoRA (fits in ~16 GB).
- Serve the fine-tuned base + adapter via vLLM with `--gpu-memory-utilization 0.85`.

The remaining 7 cards are free for parallel training runs.

---

## Step 1 — Run the migration script

```bash
# From the repo root on node1.athena
bash scripts/server_migrate.sh

# Then install the GPU stack (cu121 wheels):
pip install -r requirements-gpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

`server_migrate.sh` is idempotent — safe to re-run.  It:
- Creates / reuses the `invest_forge` conda env (Python 3.11).
- Installs `requirements.txt` (CPU deps).
- Scaffolds `.env` from `.env.example`.
- Brings up Qdrant via Docker.
- Runs `pytest -q` to confirm the offline suite is green.

---

## Step 2 — Configure `.env`

Edit `.env` (created by the migration script) and fill in at minimum:

```bash
# Teacher LLM for SFT dataset distillation
TEACHER_LLM_PROVIDER=openai
TEACHER_OPENAI_API_KEY=sk-...

# Local vLLM endpoint (used after Step 6)
LLM_PROVIDER=local
LOCAL_LLM_BASE_URL=http://localhost:8000/v1
LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LOCAL_ANALYST_MODEL=investforge-analyst

# LoRA adapter paths (for docker-compose volume mount)
LORA_ADAPTERS_DIR=./models/adapters
LORA_ADAPTER_PATH=/adapters/investforge-analyst

# Optionally: real market data
TUSHARE_TOKEN=...
```

---

## Step 3 — Build the SFT dataset

The distillation script calls the teacher LLM to generate analyst-style
reasoning traces for a set of tickers, writing JSONL records to `data/sft/`.

```bash
python scripts/build_sft_dataset.py \
    --tickers 600519.SH,688981.SH,000001.SZ,601318.SH,600036.SH \
    --out-dir data/sft \
    --val-frac 0.15 \
    --with-revisions
```

`--tickers` is a comma-separated `ts_code` list (or a path to a file of
tickers). `--with-revisions` adds a second example per ticker carrying
synthetic risk-control feedback, teaching the analyst the revision path.
The teacher is chosen by `--teacher-provider` (defaults to
`TEACHER_LLM_PROVIDER` → `LLM_PROVIDER`).

Output: `data/sft/train.jsonl` and `data/sft/val.jsonl`, chat-format records
`{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` where the
user turn is byte-identical to what the analyst node sends at inference.

`data/sft/` is git-ignored — it is regenerable and may contain API-key-derived
content.

---

## Step 4 — Fine-tune the analyst LoRA

Training uses a single A5000 (≈16 GB peak VRAM).  The other 7 cards remain
free.

```bash
python finetune/train_lora.py --config configs/lora.yaml
```

Key config values (see `configs/lora.yaml`):
- `base_model: Qwen/Qwen2.5-7B-Instruct`
- `adapter_output_dir: models/adapters/investforge-analyst`
- `lora_rank: 16`
- `lora_target_modules: [q_proj, v_proj]`

Training writes the adapter to `models/adapters/investforge-analyst/`
(`adapter_config.json` + `adapter_model.safetensors`).

---

## Step 5 — Evaluate the adapter

```bash
python finetune/eval_lora.py --config configs/lora.yaml
```

Reports ROUGE / BERTScore against the eval split and a heuristic quality
score consistent with the online eval pipeline.  Target: analyst quality
score ≥ 0.7 on the eval set before promoting to production.

---

## Step 6 — Serve base model + adapter via vLLM

The `docker-compose.yml` `vllm` service mounts the adapter directory and
passes `--enable-lora --lora-modules investforge-analyst=...` to vLLM.

**The adapter files must exist before starting the service** (Step 4 must
have completed successfully).

```bash
docker compose --profile vllm up -d
```

This starts:
- **Qdrant** on `:6333` (vector store).
- **vLLM** on `:8000` — serving `Qwen/Qwen2.5-7B-Instruct` as the base model
  plus the `investforge-analyst` LoRA adapter.

Verify that both the base model and the adapter are visible:

```bash
curl http://localhost:8000/v1/models | python -m json.tool
# Expected output includes entries for both:
#   "id": "Qwen/Qwen2.5-7B-Instruct"
#   "id": "investforge-analyst"
```

---

## Step 7 — Start the app and run an analysis

With `.env` configured (Step 2) and vLLM running (Step 6):

```bash
make api          # FastAPI on :8001
make dashboard    # Streamlit on :8501
```

Smoke-test the API:

```bash
curl -s -X POST http://localhost:8001/analyze \
     -H 'Content-Type: application/json' \
     -d '{"ts_code": "688981.SH"}' | python -m json.tool
```

Expected: a `final_recommendation` with `rating`, `confidence`, `rationale`,
`risk_factors`, and `risk_level`.

### How routing works

| Node          | Client used                                        |
|---------------|----------------------------------------------------|
| researcher    | base model (`Qwen/Qwen2.5-7B-Instruct`)           |
| analyst       | LoRA adapter (`investforge-analyst`) via vLLM      |
| risk_control  | base model (`Qwen/Qwen2.5-7B-Instruct`)           |

When `LOCAL_ANALYST_MODEL` is unset or the provider is not `local`, all nodes
share the default client (backward-compatible with `fake` / `openai` /
`anthropic` providers).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `vLLM` container exits immediately | Adapter dir missing or empty | Run Step 4 first |
| `torch` import error: CUDA mismatch | `cu124` wheels installed | Reinstall with `cu121` index URL |
| `investforge-analyst` not in `/v1/models` | Container started before adapter was trained | Stop, train, restart |
| API returns 500 on `/analyze` | `LOCAL_LLM_BASE_URL` not set in `.env` | Check `.env` and restart API |
