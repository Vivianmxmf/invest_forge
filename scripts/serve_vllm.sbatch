#!/usr/bin/env bash
#SBATCH --job-name=if-vllm
#SBATCH --partition=athena-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=vllm_%j.log
# Optional — uncomment if your partition requires explicit GPU requests:
##SBATCH --gres=gpu:1
# Optional — pin a node:  add `-w node4` on the sbatch command line.
#
# Serve Qwen2.5-7B-Instruct + the investforge-analyst LoRA adapter via vLLM on a
# SLURM GPU node, then self-smoke the adapter so proof lands in the job log.
#
# Submit (persistent, recommended):
#     sbatch scripts/serve_vllm.sbatch
#     tail -f vllm_<jobid>.log            # watch startup + smoke output
#     srun --jobid=<jobid> --overlap --pty bash   # get a shell on the same node
#         → then: bash scripts/analyze_client.sh 688981.SH
#     scancel <jobid>                     # stop the server
#
# Or run inside an interactive allocation (the #SBATCH lines are ignored):
#     srun -p athena-small -N1 -n1 --time=08:00:00 --pty /bin/bash
#     bash scripts/serve_vllm.sbatch
#
# NOTE: the LoRA adapter must already be trained (finetune/train_lora.py).
set -uo pipefail

# ---- config (override via env, e.g. `PORT=8002 sbatch ...`) --------------
VLLM_ENV="${VLLM_ENV:-vllm}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
ADAPTER_NAME="${ADAPTER_NAME:-investforge-analyst}"
ADAPTER_PATH="${ADAPTER_PATH:-$HOME/invest_forge/models/adapters/investforge-analyst}"
PORT="${PORT:-8000}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_LEN="${MAX_LEN:-4096}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$HOME/invest_forge/qwen2.5_chat_template.jinja}"
# JSON/guided-decoding backend. Default "outlines" pulls a broken pyairports
# dependency in this env; lm-format-enforcer is already installed and needs no
# outlines. The analyst/risk nodes use response_format=json, so this matters.
GUIDED_BACKEND="${GUIDED_BACKEND:-lm-format-enforcer}"

echo "================ InvestForge vLLM server ================"
echo "node     : $(hostname)"
echo "job      : ${SLURM_JOB_ID:-<interactive>}"
echo "gpus     : ${CUDA_VISIBLE_DEVICES:-<set by SLURM/partition>}"
echo "model    : $MODEL"
echo "adapter  : $ADAPTER_NAME -> $ADAPTER_PATH"
echo "port     : $PORT"
echo "connect  : srun --jobid=${SLURM_JOB_ID:-<id>} --overlap --pty bash   # then curl localhost:$PORT"
echo "========================================================"

# ---- conda --------------------------------------------------------------
# shellcheck disable=SC1091
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "$VLLM_ENV"

# ---- preconditions ------------------------------------------------------
if [[ ! -f "$ADAPTER_PATH/adapter_config.json" ]]; then
  echo "ERROR: adapter not found at $ADAPTER_PATH — run finetune/train_lora.py first." >&2
  exit 1
fi

# ---- chat template ------------------------------------------------------
# transformers 4.57 (training env) may save the adapter tokenizer's chat
# template in a separate chat_template.jinja that transformers <4.49 (this
# serving env) does not auto-load → vLLM raises "default chat template is no
# longer allowed". Pass the BASE model's template explicitly (also keeps
# train/serve parity). Extract it once if missing.
if [[ ! -f "$CHAT_TEMPLATE" ]]; then
  echo "extracting chat template from $MODEL → $CHAT_TEMPLATE ..."
  python - "$MODEL" "$CHAT_TEMPLATE" <<'PY'
import sys
from transformers import AutoTokenizer
model, out = sys.argv[1], sys.argv[2]
tmpl = AutoTokenizer.from_pretrained(model).chat_template
if not tmpl:
    raise SystemExit("ERROR: base tokenizer has no chat_template")
with open(out, "w") as f:
    f.write(tmpl)
print(f"wrote {out} ({len(tmpl)} chars)")
PY
fi

# ---- launch vLLM in the background --------------------------------------
# SLURM already exposes only the allocated GPU(s) via CUDA_VISIBLE_DEVICES.
vllm serve "$MODEL" \
  --enable-lora \
  --lora-modules "${ADAPTER_NAME}=${ADAPTER_PATH}" \
  --max-lora-rank "$MAX_LORA_RANK" \
  --chat-template "$CHAT_TEMPLATE" \
  --guided-decoding-backend "$GUIDED_BACKEND" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" &
VLLM_PID=$!

# Make sure the server dies with the job/script.
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT INT TERM

# ---- wait for readiness -------------------------------------------------
echo "waiting for vLLM on :$PORT (up to ~20 min) ..."
ready=0
for _ in $(seq 1 120); do
  if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    ready=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM exited during startup — see log above." >&2
    exit 1
  fi
  sleep 10
done
if [[ "$ready" -ne 1 ]]; then
  echo "ERROR: vLLM did not become ready in time." >&2
  exit 1
fi
echo "vLLM is READY on http://$(hostname):$PORT"

# ---- self-smoke (proof in the job log) ----------------------------------
echo "=== /v1/models ==="
curl -fsS "http://localhost:$PORT/v1/models" | python -m json.tool || true

echo "=== chat completion via adapter '$ADAPTER_NAME' ==="
curl -fsS "http://localhost:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${ADAPTER_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"给出一段A股投资观点JSON，字段: rating/confidence/rationale/risk_factors\"}],\"max_tokens\":256,\"temperature\":0.1}" \
  | python -m json.tool || true

echo "=== server stays up until walltime. stop with: scancel ${SLURM_JOB_ID:-<id>} ==="
wait "$VLLM_PID"
