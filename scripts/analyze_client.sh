#!/usr/bin/env bash
# InvestForge — run the full multi-agent pipeline against the local vLLM server
# and print the recommendation. Run this on the SAME node as serve_vllm.sbatch:
#
#     srun --jobid=<server_jobid> --overlap --pty bash
#     bash scripts/analyze_client.sh 688981.SH
#
# It starts the FastAPI app (researcher + risk on the base model, analyst routed
# through the investforge-analyst LoRA adapter), waits for /health, POSTs
# /analyze, prints the JSON, then shuts the API down.
set -uo pipefail

TS="${1:-688981.SH}"
VLLM_PORT="${VLLM_PORT:-8000}"
API_PORT="${API_PORT:-8001}"
IF_ENV="${IF_ENV:-invest_forge}"

# shellcheck disable=SC1091
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "$IF_ENV"
cd "$HOME/invest_forge"

# Point the agent layer at the self-hosted vLLM endpoint.
export LLM_PROVIDER=local
export LOCAL_LLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export LOCAL_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export LOCAL_ANALYST_MODEL="investforge-analyst"
unset TUSHARE_TOKEN   # use synthetic sample data; set it (mind the 1/hr cap) for real data

# Confirm the vLLM server is reachable before booting the API.
if ! curl -fsS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: no vLLM server on localhost:${VLLM_PORT}. Start serve_vllm.sbatch on THIS node first." >&2
  exit 1
fi

echo "starting FastAPI on :${API_PORT} ..."
uvicorn invest_forge.api.main:app --host 0.0.0.0 --port "$API_PORT" >/tmp/if_api_${API_PORT}.log 2>&1 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 30); do
  curl -fsS "http://localhost:${API_PORT}/health" >/dev/null 2>&1 && break
  sleep 2
done

echo "=== POST /analyze {\"ts_code\":\"${TS}\"} ==="
curl -fsS -X POST "http://localhost:${API_PORT}/analyze" \
  -H 'Content-Type: application/json' \
  -d "{\"ts_code\":\"${TS}\"}" | python -m json.tool

echo "(API log: /tmp/if_api_${API_PORT}.log)"
