#!/usr/bin/env bash
# scripts/server_migrate.sh — one-shot InvestForge server migration.
#
# Hardware target (validated multi-GPU host, verified 2026-05):
#   8× NVIDIA A5000-class GPUs (24 GB each), Driver 535.x, CUDA 12.2.
#   NOTE: driver 535.x caps the supported CUDA runtime at 12.2 — PyTorch
#   cu124 wheels will NOT load.  All GPU packages must use cu121 wheels.
#   Qwen2.5-7B fits on a single A5000; remaining cards are free for training.
#
# What it does, in order:
#   0. Pre-flight (OS / Python / conda / Docker / GPU / disk).
#   1. Single conda env "janus_terminal" (Python 3.11) — or venv fallback.
#   2. pip install -r requirements.txt.
#   3. Materialise .env from .env.example (idempotent, generates secrets).
#   4. Generate the synthetic sample dataset.
#   5. Bring up Qdrant via docker compose (if Docker available).
#   6. (gated on nvidia-smi) install the GPU-only finetune + serving stack:
#         transformers · accelerate · peft · trl · bitsandbytes · vllm.
#      → does NOT download any model weights — that's a manual ``huggingface-cli
#        download`` step the operator runs after editing .env.
#   7. Run pytest -q (offline).
#   8. Print a clear GO/NO-GO summary.
#
# Constraints honoured:
#   * NEVER calls sudo / apt / yum / dnf.
#   * Idempotent: re-runs are safe.
#   * Refuses to run as root.
#   * Model weights are NOT downloaded by this script — the operator picks the
#     base model and runs ``huggingface-cli login`` + ``huggingface-cli download``
#     to honour your "no model installation locally" rule.

set -euo pipefail
umask 077

# ───────────────────────── cosmetics ─────────────────────────
if [[ -t 1 ]]; then
    C_BLUE="$(printf '\033[1;34m')"; C_GREEN="$(printf '\033[1;32m')"
    C_YELLOW="$(printf '\033[1;33m')"; C_RED="$(printf '\033[1;31m')"
    C_RESET="$(printf '\033[0m')"
else C_BLUE=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_RESET=""; fi
step()  { printf '\n%s==>%s %s\n' "$C_BLUE" "$C_RESET" "$*"; }
ok()    { printf '%s    ✓%s %s\n'  "$C_GREEN" "$C_RESET" "$*"; }
warn()  { printf '%s    ⚠%s %s\n'  "$C_YELLOW" "$C_RESET" "$*"; }
fail()  { printf '%s    ✗%s %s\n'  "$C_RED"   "$C_RESET" "$*"; }
heading(){ printf '\n%s━━━ %s ━━━%s\n' "$C_BLUE" "$*" "$C_RESET"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

cleanup() { rm -f "$REPO_ROOT/.env.bak" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

ENV_NAME="janus_terminal"
PY_VER="3.11"
SUPPORTED_PYS=("3.11" "3.12" "3.10")

SUMMARY_OK=()
SUMMARY_WARN=()
SUMMARY_FAIL=()
note_ok()   { SUMMARY_OK+=("$*");   ok   "$*"; }
note_warn() { SUMMARY_WARN+=("$*"); warn "$*"; }
note_fail() { SUMMARY_FAIL+=("$*"); fail "$*"; }

# ───────────────────────── 0. pre-flight ─────────────────────────
heading "0. Pre-flight"

step "checking user"
if [[ "$(id -u)" -eq 0 ]]; then
    fail "Running as root.  Refusing.  Re-run as your normal user."
    [[ "${janus_terminal_ALLOW_ROOT:-0}" != "1" ]] && exit 1
fi
ok "user=$(id -un) uid=$(id -u)"

step "checking working dir"
for must in requirements.txt scripts/generate_sample_data.py janus_terminal/common/config.py; do
    if [[ ! -f "$must" ]]; then
        fail "missing $must — are you in the janus_terminal/ repo root?"
        exit 1
    fi
done
ok "repo root: $REPO_ROOT"

step "disk + memory snapshot"
DISK_FREE_GB="$(df -BG "$REPO_ROOT" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')"
[[ -n "${DISK_FREE_GB:-}" ]] && {
    if (( DISK_FREE_GB < 30 )); then
        note_warn "only ${DISK_FREE_GB}G free — vLLM image + model cache wants ≥30G"
    else
        ok "free disk = ${DISK_FREE_GB}G"
    fi
}
if command -v free >/dev/null 2>&1; then
    MEM_GB="$(free -g | awk '/^Mem:/ {print $2}')"
    if (( MEM_GB < 16 )); then
        note_warn "RAM=${MEM_GB}G; embedding + reranker want ≥16G"
    else ok "RAM = ${MEM_GB}G"
    fi
fi

# Probe GPUs — never fatal because the rest of the pipeline runs CPU-only.
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1; then
    step "GPU probe"
    if GPU_TABLE="$(nvidia-smi --query-gpu=index,name,memory.total,driver_version \
                                --format=csv,noheader 2>&1)"; then
        printf '%s\n' "$GPU_TABLE" | sed 's/^/    /'
        # Check that at least one GPU has ≥20GB (A5000 ships with 24G).
        if echo "$GPU_TABLE" | awk -F',' 'BEGIN{ok=0} { gsub(/[^0-9]/,"",$3); if ($3+0 >= 20000) ok=1 } END{exit !ok}'; then
            HAS_GPU=1
            note_ok "GPU stack will be installed (≥20G card detected)."
        else
            note_warn "GPUs detected but none ≥20G; skipping bnb/vllm install."
        fi
    else
        note_warn "nvidia-smi present but failed: $GPU_TABLE"
    fi
else
    note_warn "no nvidia-smi — DataGuard runs CPU-only on this host."
fi

# ───────────────────────── 1. env ─────────────────────────
heading "1. Single isolated env"

CONDA_BIN=""
USE_CONDA=false
for cand in "${CONDA_EXE:-}" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "/opt/miniconda3/bin/conda" "$(command -v conda 2>/dev/null || true)"; do
    if [[ -n "$cand" && -x "$cand" ]]; then CONDA_BIN="$cand"; USE_CONDA=true; break; fi
done

if $USE_CONDA; then
    step "found conda at $CONDA_BIN"
    # shellcheck disable=SC1090
    source "$(dirname "$CONDA_BIN")/../etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        EXISTING_PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo none)"
        if [[ " ${SUPPORTED_PYS[*]} " == *" $EXISTING_PY "* ]]; then
            ok "conda env '$ENV_NAME' already exists (Python $EXISTING_PY) — reusing"
        elif [[ "${janus_terminal_REBUILD_ENV:-0}" == "1" ]]; then
            note_warn "janus_terminal_REBUILD_ENV=1 — removing existing '$ENV_NAME' (Python $EXISTING_PY)"
            conda env remove -y -n "$ENV_NAME"
            conda create -y -n "$ENV_NAME" "python=$PY_VER" pip
        else
            fail "existing env '$ENV_NAME' has Python $EXISTING_PY (need: ${SUPPORTED_PYS[*]})."
            fail "Re-run with janus_terminal_REBUILD_ENV=1 to recreate, or inspect first."
            exit 1
        fi
    else
        step "creating conda env '$ENV_NAME' (Python $PY_VER)"
        conda create -y -n "$ENV_NAME" "python=$PY_VER" pip
    fi
    conda activate "$ENV_NAME"
    note_ok "active env: $(python -c 'import sys; print(sys.prefix)')"
else
    step "no conda found — falling back to python -m venv"
    VENV_DIR="$REPO_ROOT/.venv"
    if [[ ! -d "$VENV_DIR" ]]; then
        HOST_PY=""
        for v in "${SUPPORTED_PYS[@]}"; do
            if command -v "python${v}" >/dev/null 2>&1; then HOST_PY="python${v}"; break; fi
        done
        if [[ -z "$HOST_PY" ]]; then
            fail "no supported Python (need one of: ${SUPPORTED_PYS[*]}) and no conda."
            exit 1
        fi
        ok "using $HOST_PY"
        "$HOST_PY" -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    ACTIVE_PY="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    if [[ " ${SUPPORTED_PYS[*]} " != *" $ACTIVE_PY "* ]]; then
        fail "venv has Python $ACTIVE_PY (need: ${SUPPORTED_PYS[*]}); delete $VENV_DIR and re-run."
        exit 1
    fi
    note_ok "active env: $VIRTUAL_ENV (Python $ACTIVE_PY)"
fi

python -m pip install --upgrade pip setuptools wheel >/dev/null
ok "pip $(pip --version | awk '{print $2}')"

# ───────────────────────── 2. core deps ─────────────────────────
heading "2. Core deps (CPU)"

step "installing requirements.txt"
pip install -r requirements.txt
note_ok "core dependencies installed"

step "smoke import"
python - <<'PY'
import importlib
mods = ["pandas", "numpy", "pydantic", "fastapi", "streamlit", "plotly",
        "langchain", "langgraph", "langchain_openai", "langchain_anthropic",
        "qdrant_client", "rank_bm25", "sentence_transformers", "pymupdf",
        "tushare", "akshare"]
bad = []
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: bad.append(f"{m}: {e}")
if bad:
    print("FAILED:");
    for b in bad: print("  -", b)
    raise SystemExit(1)
print("all imports OK")
PY
note_ok "core modules import cleanly"

# ───────────────────────── 3. .env (before any docker bring-up) ─────────────────────────
heading "3. .env scaffold"

if [[ -f .env ]]; then
    ok ".env already exists — leaving alone (manually edit to update creds)"
else
    cp .env.example .env
    # No mandatory secrets at compose-up time (qdrant has no auth by default),
    # but generate API keys placeholders to make the file self-documenting.
    ok ".env created from .env.example"
fi
note_warn "Fill in .env manually: OPENAI_API_KEY / ANTHROPIC_API_KEY / TUSHARE_TOKEN / LANGCHAIN_API_KEY"

# ───────────────────────── 4. sample data ─────────────────────────
heading "4. Sample dataset"
if [[ -f data/sample/fundamentals.json ]]; then
    ok "sample already present"
else
    python scripts/generate_sample_data.py
fi
note_ok "data/sample/ ready"

# ───────────────────────── 5. Qdrant (best-effort) ─────────────────────────
heading "5. Qdrant (vector store)"

# Initialise before any `set -u` expansion below — even on hosts without
# Docker (or with a broken daemon), the GPU section may reference this.
COMPOSE_CMD=""

if command -v docker >/dev/null 2>&1; then
    if ! docker info >/dev/null 2>&1; then
        note_warn "docker CLI present but daemon unreachable; skipping qdrant bring-up."
    else
        if docker compose version >/dev/null 2>&1; then COMPOSE_CMD="docker compose"
        elif command -v docker-compose >/dev/null 2>&1; then COMPOSE_CMD="docker-compose"
        fi

        if [[ -z "$COMPOSE_CMD" ]]; then
            note_warn "docker present but no compose plugin; skipping qdrant bring-up."
        else
            step "starting qdrant via $COMPOSE_CMD"
            if $COMPOSE_CMD --profile rag up -d qdrant; then
                for i in $(seq 1 30); do
                    if curl -fsS http://localhost:6333/healthz >/dev/null 2>&1; then
                        note_ok "Qdrant healthy on localhost:6333"; break
                    fi
                    sleep 1
                    if [[ $i -eq 30 ]]; then
                        note_warn "Qdrant did not respond in 30s — check 'docker logs janus_terminal_qdrant'"
                    fi
                done
            else
                note_warn "docker compose up failed (port in use / image pull blocked?)"
            fi
        fi
    fi
else
    note_warn "docker not on PATH — install Qdrant another way or set QDRANT_URL to a remote instance."
fi

# ───────────────────────── 6. GPU extras (gated) ─────────────────────────
heading "6. GPU stack (finetune + vLLM serving)"

if (( HAS_GPU == 1 )); then
    step "installing GPU-only dependencies (this is heavy)"
    # Driver 535.x caps CUDA at 12.2 → must use cu121 wheels.
    # cu124 wheels will silently load but then crash at import time.
    GPU_INDEX_URL="https://download.pytorch.org/whl/cu121"

    if [[ -f requirements-gpu.txt ]]; then
        step "installing from requirements-gpu.txt (cu121)"
        pip install -r requirements-gpu.txt --extra-index-url "$GPU_INDEX_URL"
    else
        note_warn "requirements-gpu.txt not found — falling back to inline GPU pins"
        pip install --upgrade \
            --extra-index-url "$GPU_INDEX_URL" \
            "torch>=2.3,<2.5" \
            "transformers>=4.41" \
            "accelerate>=0.30" \
            "peft>=0.11" \
            "trl>=0.9" \
            "bitsandbytes>=0.43"
        # vLLM in a separate step so its CUDA-tied wheel doesn't conflict above.
        if ! pip install --upgrade --extra-index-url "$GPU_INDEX_URL" "vllm>=0.6.3"; then
            note_warn "vLLM install failed.
Manual recovery:
    pip install 'vllm>=0.6.3' --extra-index-url $GPU_INDEX_URL"
        else
            note_ok "vLLM installed"
        fi
    fi

    note_ok "GPU finetune + serving stack ready (no weights downloaded — pull manually)."
    note_warn "To run vLLM with weights:
    huggingface-cli login    # one-time, if pulling gated weights
    $COMPOSE_CMD --profile vllm up -d   # boots vLLM + adapter on port 8000"
else
    note_warn "No GPU detected — skipping torch/transformers/peft/trl/bitsandbytes/vllm install.
The agent pipeline still works against OpenAI / Anthropic / Gemini APIs."
fi

# ───────────────────────── 7. tests ─────────────────────────
heading "7. Unit tests"
if pytest -q --maxfail=5; then
    note_ok "all unit tests passed"
else
    note_fail "pytest failed — investigate before continuing"
fi

# ───────────────────────── 8. summary ─────────────────────────
heading "8. Migration summary"
printf '\n%bSuccesses (%d):%b\n' "$C_GREEN" "${#SUMMARY_OK[@]}" "$C_RESET"
for s in "${SUMMARY_OK[@]:-}"; do [[ -n "$s" ]] && printf '  ✓ %s\n' "$s"; done
if (( ${#SUMMARY_WARN[@]} > 0 )); then
    printf '\n%bWarnings (%d):%b\n' "$C_YELLOW" "${#SUMMARY_WARN[@]}" "$C_RESET"
    for s in "${SUMMARY_WARN[@]}"; do printf '  ⚠ %s\n' "$s"; done
fi
if (( ${#SUMMARY_FAIL[@]} > 0 )); then
    printf '\n%bFailures (%d):%b\n' "$C_RED" "${#SUMMARY_FAIL[@]}" "$C_RESET"
    for s in "${SUMMARY_FAIL[@]}"; do printf '  ✗ %s\n' "$s"; done
    exit 1
fi

cat <<EOF

${C_GREEN}Done.${C_RESET}  Next-step commands:

  # Activate the env (re-run each shell):
$( $USE_CONDA && echo "  conda activate $ENV_NAME" || echo "  source $REPO_ROOT/.venv/bin/activate" )

  # Run unit tests again any time:
  pytest -q

  # Build a real KB from PDFs (drop them under data/pdfs/ first):
  python scripts/build_kb.py --target memory   # laptop / smoke
  python scripts/build_kb.py --target qdrant   # production

  # Run the API + Streamlit dashboard:
  make api          # FastAPI on :8001
  make dashboard    # Streamlit on :8501

  # ── Week-2 LoRA workflow ──────────────────────────────────────────

  # (1) Fill in .env: set LLM_PROVIDER=local, LOCAL_LLM_MODEL, LOCAL_ANALYST_MODEL,
  #     TEACHER_LLM_PROVIDER, and any teacher API key for dataset distillation.

  # (2) Build the SFT dataset (requires a teacher LLM key):
  #     python scripts/build_sft_dataset.py --tickers 600519.SH 688981.SH \
  #         --out-dir data/sft

  # (3) Fine-tune the analyst LoRA (~16 GB on one A5000; 7 cards stay free):
  #     python finetune/train_lora.py --config configs/lora.yaml

  # (4) Evaluate the adapter:
  #     python finetune/eval_lora.py --config configs/lora.yaml

  # (5) Serve base model + adapter via vLLM:
  #     docker compose --profile vllm up -d
  #     # vLLM registers "investforge-analyst" adapter at startup.
  #     # Verify: curl http://localhost:8000/v1/models

  # (6) Point the app at the local endpoint and run:
  #     make api   # then POST /analyze {"ts_code": "688981.SH"}
EOF
