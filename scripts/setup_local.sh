#!/usr/bin/env bash
# setup_local.sh — quick bootstrap for the dev machine.
# Creates .venv, installs core deps, copies .env.example to .env (with generated
# secrets if applicable), generates the synthetic sample dataset, runs pytest.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
umask 077

SUPPORTED=("3.11" "3.12" "3.10")

# Pick a supported interpreter explicitly so we don't silently bootstrap a
# Python 3.13 venv against pyproject.toml's <3.13 constraint.
PY=""
if [[ -n "${PYTHON:-}" ]] && command -v "$PYTHON" >/dev/null 2>&1; then
    v="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    for s in "${SUPPORTED[@]}"; do
        [[ "$v" == "$s" ]] && PY="$PYTHON" && break
    done
fi
if [[ -z "$PY" ]]; then
    for s in "${SUPPORTED[@]}"; do
        if command -v "python${s}" >/dev/null 2>&1; then PY="python${s}"; break; fi
    done
fi
if [[ -z "$PY" ]]; then
    echo "error: no supported Python found (need one of: ${SUPPORTED[*]})." >&2
    echo "Install python3.11 in your \$HOME via pyenv / miniconda and re-run." >&2
    exit 1
fi
echo ">>> using $PY → $(command -v "$PY")"

if [[ ! -d .venv ]]; then
    echo ">>> creating virtual env"
    "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">>> upgrading pip"
pip install --upgrade pip setuptools wheel

echo ">>> installing runtime + dev deps"
pip install -r requirements.txt

if [[ ! -f .env ]]; then
    echo ">>> writing .env from .env.example"
    cp .env.example .env
fi

echo ">>> generating sample dataset"
python scripts/generate_sample_data.py

echo ">>> running unit tests"
pytest -q

echo
echo "Done. Activate with: source .venv/bin/activate"
