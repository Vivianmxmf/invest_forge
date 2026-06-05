# InvestForge dev Makefile

PY      := python3
VENV    := .venv
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
BLACK   := $(VENV)/bin/black
RUFF    := $(VENV)/bin/ruff
MYPY    := $(VENV)/bin/mypy
SRC     := janus_terminal tests frontend scripts

.PHONY: help venv install dev-install fmt lint type test cov sample \
        qdrant-up qdrant-down build-kb dashboard api clean

help:
	@echo "InvestForge targets:"
	@echo "  make venv          create .venv"
	@echo "  make install       install runtime deps"
	@echo "  make dev-install   editable install + dev tools"
	@echo "  make fmt           black + ruff --fix"
	@echo "  make lint          ruff + black --check"
	@echo "  make type          mypy"
	@echo "  make test          pytest unit tests"
	@echo "  make cov           pytest with coverage"
	@echo "  make sample        regenerate synthetic sample dataset"
	@echo "  make qdrant-up     start Qdrant container"
	@echo "  make build-kb      build knowledge base from data/pdfs/"
	@echo "  make dashboard     run Streamlit locally"
	@echo "  make api           run FastAPI locally"
	@echo "  make clean         remove caches"

venv:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip setuptools wheel

install: venv
	$(PIP) install -r requirements.txt

dev-install: venv
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

fmt:
	$(BLACK) $(SRC)
	$(RUFF) check $(SRC) --fix

lint:
	$(RUFF) check $(SRC)
	$(BLACK) --check $(SRC)

type:
	$(MYPY) janus_terminal

test:
	$(PYTEST) -m "unit or not integration"

cov:
	$(PYTEST) --cov=janus_terminal --cov-report=term-missing --cov-report=html

sample:
	$(PY) scripts/generate_sample_data.py

qdrant-up:
	docker compose --profile rag up -d qdrant

qdrant-down:
	docker compose --profile rag down

build-kb:
	$(VENV)/bin/python scripts/build_kb.py

dashboard:
	$(VENV)/bin/streamlit run frontend/app.py

api:
	$(VENV)/bin/uvicorn janus_terminal.api.main:app --host 0.0.0.0 --port 8001

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
