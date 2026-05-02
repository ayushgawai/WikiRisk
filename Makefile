# ─────────────────────────────────────────────────────────────────────────────
# WikiRisk – Makefile
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

.DEFAULT_GOAL := help
SHELL         := /bin/bash
PYTHON        := python3
PIP           := pip3

DATA_DIR      := data
RAW_DIR       := $(DATA_DIR)/raw/dumps
PROCESSED_DIR := $(DATA_DIR)/processed

.PHONY: help install setup up down logs clean train stream serve ui test \
        batch full-pipeline lint format check-data

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  WikiRisk – Real-Time Wikipedia Edit Risk System"
	@echo ""
	@echo "  Commands:"
	@echo "    make install        Install Python dependencies"
	@echo "    make setup          First-time setup (install + create dirs)"
	@echo "    make up             Start all Docker services"
	@echo "    make down           Stop all Docker services"
	@echo "    make logs           Tail Docker logs"
	@echo ""
	@echo "  Pipeline:"
	@echo "    make batch          Run batch pipeline (download → parse → feature eng)"
	@echo "    make train          Train SparkML model + log to MLflow"
	@echo "    make stream         Start Spark Structured Streaming processor"
	@echo "    make collect        Start Wikimedia SSE stream collector"
	@echo "    make full-pipeline  Run batch + train in sequence"
	@echo ""
	@echo "  Dev:"
	@echo "    make serve          Start FastAPI backend (dev mode)"
	@echo "    make ui             Start Streamlit dashboard"
	@echo "    make test           Run test suite"
	@echo "    make lint           Run ruff linter"
	@echo "    make format         Run ruff formatter"
	@echo "    make check-data     Show data directory sizes"
	@echo "    make clean          Remove generated data and artifacts"
	@echo ""

# ── Install ───────────────────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

setup: install
	@mkdir -p $(DATA_DIR)/{raw/dumps,processed,stream/{incoming,checkpoints},predictions}
	@mkdir -p manifests
	@cp -n .env.example .env 2>/dev/null || true
	@echo "Setup complete. Edit .env with your API keys."

# ── Docker ────────────────────────────────────────────────────────────────────
up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

restart: down up

# ── Pipeline ──────────────────────────────────────────────────────────────────
batch:
	$(PYTHON) -m src.batch.pipeline --skip-download

batch-download:
	$(PYTHON) -m src.batch.pipeline

train:
	$(PYTHON) -m src.ml.trainer

collect:
	$(PYTHON) -m src.streaming.collector

stream:
	$(PYTHON) -m src.streaming.processor

full-pipeline: batch train
	@echo "Full pipeline complete. Model ready for streaming."

# ── Dev servers ───────────────────────────────────────────────────────────────
serve:
	uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

ui:
	streamlit run src/ui/app.py \
		--server.port 8501 \
		--server.address 0.0.0.0 \
		--theme.base light

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v --tb=short -m "not slow"

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

# ── Utilities ─────────────────────────────────────────────────────────────────
check-data:
	@echo "Data directory sizes:"
	@du -sh $(DATA_DIR)/* 2>/dev/null || echo "  (no data yet)"
	@echo ""
	@echo "Manifests:"
	@cat manifests/dataset_manifest.json 2>/dev/null | python3 -m json.tool || echo "  (no manifest yet)"

clean:
	rm -rf $(DATA_DIR)/processed \
	       $(DATA_DIR)/stream \
	       $(DATA_DIR)/predictions \
	       $(DATA_DIR)/wikirisk.db
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean done (raw dumps preserved)."

clean-all: clean
	rm -rf $(DATA_DIR)/raw mlruns/
	@echo "Full clean done."
