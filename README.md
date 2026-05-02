# WikiRisk — Real-Time Wikipedia Edit Risk & Knowledge Integrity System

> **DATA 228 (SP26) Group Project** · End-to-End Big Data Pipeline  
> SparkML · MLflow · Spark Structured Streaming · FastAPI · Streamlit · OpenAI

---

## Overview

WikiRisk is a production-grade big data system that ingests ≥ 10 GB of
English Wikipedia revision history, trains a distributed SparkML classifier
to score "edit risk" (probability that an edit is vandalism or will be reverted),
and deploys real-time scoring via Spark Structured Streaming connected to
[Wikimedia EventStreams](https://stream.wikimedia.org/v2/stream/recentchange).

A FastAPI backend exposes REST endpoints; a Streamlit dashboard lets users
monitor and investigate risky edits in real time, with AI-generated explanations
powered by OpenAI GPT-4o-mini.

---

## Architecture

```
Wikipedia Dumps (≥10 GB)
        │
        ▼
  Spark Batch Ingest   ←── XML iterparse (bz2 streaming)
        │
  Feature Engineering  ←── TF-IDF(comment) + TF-IDF(title) + numeric
        │
  SparkML Training     ←── LogisticRegression + CrossValidator
        │
  MLflow Tracking      ←── params, metrics, confusion matrix
        │
  Model Registry ──────────────────────────────────┐
                                                    │
Wikimedia EventStreams (SSE)                        │
        │                                           │
  SSE Collector  ──► data/stream/incoming/          │
                             │                      │
                    Spark Structured Streaming  ◄───┘
                             │
                   Risk Scoring (probability)
                             │
               ┌─────────────┴──────────────┐
               ▼                             ▼
        SQLite DB                   Parquet Archive
               │
        FastAPI Backend  ──►  /edits/recent, /explain/{id}
               │
        Streamlit Dashboard  +  OpenAI Explainer
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Java 17+ (for PySpark)
- Docker + Docker Compose (for full stack)
- OpenAI API key (optional — AI explanations fall back gracefully)

### 1. Local development

```bash
# Install dependencies
make setup

# Edit environment variables
cp .env.example .env
# → fill in OPENAI_API_KEY (optional)

# Run batch pipeline (skip download if you already have dumps)
make batch            # parses existing dumps
make train            # trains model + logs to MLflow

# Start services (3 terminals)
make serve            # FastAPI on :8000
make ui               # Streamlit on :8501
make collect          # SSE collector (populates DB with live edits)
make stream           # Spark Structured Streaming processor
```

### 2. Full Docker stack

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY in .env

make up               # builds + starts all services

# Services:
#   MinIO console    →  http://localhost:9001
#   MLflow UI        →  http://localhost:5001
#   API docs         →  http://localhost:8000/docs
#   Dashboard        →  http://localhost:8501
```

---

## Dataset

| Property | Value |
|---|---|
| Source | [dumps.wikimedia.org/enwiki/](https://dumps.wikimedia.org/enwiki/) |
| Dump date | 2024-04-20 |
| Format | bz2-compressed MediaWiki XML |
| Files | 3 revision history files |
| Compressed | ~4.2 GB |
| **Decompressed** | **≥ 29 GB** (≫ 10 GB requirement) |
| Records | Millions of revisions across hundreds of thousands of articles |

See [`manifests/dataset_manifest.json`](manifests/dataset_manifest.json) for
full details.

---

## Machine Learning

### Feature Set

| Feature | Type | Description |
|---|---|---|
| `comment_tfidf` | Sparse vector (256-d) | TF-IDF on edit comment |
| `title_tfidf` | Sparse vector (128-d) | TF-IDF on page title words |
| `length_delta` | Float | Net bytes added/removed |
| `is_anon_int` | Binary | 1 if anonymous (IP) editor |
| `namespace` | Int | Wikipedia namespace (filtered to 0 = articles) |
| `hour_of_day` | Int (0–23) | Hour of edit |
| `day_of_week` | Int (1–7) | Day of week |

### Model

- **Algorithm**: SparkML `LogisticRegression`
- **Pipeline**: `RegexTokenizer → HashingTF → IDF → VectorAssembler → StandardScaler → LR`
- **Tuning**: `CrossValidator` (3-fold CV) over `regParam ∈ {0.001, 0.01, 0.1}`
- **Weak labels**: Heuristic rules on comment content, size delta, anonymity
- **Tracked**: AUC, PR-AUC, F1, Precision, Recall, Confusion Matrix

### Reproducibility

```bash
# Re-train from scratch
make full-pipeline

# Re-train from existing parsed data
make train

# Check latest model
cat manifests/model_manifest.json
```

MLflow UI: http://localhost:5001

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health + external deps |
| `/stats` | GET | Aggregate risk statistics |
| `/edits/recent` | GET | Paginated recent edits (filterable by risk_label) |
| `/edits/{id}` | GET | Single edit record |
| `/explain/{id}` | GET | AI explanation for risk |
| `/explain` | POST | AI explanation (body: `{"edit_id": "..."}`) |

Full interactive docs: http://localhost:8000/docs

---

## Project Structure

```
wikirisk/
├── docker-compose.yml        All services
├── Makefile                  Dev commands
├── requirements.txt          Python dependencies
├── .env.example              Environment variables template
│
├── src/
│   ├── config/settings.py    Pydantic-settings centralised config
│   ├── common/
│   │   ├── logger.py         Structured logging (structlog + JSON)
│   │   └── spark_session.py  Singleton SparkSession factory
│   ├── batch/
│   │   ├── downloader.py     Wikipedia dump downloader
│   │   ├── parser.py         bz2 XML → Parquet via iterparse
│   │   └── pipeline.py       Batch orchestrator (download→parse→features→train)
│   ├── ml/
│   │   ├── features.py       SparkML feature pipeline (TF-IDF + numeric)
│   │   ├── labeler.py        Weak supervision labelling heuristics
│   │   ├── trainer.py        CrossValidator training + MLflow logging
│   │   └── evaluator.py      AUC/PR-AUC/F1 + confusion matrix
│   ├── streaming/
│   │   ├── collector.py      Wikimedia SSE → JSONL files + SQLite
│   │   └── processor.py      Spark Structured Streaming + foreachBatch
│   ├── serving/
│   │   ├── main.py           FastAPI app (CORS, logging, lifespan)
│   │   ├── models.py         Pydantic response models
│   │   ├── database.py       Async SQLite layer (aiosqlite)
│   │   └── routes/           health, edits, explain endpoints
│   ├── ui/app.py             Streamlit dashboard
│   └── ai/explainer.py       OpenAI GPT + rule-based fallback
│
├── manifests/
│   ├── dataset_manifest.json Dataset provenance + size verification
│   └── model_manifest.json   Latest trained model metadata
│
├── docker/
│   ├── mlflow.Dockerfile
│   ├── api.Dockerfile        API + Spark container
│   └── ui.Dockerfile         Streamlit container
│
└── tests/
    ├── conftest.py            Fixtures
    ├── test_features.py       Feature + labelling unit tests
    └── test_api.py            FastAPI integration tests
```

---

## Testing

```bash
make test             # full suite with coverage
make test-fast        # skip slow (Spark) tests
```

---

## Collaboration

| Contributor | Responsibilities |
|---|---|
| Team Member 1 | Batch pipeline, XML parsing, feature engineering |
| Team Member 2 | SparkML training, MLflow integration, model evaluation |
| Team Member 3 | Streaming pipeline, Spark Structured Streaming |
| Team Member 4 | FastAPI backend, SQLite integration, AI explainer |
| Team Member 5 | Streamlit UI, Docker setup, documentation |

See GitHub Issues and Pull Requests for collaboration history.

---

## Course Requirement Checklist

- [x] Dataset ≥ 10 GB decompressed (≈ 29 GB confirmed)
- [x] Spark batch processing
- [x] MLflow experiment tracking + model registry
- [x] Spark Structured Streaming (real-time inference)
- [x] FastAPI model deployment (`/edits/recent`, `/explain/{id}`)
- [x] Streamlit frontend (live dashboard)
- [x] AI explainability (OpenAI GPT-4o-mini + fallback)
- [x] Docker + docker-compose (containerised stack)
- [x] Structured logging (structlog JSON)
- [x] Reproducibility (re-train via `make full-pipeline`)
- [x] Dataset + model manifests

---

## License

For educational use (DATA 228, San Jose State University, Spring 2026).
