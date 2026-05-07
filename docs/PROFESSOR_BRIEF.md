# WikiRisk — Brief for instructor review

## What the system does

WikiRisk ingests **live** English Wikipedia edits (Wikimedia EventStreams), scores each edit with a **trained SparkML model**, stores results in SQLite, exposes them through a **FastAPI** service, and shows a **Streamlit** dashboard with **OpenAI** explanations. **MLflow** tracks training runs and metrics. **Apache Spark** is used both for **batch training** on historical dumps and for **structured streaming** in the architecture.

---

## Historical vs streaming data

| Layer | Source | Role |
|--------|--------|------|
| **Historical (batch)** | Wikimedia **mediawiki_history** monthly TSV.bz2 dumps (~12 GB uncompressed across chosen files) | Train and evaluate the classifier on millions of revision rows with a **verifiable label**. |
| **Streaming (online)** | Wikimedia **EventStreams** (SSE) | Drive the live dashboard; each incoming edit is featurized the same way as in training and scored by the **same** saved SparkML `PipelineModel` loaded from `data/model/`. |

We do **not** duplicate the full dump as a giant uncompressed file on disk: Spark reads **`.bz2` directly**; we only cache **exact uncompressed byte totals** in `data/raw/mediawiki_history/sizes.json` (one-time streaming decode) so re-runs do not repeat that work.

---

## Target variable and training labels

- **Target (what we predict):** probability that an edit is **problematic** in the same sense Wikipedia’s own history records it—i.e. the edit was **identity-reverted** later by the community.
- **Ground-truth column in the dump:** `revision_is_identity_reverted` (boolean from Wikimedia’s analytics pipeline).  
  - `true` → **label 1** (“reverted / risky” for training purposes)  
  - `false` → **label 0** (not identity-reverted)
- **Features (inputs):** TF-IDF on cleaned edit comment and page title (hashing buckets), byte change (`revision_text_bytes_diff`), anonymous flag, hour-of-day, day-of-week—aligned between batch training and live scoring.

This is **not** a subjective “we guessed risk” label; it is **observed community outcome** encoded in an official dataset.

---

## Model, hyperparameters, and training setup

- **Algorithm:** SparkML **Logistic Regression** on top of a **Pipeline**: `RegexTokenizer` → `HashingTF` → `IDF` (comment + title) → `VectorAssembler` → `StandardScaler` → `LogisticRegression`.
- **Hyperparameter search:** `CrossValidator` over a small grid, e.g. `regParam ∈ {0.001, 0.01, 0.1}` and `maxIter ∈ {50, 100}` (see notebook for exact grid used in the demo vs full run).
- **Split:** random 80% train / 20% test (fixed seed for reproducibility).
- **Metrics logged:** AUC-ROC, AUC-PR, F1, precision, recall, accuracy, training time—stored in **MLflow** and in `manifests/model_manifest.json`.

---

## Does something like this already exist? Why is this project still useful?

- **Wikimedia already runs** large-scale quality models (e.g. **ORES / Lift Wing** “damaging” scores). Those are **production** services; students typically do not retrain them end-to-end on raw multi-GB dumps.
- **Our contribution** is an **integrated teaching / demo stack**: download → Spark feature pipeline → cross-validated LR → MLflow → **local** model artifact → **same** model applied to **live** edits in an app with API + UI + LLM explanations. That full loop in one repo is what is “unique” for a course project—not beating ORES on accuracy.

---

## How accurate is it?

- Report **test-set metrics from MLflow / the notebook output** (AUC-ROC, F1, etc.) after your run; numbers depend on the exact months sampled and class balance (~low base rate of reverts).
- The dashboard maps model probability to **LOW / MEDIUM / HIGH** using calibrated thresholds in `src/config/settings.py` (tuned against real score distributions), which is a **product** choice, not the raw ML metric.

---

## How the app uses the model

1. Training saves `PipelineModel` under `data/model/wikirisk_lr_model/` (or `…_smoke` for the fast path).
2. The streaming / ingestion path loads that model (see `src/ml/evaluator.py`) and applies the same transforms to live edits.
3. FastAPI persists scores; Streamlit reads them; OpenAI text is **explanatory**, not the source of truth for the score.

---

## Reproducibility and “no rework” on re-run

- Downloads: skip if files already exist.
- Uncompressed size: cached JSON; no repeated full decompress for measurement.
- Training: if model directory + manifest + label stats exist, notebook can **skip** the heavy Spark pass (configurable paths for full vs smoke demo).
- Spark shuffle: directed to `data/spark-local/` (gitignored), not ad hoc OS temp paths for project policy.

---

## What you should run in class

**We do not run the notebook for you from automation**—run `notebooks/WikiRisk_Training_Pipeline.ipynb` yourself (cell-by-cell is ideal for showing outputs).

1. **Fast path:**  
   `python scripts/smoke_test_notebook.py --generate-only`  
   then set `USE_SMOKE_DEMO = True` in **cell 2** of the notebook.
2. **Full path:** leave `USE_SMOKE_DEMO = False`, ensure the three monthly `.bz2` files are present, then execute the notebook (long run).

CLI check (fast, no notebook):  
`python scripts/smoke_test_notebook.py`
