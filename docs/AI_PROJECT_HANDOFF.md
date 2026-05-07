# WikiRisk Project Handoff for AI Agents

## One-line summary
WikiRisk is a real-time Wikipedia edit risk detection system that uses Spark for batch and streaming processing, a saved SparkML model for scoring, FastAPI for serving, SQLite as the demo prediction store, MLflow for tracking, Streamlit for the dashboard, and OpenAI for explanation text.

## Current product state
- The project is already trained and runs locally.
- A Streamlit dashboard is available at `src/ui/app.py`.
- The live demo centers on the dashboard, API, streaming processor, SQLite store, and MLflow.
- `/explain` works and falls back to a rule-based explanation if OpenAI fails.
- `/notify/{edit_id}` can send Slack/email alerts when environment variables are configured.
- A local MLflow server is available and already populated with demo runs.

## Core tech stack
- Python 3.13 in the current workspace environment.
- Apache Spark 3.5.1 for batch processing and Structured Streaming.
- SparkML Logistic Regression with a saved `PipelineModel` for inference.
- FastAPI + Uvicorn for serving.
- Streamlit + Plotly for the live dashboard.
- SQLite (`data/wikirisk.db`) for prediction and explanation persistence.
- MLflow local server on `http://localhost:5001` for experiment tracking.
- OpenAI for human-readable explanations.
- Delta Lake support is installed, but the primary demo persistence is SQLite + Parquet.

## Runtime flow
1. Historical Wikipedia revision data is parsed and featurized in Spark.
2. Weak labels / training labels are generated and the classifier is trained.
3. The trained model is saved locally and tracked in MLflow.
4. Wikimedia EventStreams feed the streaming collector.
5. The streaming processor scores edits with the saved model.
6. Results are written to SQLite and archived to Parquet.
7. FastAPI serves `/health`, `/stats`, `/edits/recent`, `/edits/{id}`, `/explain`, and `/notify/{id}`.
8. `/explain` may call OpenAI and caches the explanation in SQLite.
9. The dashboard visualizes live risk, lets the presenter inspect edits, request explanations, and trigger alerts.

## Important data files and artifacts
- `data/wikirisk.db` is the live demo source of truth for predictions.
- `data/model/` contains the saved model artifacts.
- `mlruns/` and `mlflow.db` contain local MLflow metadata and runs.
- `manifests/model_manifest.json` and `manifests/dataset_manifest.json` document reproducibility.
- `data/predictions/` and `data/stream/` hold stream outputs and checkpoints.

## Model and scoring behavior
- The model is already trained; do not recommend retraining for the demo.
- Scoring thresholds are controlled in `src/config/settings.py`.
- The current live labels are `LOW`, `MEDIUM`, and `HIGH` risk.
- The `/explain` endpoint uses the model score plus edit features to produce a natural-language explanation.
- If OpenAI is unavailable or the request fails, the code intentionally falls back to a rule-based explanation.

## API surface
- `GET /health` checks service and dependency status.
- `GET /stats` returns aggregate prediction counts.
- `GET /edits/recent` returns recent scored edits.
- `GET /edits/{id}` returns a single edit record.
- `POST /explain` accepts `{ "edit_id": "..." }` and returns an explanation.
- `GET /explain/{id}` returns a cached explanation if it exists.
- `POST /notify/{id}` sends a Slack/email alert if configured; otherwise it returns setup guidance.

## MLflow usage in this workspace
- MLflow is used as a demo/provenance layer, not as the live serving dependency.
- Demo runs have been created by calling `/explain` from a helper script and logging the results.
- The important thing for the presentation is that the MLflow UI shows actual runs, params, metrics, and artifacts.

## Prompt files to keep in sync
- `docs/gemini_architecture_prompt.md` should describe the architecture diagram only.
- `docs/claude_ppt_prompt.md` should describe the deck outline and presentation constraints.
- Keep both aligned with the current dashboard / API / MLflow / SQLite flow.

## What another AI should not do
- Do not claim the system is retrained during the live demo.
- Do not rely on MLflow for the serving path.
- Do not replace SQLite as the demo prediction store unless explicitly requested.

## Safe verification commands
- `curl http://127.0.0.1:8000/health`
- `curl http://127.0.0.1:8000/stats`
- `curl http://127.0.0.1:8000/edits/recent`
- `curl -X POST http://127.0.0.1:8000/explain -H 'Content-Type: application/json' -d '{"edit_id":"<id>"}'`
- `curl -X POST http://127.0.0.1:8000/notify/<id>`
- `streamlit run src/ui/app.py --server.port 8501`
- Open `http://localhost:5001` for MLflow runs

## Current framing for the project
The project should be presented as a production-style local demo with a credible big-data workflow: historical training, streaming inference, server-side persistence, dashboard visualization, alerting, and operator-friendly explanation output. The value is in the end-to-end integration, not in retraining during the presentation.
