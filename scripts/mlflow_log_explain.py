#!/usr/bin/env python3
"""Log a few /explain responses to the local MLflow server for demo purposes.

This script picks a small set of uncached edit IDs from the local SQLite DB,
calls the running API at /explain, and logs the returned explanation text as an
MLflow run artifact along with a few simple metrics/params so the MLflow UI
shows meaningful content.
"""
import json
import sqlite3
from pathlib import Path
import requests
import mlflow

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "wikirisk.db"

MLFLOW_URI = "http://localhost:5001"
EXPERIMENT = "wikirisk-demo"
API_BASE = "http://127.0.0.1:8000"


def pick_edit_ids(limit=3):
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute(
        "SELECT id FROM predictions WHERE id NOT IN (SELECT edit_id FROM explanations) LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    con.close()
    return [r[0] for r in rows]


def call_explain(edit_id: str) -> dict:
    resp = requests.post(f"{API_BASE}/explain", json={"edit_id": edit_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    ids = pick_edit_ids(5)
    if not ids:
        print("No uncached edits found; nothing to log.")
        return

    for eid in ids:
        print("Calling explain for:", eid)
        out = call_explain(eid)
        model = out.get("model")
        explanation = out.get("explanation") or ""
        generated_at = out.get("generated_at")

        with mlflow.start_run(nested=False) as run:
            run_id = run.info.run_id
            mlflow.log_param("edit_id", eid)
            mlflow.log_param("model", model)
            if generated_at:
                mlflow.log_param("generated_at", generated_at)

            # log explanation as an artifact
            artifact_path = Path("artifacts")
            artifact_path.mkdir(exist_ok=True)
            fpath = artifact_path / f"explanation_{eid}.txt"
            fpath.write_text(explanation)
            mlflow.log_artifact(str(fpath))

            # optional metric: length of explanation
            mlflow.log_metric("explanation_length", len(explanation))

            print(f"Logged run {run_id} for edit {eid} (model={model})")


if __name__ == "__main__":
    main()
