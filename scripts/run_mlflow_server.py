"""Start a local MLflow tracking server for WikiRisk."""
from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    mlruns_dir = root / "mlruns"
    artifacts_dir = mlruns_dir / "artifacts"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    db_uri = f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}"
    artifact_root = f"file://{artifacts_dir.as_posix()}"

    os.execvp(
        "mlflow",
        [
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            "5001",
            "--backend-store-uri",
            db_uri,
            "--default-artifact-root",
            artifact_root,
            "--serve-artifacts",
        ],
    )


if __name__ == "__main__":
    main()