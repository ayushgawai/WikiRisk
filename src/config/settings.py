"""
WikiRisk – Centralised application settings.

All values can be overridden via environment variables or a .env file.
Pydantic-settings validates types at import time.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root (two levels up from this file: src/config/settings.py → /)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Spark ──────────────────────────────────────────────────────────────
    spark_master: str = Field("local[*]", description="Spark master URL")
    spark_app_name: str = "WikiRisk"
    spark_driver_memory: str = "4g"
    spark_executor_memory: str = "4g"

    # ── MLflow ─────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(
        "http://localhost:5001", description="MLflow tracking server URI"
    )
    mlflow_experiment_name: str = "wikirisk-edit-risk"
    mlflow_model_name: str = "wikirisk-risk-classifier"
    mlflow_s3_endpoint_url: str = "http://localhost:9000"

    # ── MinIO / S3 ─────────────────────────────────────────────────────────
    aws_access_key_id: str = "wikirisk"
    aws_secret_access_key: str = "wikirisk123"

    # ── Data Paths ─────────────────────────────────────────────────────────
    data_dir: Path = ROOT_DIR / "data"
    raw_data_dir: Path = ROOT_DIR / "data" / "raw" / "dumps"
    processed_data_dir: Path = ROOT_DIR / "data" / "processed"
    predictions_dir: Path = ROOT_DIR / "data" / "predictions"
    stream_input_dir: Path = ROOT_DIR / "data" / "stream" / "incoming"
    stream_checkpoint_dir: Path = ROOT_DIR / "data" / "stream" / "checkpoints"
    manifests_dir: Path = ROOT_DIR / "manifests"

    # ── API ────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    database_url: str = Field(
        "sqlite:///data/wikirisk.db",
        description="Async SQLite DSN for predictions store",
    )
    api_cors_origins: list[str] = ["*"]

    # ── UI ─────────────────────────────────────────────────────────────────
    api_url: str = "http://localhost:8000"

    # ── OpenAI ─────────────────────────────────────────────────────────────
    openai_api_key: str = Field("", description="OpenAI API key")
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = 300
    openai_cache_ttl_seconds: int = 3600

    # ── Wikimedia Streaming ────────────────────────────────────────────────
    wikimedia_stream_url: str = (
        "https://stream.wikimedia.org/v2/stream/recentchange"
    )
    stream_target_wiki: str = "enwiki"
    stream_batch_size: int = 100
    stream_flush_interval_seconds: int = 10

    # ── ML Training ────────────────────────────────────────────────────────
    ml_max_iter: int = 100
    ml_reg_param: float = 0.01
    ml_elastic_net_param: float = 0.0
    ml_num_folds: int = 3
    ml_train_test_split: float = 0.8
    ml_min_records_required: int = 1_000

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "pretty"

    # ── Risk Thresholds ────────────────────────────────────────────────────
    # Operational bands for the live feed.
    # The model is conservative, so these cutoffs emphasize relative risk.
    risk_high_threshold: float = 0.20
    risk_medium_threshold: float = 0.08

    @field_validator("data_dir", "raw_data_dir", "processed_data_dir",
                     "predictions_dir", "stream_input_dir",
                     "stream_checkpoint_dir", "manifests_dir", mode="before")
    @classmethod
    def _make_path(cls, v: str | Path) -> Path:
        return Path(v)

    def ensure_dirs(self) -> None:
        """Create all required data directories."""
        dirs = [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.predictions_dir,
            self.stream_input_dir,
            self.stream_checkpoint_dir,
            self.manifests_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> str:
        """Absolute path to SQLite database (strips sqlite:/// prefix)."""
        dsn = self.database_url
        if dsn.startswith("sqlite:///"):
            raw = dsn[len("sqlite:///"):]
            p = Path(raw)
            if not p.is_absolute():
                p = ROOT_DIR / p
            return str(p)
        return dsn

    @property
    def risk_label(self) -> dict[str, str]:
        return {
            "high": "HIGH",
            "medium": "MEDIUM",
            "low": "LOW",
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return singleton Settings (cached after first call)."""
    s = Settings()
    s.ensure_dirs()
    return s
