"""
WikiRisk – Spark Structured Streaming Processor.

Reads JSONL batches produced by the SSE collector, applies the trained
SparkML model, and writes scored predictions to:
  • data/predictions/  (Parquet, partitioned by date)
  • data/wikirisk.db   (SQLite via foreachBatch, for the FastAPI backend)

The process runs continuously until SIGTERM/SIGINT.

CLI:
    python -m src.streaming.processor [--trigger-interval 30]
"""
from __future__ import annotations

import signal
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import click

from src.common.logger import configure_logging, get_logger
from src.config import get_settings

log = get_logger(__name__)

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id            TEXT PRIMARY KEY,
    rev_id        TEXT,
    page_title    TEXT,
    namespace     INTEGER,
    user          TEXT,
    is_anon       INTEGER,
    comment       TEXT,
    length_delta  REAL,
    timestamp     TEXT,
    wiki          TEXT,
    risk_score    REAL,
    risk_label    TEXT,
    scored        INTEGER DEFAULT 1,
    created_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_created ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk    ON predictions (risk_score DESC);
"""

# Input JSON schema matching collector's output
_EVENT_SCHEMA = """
    `id` LONG,
    `type` STRING,
    `namespace` INT,
    `title` STRING,
    `comment` STRING,
    `timestamp` LONG,
    `user` STRING,
    `bot` BOOLEAN,
    `wiki` STRING,
    `server_name` STRING,
    `length` STRUCT<old: INT, new: INT>,
    `revision` STRUCT<old: LONG, new: LONG>,
    `meta` STRUCT<dt: STRING, id: STRING>
"""


def _ensure_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    for stmt in _SCHEMA_DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()
    conn.close()


def _upsert_predictions(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO predictions
            (id, rev_id, page_title, namespace, user, is_anon, comment,
             length_delta, timestamp, wiki, risk_score, risk_label,
             scored, created_at)
        VALUES (:id, :rev_id, :page_title, :namespace, :user, :is_anon,
                :comment, :length_delta, :timestamp, :wiki,
                :risk_score, :risk_label, 1, :created_at)
        """,
        rows,
    )
    conn.commit()


def _write_delta_archive(scored_df, cfg) -> None:
    """Append scored rows to a Delta Lake archive when Delta is available."""
    delta_path = cfg.predictions_dir / "delta"
    try:
        scored_df.write.format("delta").mode("append").save(str(delta_path))
        log.info("delta_archive_written", path=str(delta_path))
    except Exception as exc:
        log.warning("delta_archive_skipped", error=str(exc), path=str(delta_path))


def _risk_label(score: float, cfg) -> str:
    if score >= cfg.risk_high_threshold:
        return "HIGH"
    if score >= cfg.risk_medium_threshold:
        return "MEDIUM"
    return "LOW"


def run_streaming_processor(
    trigger_interval_seconds: int = 30,
    await_termination: bool = True,
) -> None:
    cfg = get_settings()
    cfg.ensure_dirs()

    from src.common.spark_session import get_spark
    from src.ml.features import FEATURES_COL, prepare_streaming_record

    spark = get_spark()
    db_path = cfg.db_path
    _ensure_db(db_path)

    # ── Load trained model ─────────────────────────────────────────────────
    log.info("loading_model")
    try:
        from src.ml.evaluator import load_production_model
        model = load_production_model(cfg)
    except RuntimeError as exc:
        log.error("model_load_failed", error=str(exc))
        log.warning(
            "running_in_passthrough_mode",
            note="Predictions will use random scores until a model is trained",
        )
        model = None

    # ── Define streaming source ────────────────────────────────────────────
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField,
        StringType, IntegerType, LongType, BooleanType,
        DoubleType,
    )

    # Schema for JSONL files written by collector
    event_schema = (
        StructType()
        .add("id", LongType())
        .add("type", StringType())
        .add("namespace", IntegerType())
        .add("title", StringType())
        .add("comment", StringType())
        .add("timestamp", LongType())
        .add("user", StringType())
        .add("bot", BooleanType())
        .add("wiki", StringType())
        .add("server_name", StringType())
        .add(
            "length",
            StructType()
            .add("old", IntegerType())
            .add("new", IntegerType()),
        )
        .add(
            "revision",
            StructType()
            .add("old", LongType())
            .add("new", LongType()),
        )
        .add(
            "meta",
            StructType()
            .add("dt", StringType())
            .add("id", StringType()),
        )
    )

    stream_df = (
        spark.readStream
        .schema(event_schema)
        .option("maxFilesPerTrigger", 10)
        .json(str(cfg.stream_input_dir))
    )

    # ── Feature engineering (streaming) ───────────────────────────────────
    processed = (
        stream_df
        .filter(F.col("type") == "edit")
        .filter(F.col("namespace") == 0)
        .withColumn(
            "length_delta",
            (F.col("length.new") - F.col("length.old")).cast("double"),
        )
        .withColumn(
            "is_anon",
            F.col("user").rlike(r"^(\d{1,3}\.){3}\d{1,3}$"),
        )
        .withColumn("is_anon_int", F.col("is_anon").cast("int"))
        .withColumn(
            "event_time",
            F.to_timestamp(F.col("meta.dt")),
        )
        .withColumn("hour_of_day", F.hour("event_time"))
        .withColumn("day_of_week", F.dayofweek("event_time"))
        .withColumn(
            "comment_clean",
            F.lower(F.regexp_replace(F.col("comment"), r"[^a-zA-Z0-9\s]", " ")),
        )
        .withColumn(
            "title_clean",
            F.lower(F.regexp_replace(F.col("title"), r"[^a-zA-Z0-9\s]", " ")),
        )
        .na.fill(
            {
                "comment_clean": "",
                "title_clean": "",
                "hour_of_day": 0,
                "day_of_week": 1,
                "length_delta": 0.0,
                "is_anon_int": 0,
            }
        )
    )

    # ── foreachBatch sink ─────────────────────────────────────────────────
    def process_batch(batch_df, batch_id: int) -> None:
        if batch_df.rdd.isEmpty():
            return

        log.info("processing_batch", batch_id=batch_id, count=batch_df.count())

        if model is not None:
            scored = model.transform(batch_df)
            # Extract probability of class 1
            get_prob = F.udf(lambda v: float(v[1]) if v is not None else 0.5, DoubleType())
            scored = scored.withColumn("risk_score", get_prob(F.col("probability")))
        else:
            import random
            scored = batch_df.withColumn(
                "risk_score", (F.rand(seed=42) * 0.4 + 0.1).cast("double")
            )

        rows_pd = scored.select(
            F.col("revision.new").cast("string").alias("rev_id"),
            F.col("title").alias("page_title"),
            F.col("namespace"),
            F.col("user"),
            F.col("is_anon_int").alias("is_anon"),
            F.col("comment"),
            F.col("length_delta"),
            F.col("meta.dt").alias("timestamp"),
            F.col("wiki"),
            F.col("risk_score"),
        ).toPandas()

        import uuid as _uuid
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        db_rows = []
        for _, r in rows_pd.iterrows():
            score = float(r.get("risk_score") or 0.0)
            db_rows.append(
                {
                    "id": str(_uuid.uuid4()),
                    "rev_id": str(r.get("rev_id") or ""),
                    "page_title": str(r.get("page_title") or ""),
                    "namespace": int(r.get("namespace") or 0),
                    "user": str(r.get("user") or ""),
                    "is_anon": int(r.get("is_anon") or 0),
                    "comment": str(r.get("comment") or ""),
                    "length_delta": float(r.get("length_delta") or 0.0),
                    "timestamp": str(r.get("timestamp") or now_iso),
                    "wiki": str(r.get("wiki") or ""),
                    "risk_score": score,
                    "risk_label": _risk_label(score, cfg),
                    "created_at": now_iso,
                }
            )

        conn = sqlite3.connect(db_path)
        _upsert_predictions(conn, db_rows)
        conn.close()

        # Also write to Parquet for archival
        parquet_path = (
            cfg.predictions_dir
            / datetime.now(tz=timezone.utc).strftime("date=%Y-%m-%d")
        )
        scored.write.mode("append").parquet(str(parquet_path))
        _write_delta_archive(scored, cfg)
        log.info(
            "batch_processed",
            batch_id=batch_id,
            rows=len(db_rows),
        )

    # ── Start streaming query ──────────────────────────────────────────────
    query = (
        processed.writeStream
        .foreachBatch(process_batch)
        .trigger(processingTime=f"{trigger_interval_seconds} seconds")
        .option("checkpointLocation", str(cfg.stream_checkpoint_dir))
        .start()
    )

    log.info(
        "streaming_query_started",
        trigger_interval=trigger_interval_seconds,
        input_dir=str(cfg.stream_input_dir),
    )

    if await_termination:
        try:
            query.awaitTermination()
        except KeyboardInterrupt:
            log.info("keyboard_interrupt_stopping_stream")
            query.stop()


@click.command()
@click.option(
    "--trigger-interval",
    default=30,
    help="Spark micro-batch interval in seconds",
)
def main(trigger_interval: int) -> None:
    """Run Spark Structured Streaming processor for WikiRisk."""
    cfg = get_settings()
    configure_logging(cfg.log_level, cfg.log_format)
    run_streaming_processor(trigger_interval_seconds=trigger_interval)


if __name__ == "__main__":
    main()
