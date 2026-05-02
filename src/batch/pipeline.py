"""
WikiRisk – Batch pipeline orchestrator.

Phase 1: Download Wikipedia dumps (optional, --skip-download to bypass)
Phase 2: Parse XML → Parquet
Phase 3: Feature engineering (delegates to src/ml/features.py)
Phase 4: Weak labelling (delegates to src/ml/labeler.py)
Phase 5: SparkML training + MLflow logging (delegates to src/ml/trainer.py)

CLI:
    python -m src.batch.pipeline [--skip-download] [--skip-parse]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import click

from src.common.logger import configure_logging, get_logger
from src.config import get_settings

log = get_logger(__name__)


def _check_manifests(cfg) -> None:
    """Warn if dataset_manifest.json is missing."""
    p = cfg.manifests_dir / "dataset_manifest.json"
    if not p.exists():
        log.warning("manifest_missing", path=str(p))


def run_download(cfg) -> None:
    from src.batch.downloader import download_dumps, write_manifest

    log.info("phase_download_start")
    records = download_dumps(cfg.raw_data_dir, min_gb=10.0)
    write_manifest(records, cfg.manifests_dir / "dataset_manifest.json")
    log.info("phase_download_done", files=len(records))


def run_parse(cfg, spark) -> int:
    from src.batch.parser import parse_dump_to_parquet

    dump_files = sorted(cfg.raw_data_dir.glob("*.bz2"))
    if not dump_files:
        raise FileNotFoundError(
            f"No .bz2 dump files found in {cfg.raw_data_dir}. "
            "Run with --no-skip-download first."
        )

    log.info("phase_parse_start", num_files=len(dump_files))
    total_revisions = 0

    for dump_path in dump_files:
        n = parse_dump_to_parquet(
            dump_path,
            output_dir=cfg.processed_data_dir / "revisions",
            spark_session=spark,
        )
        total_revisions += n
        log.info("dump_parsed", file=dump_path.name, revisions=n)

    log.info("phase_parse_done", total_revisions=total_revisions)
    return total_revisions


def run_feature_engineering(cfg, spark):
    """Load parsed Parquet, apply feature transforms, write features Parquet."""
    from pyspark.sql import functions as F

    revisions_path = str(cfg.processed_data_dir / "revisions")
    log.info("phase_features_start", path=revisions_path)

    df = spark.read.parquet(revisions_path)
    log.info("revisions_loaded", count=df.count())

    # ── Basic derived columns ──────────────────────────────────────────────
    df = (
        df
        # namespace 0 = main article space
        .filter(F.col("namespace") == 0)
        # Parse timestamp
        .withColumn(
            "event_time",
            F.to_timestamp("timestamp", "yyyy-MM-dd'T'HH:mm:ss'Z'"),
        )
        .withColumn("hour_of_day", F.hour("event_time"))
        .withColumn("day_of_week", F.dayofweek("event_time"))
        # Size delta: current text size relative to prior
        .withColumn("text_bytes", F.col("text_bytes").cast("long"))
        .withColumn("parent_bytes", F.lit(0).cast("long"))  # placeholder
        .withColumn(
            "length_delta",
            F.col("text_bytes") - F.col("parent_bytes"),
        )
        # Anonymous flag
        .withColumn("is_anon_int", F.col("is_anon").cast("int"))
        # Sanitise text
        .withColumn(
            "comment_clean",
            F.lower(F.regexp_replace(F.col("comment"), r"[^a-zA-Z0-9\s]", " ")),
        )
        .withColumn(
            "title_clean",
            F.lower(F.regexp_replace(F.col("page_title"), r"[^a-zA-Z0-9\s]", " ")),
        )
    )

    df = df.na.fill(
        {
            "comment_clean": "",
            "title_clean": "",
            "hour_of_day": 0,
            "day_of_week": 1,
            "length_delta": 0,
            "is_anon_int": 0,
        }
    )

    features_path = str(cfg.processed_data_dir / "features")
    df.write.mode("overwrite").parquet(features_path)

    count = df.count()
    log.info("phase_features_done", records=count, path=features_path)
    return count


def run_labeling(cfg, spark):
    """Apply weak supervision labels to the features dataset."""
    from src.ml.labeler import apply_weak_labels

    features_path = str(cfg.processed_data_dir / "features")
    log.info("phase_labeling_start")

    df = spark.read.parquet(features_path)
    df_labeled = apply_weak_labels(df)

    labeled_path = str(cfg.processed_data_dir / "labeled")
    df_labeled.write.mode("overwrite").parquet(labeled_path)

    pos = df_labeled.filter("label = 1").count()
    neg = df_labeled.filter("label = 0").count()
    log.info(
        "phase_labeling_done",
        total=pos + neg,
        positive=pos,
        negative=neg,
        pos_rate=round(pos / max(pos + neg, 1), 3),
    )
    return pos + neg


def run_training(cfg) -> str:
    """Train SparkML model, log to MLflow, return run_id."""
    from src.ml.trainer import train

    log.info("phase_training_start")
    run_id = train()
    log.info("phase_training_done", run_id=run_id)
    return run_id


@click.command()
@click.option(
    "--skip-download",
    is_flag=True,
    default=False,
    help="Skip downloading dumps (use existing files in data/raw/dumps/)",
)
@click.option(
    "--skip-parse",
    is_flag=True,
    default=False,
    help="Skip XML parsing (use existing Parquet in data/processed/revisions/)",
)
@click.option(
    "--skip-train",
    is_flag=True,
    default=False,
    help="Skip model training (only process data)",
)
def main(
    skip_download: bool,
    skip_parse: bool,
    skip_train: bool,
) -> None:
    """Run the full WikiRisk batch pipeline."""
    cfg = get_settings()
    configure_logging(cfg.log_level, cfg.log_format)

    log.info(
        "batch_pipeline_start",
        skip_download=skip_download,
        skip_parse=skip_parse,
        skip_train=skip_train,
    )
    t0 = time.monotonic()

    # Phase 1: Download
    if not skip_download:
        run_download(cfg)
    else:
        log.info("phase_download_skipped")
        _check_manifests(cfg)

    # Initialise Spark (after download to avoid long Spark startup for net ops)
    from src.common.spark_session import get_spark
    spark = get_spark()

    # Phase 2: Parse
    if not skip_parse:
        run_parse(cfg, spark)
    else:
        log.info("phase_parse_skipped")

    # Phase 3+4: Features + Labels
    run_feature_engineering(cfg, spark)
    run_labeling(cfg, spark)

    # Phase 5: Train
    if not skip_train:
        run_id = run_training(cfg)
        log.info("model_run_id", run_id=run_id)

    elapsed = time.monotonic() - t0
    log.info("batch_pipeline_complete", elapsed_seconds=round(elapsed, 1))


if __name__ == "__main__":
    main()
