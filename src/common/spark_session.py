"""
WikiRisk – Spark session factory.

Provides a singleton SparkSession pre-configured for:
  • Delta Lake
  • MLflow integration
  • MinIO (S3-compatible) artifact access
  • Structured streaming with JSON file source
"""
from __future__ import annotations

import os
from typing import Optional

from src.common.logger import get_logger

log = get_logger(__name__)

_spark: Optional[SparkSession] = None


def get_spark(
    *,
    app_name: str | None = None,
    master: str | None = None,
    enable_delta: bool = True,
    extra_conf: dict[str, str] | None = None,
):
    """Return a singleton SparkSession, creating one if necessary."""
    from pyspark.sql import SparkSession

    global _spark

    if _spark is not None and not _spark.sparkContext._jvm is None:
        try:
            _ = _spark.sparkContext.applicationId
            return _spark
        except Exception:
            _spark = None

    from src.config import get_settings
    cfg = get_settings()

    resolved_master = master or cfg.spark_master
    resolved_app = app_name or cfg.spark_app_name

    log.info("creating_spark_session", master=resolved_master, app=resolved_app)

    packages = []
    if enable_delta:
        packages.append("io.delta:delta-spark_2.12:3.1.0")
    # AWS SDK for MinIO / S3
    packages.append("org.apache.hadoop:hadoop-aws:3.3.4")
    packages.append("com.amazonaws:aws-java-sdk-bundle:1.12.262")

    builder = (
        SparkSession.builder.master(resolved_master)
        .appName(resolved_app)
        .config("spark.driver.memory", cfg.spark_driver_memory)
        .config("spark.executor.memory", cfg.spark_executor_memory)
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Silence verbose logging
        .config("spark.ui.showConsoleProgress", "false")
        # Delta
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Streaming
        .config("spark.sql.streaming.schemaInference", "true")
        # S3 / MinIO
        .config(
            "spark.hadoop.fs.s3a.endpoint",
            cfg.mlflow_s3_endpoint_url,
        )
        .config("spark.hadoop.fs.s3a.access.key", cfg.aws_access_key_id)
        .config("spark.hadoop.fs.s3a.secret.key", cfg.aws_secret_access_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config(
            "spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem",
        )
        .config("spark.jars.packages", ",".join(packages))
    )

    if extra_conf:
        for k, v in extra_conf.items():
            builder = builder.config(k, v)

    _spark = builder.getOrCreate()
    _spark.sparkContext.setLogLevel("WARN")

    log.info(
        "spark_session_ready",
        version=_spark.version,
        app_id=_spark.sparkContext.applicationId,
    )
    return _spark


def stop_spark() -> None:
    """Gracefully stop the singleton SparkSession."""
    global _spark
    if _spark is not None:
        log.info("stopping_spark_session")
        _spark.stop()
        _spark = None
