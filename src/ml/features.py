"""
WikiRisk – Feature engineering pipeline (SparkML).

Transforms raw revision records into a dense feature vector suitable for
SparkML classification.

Feature set (aligned between batch training and streaming inference):
  1. TF-IDF on edit comment   (256 hash buckets → IDF)
  2. TF-IDF on page title     (128 hash buckets → IDF)
  3. length_delta             (normalised numeric)
  4. is_anon_int              (0/1 flag)
  5. namespace                (passthrough; always 0 after filter)
  6. hour_of_day              (0–23)
  7. day_of_week              (1–7)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy PySpark imports – deferred so this module can be used in environments
# without PySpark (e.g. the API server, unit tests).
if TYPE_CHECKING:
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.sql import DataFrame

COMMENT_TOKENS_COL = "comment_tokens"
TITLE_TOKENS_COL = "title_tokens"
COMMENT_TF_COL = "comment_tf"
TITLE_TF_COL = "title_tf"
COMMENT_IDF_COL = "comment_idf"
TITLE_IDF_COL = "title_idf"
NUMERIC_COLS = ["length_delta", "is_anon_int", "hour_of_day", "day_of_week"]
ASSEMBLED_COL = "raw_features"
FEATURES_COL = "features"


def build_feature_pipeline(
    comment_col: str = "comment_clean",
    title_col: str = "title_clean",
    comment_hash_buckets: int = 256,
    title_hash_buckets: int = 128,
):
    """
    Build an unfitted SparkML Pipeline that transforms revision records
    into a single 'features' vector column.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import (
        HashingTF, IDF, RegexTokenizer, VectorAssembler, StandardScaler,
    )

    # ── Text tokenisation ──────────────────────────────────────────────────
    comment_tokenizer = RegexTokenizer(
        inputCol=comment_col,
        outputCol=COMMENT_TOKENS_COL,
        pattern=r"\s+",
        toLowercase=True,
        minTokenLength=2,
    )
    title_tokenizer = RegexTokenizer(
        inputCol=title_col,
        outputCol=TITLE_TOKENS_COL,
        pattern=r"\s+",
        toLowercase=True,
        minTokenLength=2,
    )

    # ── TF ────────────────────────────────────────────────────────────────
    comment_tf = HashingTF(
        inputCol=COMMENT_TOKENS_COL,
        outputCol=COMMENT_TF_COL,
        numFeatures=comment_hash_buckets,
    )
    title_tf = HashingTF(
        inputCol=TITLE_TOKENS_COL,
        outputCol=TITLE_TF_COL,
        numFeatures=title_hash_buckets,
    )

    # ── IDF ───────────────────────────────────────────────────────────────
    comment_idf = IDF(
        inputCol=COMMENT_TF_COL,
        outputCol=COMMENT_IDF_COL,
        minDocFreq=2,
    )
    title_idf = IDF(
        inputCol=TITLE_TF_COL,
        outputCol=TITLE_IDF_COL,
        minDocFreq=2,
    )

    # ── Assemble all features ─────────────────────────────────────────────
    assembler = VectorAssembler(
        inputCols=[COMMENT_IDF_COL, TITLE_IDF_COL, *NUMERIC_COLS],
        outputCol=ASSEMBLED_COL,
        handleInvalid="keep",
    )

    # ── Scale numeric portion ─────────────────────────────────────────────
    scaler = StandardScaler(
        inputCol=ASSEMBLED_COL,
        outputCol=FEATURES_COL,
        withMean=False,
        withStd=True,
    )

    return Pipeline(
        stages=[
            comment_tokenizer,
            title_tokenizer,
            comment_tf,
            title_tf,
            comment_idf,
            title_idf,
            assembler,
            scaler,
        ]
    )


def fit_features(df, **kwargs):
    """Fit feature pipeline on training data, return fitted PipelineModel."""
    pipeline = build_feature_pipeline(**kwargs)
    return pipeline.fit(df)


def prepare_streaming_record(raw: dict) -> dict:
    """
    Convert a raw Wikimedia EventStreams dict into a feature-ready record.

    Used by the streaming processor to align field names with the
    trained pipeline's expected column names.
    """
    import re
    from datetime import datetime, timezone

    # ── Parse timestamp ────────────────────────────────────────────────────
    ts_str = raw.get("timestamp", "")
    try:
        if isinstance(ts_str, int):
            dt = datetime.fromtimestamp(ts_str, tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        hour = dt.hour
        dow = dt.isoweekday()  # 1=Mon … 7=Sun
    except Exception:
        hour, dow = 0, 1

    # ── Length delta ───────────────────────────────────────────────────────
    length_info = raw.get("length", {}) or {}
    old_len = length_info.get("old") or 0
    new_len = length_info.get("new") or 0
    length_delta = new_len - old_len

    # ── Anonymous flag ─────────────────────────────────────────────────────
    user = raw.get("user", "") or ""
    is_anon = bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$|^[0-9a-fA-F:]+:[0-9a-fA-F:]+$", user))

    # ── Text cleaning ──────────────────────────────────────────────────────
    comment = raw.get("comment", "") or ""
    title = raw.get("title", "") or ""
    comment_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", comment.lower())
    title_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", title.lower())

    return {
        "page_id": str(raw.get("id", "")),
        "page_title": title,
        "namespace": int(raw.get("namespace", 0) or 0),
        "rev_id": str((raw.get("revision", {}) or {}).get("new", "")),
        "comment": comment,
        "comment_clean": comment_clean,
        "title_clean": title_clean,
        "hour_of_day": hour,
        "day_of_week": dow,
        "length_delta": float(length_delta),
        "is_anon": is_anon,
        "is_anon_int": int(is_anon),
        "contributor_username": user,
        "timestamp": raw.get("meta", {}).get("dt", "") if isinstance(raw.get("meta"), dict) else "",
    }
