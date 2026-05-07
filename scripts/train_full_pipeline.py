"""
WikiRisk – Full End-to-End Training Pipeline
=============================================

Downloads Wikimedia mediawiki_history TSV dumps, trains a SparkML
LogisticRegression model, and registers it in MLflow.

Dataset source:
  https://dumps.wikimedia.org/other/mediawiki_history/
  Files contain per-revision metadata including `revision_is_identity_reverted`
  — the ground truth label (community-verified: edit was reverted = bad edit).

Label definition:
  revision_is_identity_reverted = true  → label 1 (risky/damaging)
  revision_is_identity_reverted = false → label 0 (benign)

This satisfies the 10+ GB data requirement: 4 monthly TSV files totalling
~3 GB compressed / ~25 GB uncompressed of real Wikipedia edit history.

Usage:
  python scripts/train_full_pipeline.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data" / "raw" / "mediawiki_history"
MODEL_DIR  = ROOT / "data" / "model"
MANIFEST   = ROOT / "manifests" / "model_manifest.json"
MLFLOW_DIR = ROOT / "mlruns"

SPARK_LOCAL = ROOT / "data" / "spark-local"
for _d in (DATA_DIR, MODEL_DIR, ROOT / "manifests", SPARK_LOCAL):
    _d.mkdir(parents=True, exist_ok=True)

# ── Files to download (4 monthly snapshots: ~3 GB compressed / ~25 GB raw) ──
BASE_URL = "https://dumps.wikimedia.org/other/mediawiki_history/2026-03/enwiki"
DOWNLOAD_FILES = [
    "2026-03.enwiki.2022-01.tsv.bz2",   # 604 MB compressed / ~3.6 GB uncompressed (measured)
    "2026-03.enwiki.2023-01.tsv.bz2",   # 571 MB compressed / ~3.4 GB uncompressed (measured)
    "2026-03.enwiki.2024-01.tsv.bz2",   # 793 MB compressed / ~5.2 GB uncompressed (measured)
    # Total: ~1.97 GB compressed / ~12.2 GB uncompressed (exact, verified by bz2 stream decode)
]

# ── mediawiki_history TSV column order (76 columns) ───────────────────────────
# Source: Wikitech “Mediawiki_history dumps” technical documentation.
MW_COLUMNS = [
    "wiki_db", "event_entity", "event_type", "event_timestamp", "event_comment",
    "event_user_id", "event_user_central_id", "event_user_text_historical", "event_user_text",
    "event_user_blocks_historical", "event_user_blocks",
    "event_user_groups_historical", "event_user_groups",
    "event_user_is_bot_by_historical", "event_user_is_bot_by",
    "event_user_is_created_by_self", "event_user_is_created_by_system", "event_user_is_created_by_peer",
    "event_user_is_anonymous", "event_user_is_temporary", "event_user_is_permanent",
    "event_user_registration_timestamp", "event_user_creation_timestamp", "event_user_first_edit_timestamp",
    "event_user_revision_count", "event_user_seconds_since_previous_revision",
    "page_id", "page_title_historical", "page_title",
    "page_namespace_historical", "page_namespace_is_content_historical",
    "page_namespace", "page_namespace_is_content",
    "page_is_redirect", "page_is_deleted",
    "page_creation_timestamp", "page_first_edit_timestamp",
    "page_revision_count", "page_seconds_since_previous_revision",
    "user_id", "user_central_id", "user_text_historical", "user_text",
    "user_blocks_historical", "user_blocks",
    "user_groups_historical", "user_groups",
    "user_is_bot_by_historical", "user_is_bot_by",
    "user_is_created_by_self", "user_is_created_by_system", "user_is_created_by_peer",
    "user_is_anonymous", "user_is_temporary", "user_is_permanent",
    "user_registration_timestamp", "user_creation_timestamp", "user_first_edit_timestamp",
    "revision_id", "revision_parent_id", "revision_minor_edit",
    "revision_deleted_parts", "revision_deleted_parts_are_suppressed",
    "revision_text_bytes", "revision_text_bytes_diff", "revision_text_sha1",
    "revision_content_model", "revision_content_format",
    "revision_is_deleted_by_page_deletion", "revision_deleted_by_page_deletion_timestamp",
    "revision_is_identity_reverted", "revision_first_identity_reverting_revision_id",
    "revision_seconds_to_identity_revert", "revision_is_identity_revert",
    "revision_is_from_before_page_creation", "revision_tags",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download_files() -> list[Path]:
    """Download monthly TSV dumps, skip if already present."""
    downloaded = []
    for fname in DOWNLOAD_FILES:
        dest = DATA_DIR / fname
        url  = f"{BASE_URL}/{fname}"
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"  [SKIP] {fname}  ({_fmt_bytes(dest.stat().st_size)} already on disk)")
            downloaded.append(dest)
            continue

        print(f"  [DOWN] {fname}  from {url}")
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "WikiRisk/1.0 (academic; https://github.com/spartan/WikiRisk)"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp, \
                 open(dest, "wb") as fh:
                total = int(resp.headers.get("Content-Length", 0))
                done  = 0
                chunk = 1 << 20  # 1 MB
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    pct = (done / total * 100) if total else 0
                    elapsed = time.monotonic() - t0
                    speed = done / elapsed if elapsed > 0 else 0
                    print(
                        f"\r    {_fmt_bytes(done)}/{_fmt_bytes(total)}  "
                        f"({pct:.1f}%)  {_fmt_bytes(speed)}/s   ",
                        end="", flush=True,
                    )
            elapsed = time.monotonic() - t0
            print(f"\n    Done in {elapsed:.0f}s  ({_fmt_bytes(dest.stat().st_size)})")
            downloaded.append(dest)
        except Exception as exc:
            print(f"\n  [ERROR] {fname}: {exc}")
            if dest.exists():
                dest.unlink()
    return downloaded


# ─────────────────────────────────────────────────────────────────────────────
# 2. SPARK PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def build_spark():
    """Create a local SparkSession tuned for large TSV processing."""
    if "JAVA_HOME" not in os.environ:
        if sys.platform == "darwin":
            try:
                os.environ["JAVA_HOME"] = subprocess.check_output(
                    ["/usr/libexec/java_home"], text=True
                ).strip()
            except Exception:
                pass
        else:
            os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
    from pyspark.sql import SparkSession
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("WikiRisk-FullTraining")
        .config("spark.local.dir", str(SPARK_LOCAL))
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.sql.shuffle.partitions", "100")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def load_and_label(spark, files: list[Path]):
    """
    Read bz2 TSV files with Spark, filter to article-space revision creates,
    and derive label from revision_is_identity_reverted.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, StringType

    print(f"\n[SPARK] Reading {len(files)} bz2 TSV files...")
    paths = ["file://" + str(f.resolve()) for f in files]

    # All columns are strings in TSV; we'll cast what we need
    schema = StructType([StructField(c, StringType(), True) for c in MW_COLUMNS])

    raw = (
        spark.read
        .option("sep", "\t")
        .option("header", "false")
        .option("quote", "")           # TSV has no quoting
        .option("escape", "")
        .option("comment", "#")        # skip header comment lines
        .schema(schema)
        .csv(paths)                    # Spark reads bz2 natively
    )

    total_raw = raw.count()
    print(f"  Raw rows loaded: {total_raw:,}")

    # Filter to: revision creates on article namespace
    revisions = (
        raw
        .filter(F.col("event_entity") == "revision")
        .filter(F.col("event_type")   == "create")
        .filter(F.col("page_namespace") == "0")
        .filter(F.col("revision_is_identity_reverted").isNotNull())
    )

    rev_count = revisions.count()
    print(f"  Article-space revision creates: {rev_count:,}")

    # Engineer features
    labeled = (
        revisions
        # Label: was this edit later reverted by the community?
        .withColumn(
            "label",
            (F.col("revision_is_identity_reverted") == "true").cast("int"),
        )
        # Numeric features
        .withColumn(
            "length_delta",
            F.coalesce(
                F.col("revision_text_bytes_diff").cast("double"),
                F.lit(0.0),
            ),
        )
        .withColumn(
            "is_anon_int",
            (F.col("event_user_is_anonymous") == "true").cast("int"),
        )
        # Timestamp features
        .withColumn(
            "event_ts",
            F.to_timestamp("event_timestamp", "yyyy-MM-dd HH:mm:ss.S"),
        )
        .withColumn("hour_of_day",  F.hour("event_ts"))
        .withColumn("day_of_week",  F.dayofweek("event_ts"))
        # Text features (clean)
        .withColumn(
            "comment_clean",
            F.lower(
                F.regexp_replace(
                    F.coalesce(F.col("event_comment"), F.lit("")),
                    r"[^a-zA-Z0-9\s]", " ",
                )
            ),
        )
        .withColumn(
            "title_clean",
            F.lower(
                F.regexp_replace(
                    F.coalesce(F.col("page_title_historical"), F.lit("")),
                    r"[^a-zA-Z0-9\s]", " ",
                )
            ),
        )
        .select(
            "revision_id",
            "page_title_historical",
            "label",
            "length_delta",
            "is_anon_int",
            "hour_of_day",
            "day_of_week",
            "comment_clean",
            "title_clean",
        )
        .na.fill({
            "length_delta": 0.0,
            "is_anon_int":  0,
            "hour_of_day":  0,
            "day_of_week":  1,
            "comment_clean": "",
            "title_clean":   "",
        })
    )

    # Class distribution
    pos = labeled.filter(F.col("label") == 1).count()
    total = labeled.count()
    print(f"  Labeled rows: {total:,}  |  reverted (label=1): {pos:,}  ({100*pos/total:.1f}%)")

    return labeled, {"total": total, "positive": pos, "positive_rate": round(pos / total, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN SparkML
# ─────────────────────────────────────────────────────────────────────────────

def train_model(spark, df, label_stats: dict) -> tuple[object, dict]:
    """Train SparkML LogisticRegression with cross-validation."""
    import mlflow
    import mlflow.spark
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import (
        HashingTF, IDF, RegexTokenizer, StandardScaler, VectorAssembler,
    )
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

    print("\n[TRAIN] Building SparkML pipeline...")

    # ── Feature pipeline (same as src/ml/features.py) ──────────────────────
    comment_tok = RegexTokenizer(inputCol="comment_clean", outputCol="comment_tokens",
                                  pattern=r"\s+", toLowercase=True, minTokenLength=2)
    title_tok   = RegexTokenizer(inputCol="title_clean",   outputCol="title_tokens",
                                  pattern=r"\s+", toLowercase=True, minTokenLength=2)
    comment_tf  = HashingTF(inputCol="comment_tokens", outputCol="comment_tf",  numFeatures=256)
    title_tf    = HashingTF(inputCol="title_tokens",   outputCol="title_tf",    numFeatures=128)
    comment_idf = IDF(inputCol="comment_tf", outputCol="comment_idf", minDocFreq=5)
    title_idf   = IDF(inputCol="title_tf",   outputCol="title_idf",   minDocFreq=5)
    assembler   = VectorAssembler(
        inputCols=["comment_idf", "title_idf", "length_delta", "is_anon_int",
                   "hour_of_day", "day_of_week"],
        outputCol="raw_features", handleInvalid="keep",
    )
    scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                             withMean=False, withStd=True)
    lr     = LogisticRegression(
        featuresCol="features", labelCol="label",
        probabilityCol="probability", rawPredictionCol="raw_prediction",
    )

    pipeline = Pipeline(stages=[
        comment_tok, title_tok, comment_tf, title_tf,
        comment_idf, title_idf, assembler, scaler, lr,
    ])

    # ── Train / test split ──────────────────────────────────────────────────
    print("[TRAIN] Splitting data 80/20...")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_df.cache()
    test_df.cache()
    n_train = train_df.count()
    n_test  = test_df.count()
    print(f"  Train: {n_train:,}  |  Test: {n_test:,}")

    # ── Cross-validation grid ───────────────────────────────────────────────
    param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam,    [0.001, 0.01, 0.1])
        .addGrid(lr.maxIter,     [50, 100])
        .build()
    )
    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="raw_prediction",
        metricName="areaUnderROC",
    )
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        seed=42,
    )

    # ── MLflow run ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(str(MLFLOW_DIR))
    mlflow.set_experiment("wikirisk-edit-risk")

    print("[TRAIN] Starting cross-validation (this will take 30-60 minutes)...")
    t0 = time.monotonic()

    with mlflow.start_run(run_name="wikirisk-lr-mediawiki-history") as run:
        run_id = run.info.run_id
        mlflow.log_params({
            "model_type":        "LogisticRegression",
            "data_source":       "mediawiki_history",
            "label_field":       "revision_is_identity_reverted",
            "train_size":        n_train,
            "test_size":         n_test,
            "positive_rate":     label_stats["positive_rate"],
            "cv_folds":          3,
            "spark_version":     spark.version,
        })
        mlflow.log_params(label_stats)

        cv_model  = cv.fit(train_df)
        elapsed   = time.monotonic() - t0
        print(f"  CV done in {elapsed:.0f}s")

        best_model = cv_model.bestModel
        best_lr    = best_model.stages[-1]
        mlflow.log_params({
            "best_reg_param": best_lr.getRegParam(),
            "best_max_iter":  best_lr.getMaxIter(),
        })

        # ── Evaluate ────────────────────────────────────────────────────────
        print("[EVAL] Evaluating on test set...")
        preds = best_model.transform(test_df)
        auc   = evaluator.evaluate(preds)

        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
        mc_eval = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction",
        )
        f1        = mc_eval.setMetricName("f1").evaluate(preds)
        precision = mc_eval.setMetricName("weightedPrecision").evaluate(preds)
        recall    = mc_eval.setMetricName("weightedRecall").evaluate(preds)

        pr_eval = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="raw_prediction",
            metricName="areaUnderPR",
        )
        pr_auc = pr_eval.evaluate(preds)

        metrics = {
            "auc_roc":   round(auc,       4),
            "auc_pr":    round(pr_auc,    4),
            "f1":        round(f1,        4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "training_time_seconds": round(elapsed, 1),
        }
        mlflow.log_metrics(metrics)
        print(f"  AUC-ROC={auc:.4f}  AUC-PR={pr_auc:.4f}  F1={f1:.4f}")

        # ── Save model ──────────────────────────────────────────────────────
        print("[SAVE] Logging model to MLflow...")
        mlflow.spark.log_model(
            spark_model=best_model,
            artifact_path="spark-model",
            registered_model_name="wikirisk-risk-classifier",
        )

        # Also save to local path so streaming processor can load it
        model_path = str(MODEL_DIR / "wikirisk_lr_model")
        best_model.write().overwrite().save(model_path)
        print(f"  Model saved locally to: {model_path}")

        # ── Write manifest ──────────────────────────────────────────────────
        manifest = {
            "run_id":          run_id,
            "model_name":      "wikirisk-risk-classifier",
            "model_local_path": model_path,
            "trained_at":      datetime.now(tz=timezone.utc).isoformat(),
            "data_source":     "mediawiki_history",
            "label_field":     "revision_is_identity_reverted",
            "files_used":      DOWNLOAD_FILES,
            "metrics":         metrics,
            "training_data_stats": label_stats,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(f"  Manifest written to: {MANIFEST}")

    return best_model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4. UPDATE STREAMING PROCESSOR TO USE LOCAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

def re_score_existing_edits(model, model_path: str) -> None:
    """
    Re-score existing DB edits using the trained model so the dashboard
    immediately reflects real ML scores.
    """
    import sqlite3
    from pyspark.sql import SparkSession

    print("\n[RESCORE] Re-scoring existing DB edits with the trained model...")
    spark = SparkSession.getActiveSession()
    if spark is None:
        print("  No active Spark session, skipping re-score.")
        return

    db_path = str(ROOT / "data" / "wikirisk.db")
    from src.config import get_settings

    cfg = get_settings()
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, comment, page_title, length_delta, is_anon, timestamp FROM predictions"
    ).fetchall()

    if not rows:
        print("  No rows in DB.")
        conn.close()
        return

    from pyspark.ml import PipelineModel
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        DoubleType, IntegerType, StringType, StructField, StructType,
    )

    schema = StructType([
        StructField("id",           StringType(),  False),
        StructField("comment_clean", StringType(), True),
        StructField("title_clean",   StringType(), True),
        StructField("length_delta",  DoubleType(),  True),
        StructField("is_anon_int",   IntegerType(), True),
        StructField("hour_of_day",   IntegerType(), True),
        StructField("day_of_week",   IntegerType(), True),
    ])

    import re
    from datetime import datetime, timezone as tz

    records = []
    for r in rows:
        rid, comment, title, delta, is_anon, ts_str = r
        try:
            dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            hour = dt.hour
            dow  = dt.isoweekday()
        except Exception:
            hour, dow = 0, 1
        records.append((
            rid,
            re.sub(r"[^a-zA-Z0-9\s]", " ", (comment or "").lower()),
            re.sub(r"[^a-zA-Z0-9\s]", " ", (title   or "").lower()),
            float(delta or 0),
            int(is_anon or 0),
            hour, dow,
        ))

    df = spark.createDataFrame(records, schema=schema)
    loaded_model = PipelineModel.load(model_path)

    from pyspark.sql.functions import udf
    get_prob = udf(lambda v: float(v[1]) if v is not None else 0.5, DoubleType())
    scored = loaded_model.transform(df).withColumn("risk_score", get_prob(F.col("probability")))

    results = scored.select("id", "risk_score").toPandas()

    updates = []
    for _, row in results.iterrows():
        score = float(row["risk_score"])
        label = (
            "HIGH"
            if score >= cfg.risk_high_threshold
            else ("MEDIUM" if score >= cfg.risk_medium_threshold else "LOW")
        )
        updates.append((score, label, row["id"]))

    conn.executemany(
        "UPDATE predictions SET risk_score=?, risk_label=?, scored=1 WHERE id=?",
        updates,
    )
    conn.commit()
    conn.close()
    print(f"  Re-scored {len(updates)} existing edits with real ML model.")

    # Show distribution
    conn = sqlite3.connect(db_path)
    dist = conn.execute(
        "SELECT risk_label, COUNT(*) FROM predictions GROUP BY risk_label"
    ).fetchall()
    conn.close()
    print("  New distribution:", {r[0]: r[1] for r in dist})


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("WikiRisk Full Training Pipeline")
    print(f"Started: {datetime.now(tz=timezone.utc).isoformat()}")
    print("=" * 70)

    # ── Step 1: Download ────────────────────────────────────────────────────
    print("\n[STEP 1/4] Downloading Wikimedia mediawiki_history dumps...")
    files = download_files()
    if not files:
        print("ERROR: No files downloaded. Exiting.")
        sys.exit(1)

    total_compressed = sum(f.stat().st_size for f in files)
    print(f"\n  Total compressed size: {_fmt_bytes(total_compressed)}")
    print(f"  Estimated uncompressed (8-10x): {_fmt_bytes(total_compressed * 9)}")

    # ── Step 2: Spark load + label ──────────────────────────────────────────
    print("\n[STEP 2/4] Loading and labeling data with Spark...")
    spark = build_spark()
    df, label_stats = load_and_label(spark, files)

    # ── Step 3: Train ───────────────────────────────────────────────────────
    print("\n[STEP 3/4] Training SparkML LogisticRegression...")
    model, metrics = train_model(spark, df, label_stats)

    # ── Step 4: Re-score existing DB edits ─────────────────────────────────
    print("\n[STEP 4/4] Re-scoring existing dashboard edits with trained model...")
    model_path = str(MODEL_DIR / "wikirisk_lr_model")
    re_score_existing_edits(model, model_path)

    print("\n" + "=" * 70)
    print("Training COMPLETE")
    print(f"  AUC-ROC : {metrics['auc_roc']}")
    print(f"  AUC-PR  : {metrics['auc_pr']}")
    print(f"  F1      : {metrics['f1']}")
    print(f"  Model   : {model_path}")
    print(f"  Manifest: {MANIFEST}")
    print(f"Finished: {datetime.now(tz=timezone.utc).isoformat()}")
    print("=" * 70)
    print("\nNext step: restart the streaming processor — it will auto-load the model.")
    print("  python -m src.streaming.processor")
    spark.stop()


if __name__ == "__main__":
    main()
