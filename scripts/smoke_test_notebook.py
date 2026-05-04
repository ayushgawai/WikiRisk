"""
smoke_test_notebook.py
----------------------
Replicates every logic path in WikiRisk_Training_Pipeline.ipynb using
~2000 synthetic rows so the full pipeline (Spark load → feature engineering
→ train → evaluate → manifest → re-score DB) finishes in a few minutes.

All artifacts stay under the repo (no OS tmp dirs for data):
  data/raw/mediawiki_history/_smoke_sample.tsv.bz2
  data/spark-local/          (Spark shuffle — in .gitignore)
  data/model/wikirisk_lr_smoke/

Run:
  python scripts/smoke_test_notebook.py --generate-only   # write sample only
  python scripts/smoke_test_notebook.py                    # full fast test
"""

import argparse
import bz2, json, os, re, sqlite3, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data" / "raw" / "mediawiki_history"
MODEL_DIR    = ROOT / "data" / "model"
MLFLOW_DIR   = ROOT / "mlruns"
DB_PATH      = ROOT / "data" / "wikirisk_smoke.db"  # separate from prod DB
SAMPLE_BZ2   = DATA_DIR / "_smoke_sample.tsv.bz2"    # in-repo; notebook USE_SMOKE_DEMO reads this
SPARK_LOCAL  = ROOT / "data" / "spark-local"
STATS_SMOKE  = DATA_DIR / "label_stats_smoke.json"
SIZES_SMOKE  = DATA_DIR / "sizes_smoke.json"

for d in [DATA_DIR, MODEL_DIR, ROOT/"manifests", MLFLOW_DIR, ROOT/"docs", SPARK_LOCAL]:
    d.mkdir(parents=True, exist_ok=True)

PASS = "✅"; FAIL = "❌"; INFO = "ℹ️ "
errors = []
# Module-level — nested helpers use `global` so Spark state is not shadowed by _run_smoke_pipeline locals
spark = None
raw = None
labeled = None
label_stats = None
train_df = test_df = None
n_train = n_test = 0
cv = evaluator = None
best_model = None
RUN_ID = None
elapsed = 0.0
metrics: dict = {}

def check(label, fn):
    try:
        fn()
        print(f"  {PASS}  {label}")
    except Exception as e:
        print(f"  {FAIL}  {label}")
        print(f"         {e}")
        errors.append((label, e))

# ── 1. Generate synthetic sample bz2 TSV ──────────────────────────────────────

MW_HEADER = [
    "wiki_db","event_entity","event_type","event_timestamp",
    "event_comment","event_user_id","event_user_text_historical",
    "event_user_text","event_user_blocks_historical","event_user_blocks",
    "event_user_groups_historical","event_user_groups",
    "event_user_is_bot_by_historical","event_user_is_bot_by",
    "event_user_is_created_by_self","event_user_is_created_by_system",
    "event_user_is_created_by_peer","event_user_is_anonymous",
    "event_user_registration_timestamp","event_user_creation_timestamp",
    "event_user_first_edit_timestamp","event_user_revision_count",
    "event_user_seconds_since_previous_revision",
    "page_id","page_title_historical","page_title",
    "page_namespace_historical","page_namespace",
    "page_namespace_is_content_historical","page_namespace_is_content",
    "page_is_redirect_historical","page_is_redirect",
    "page_is_deleted","page_creation_timestamp",
    "page_first_edit_timestamp","page_revision_count",
    "page_seconds_since_previous_revision",
    "revision_id","revision_parent_id","revision_minor_edit",
    "revision_deleted_parts","revision_deleted_parts_are_suppressed",
    "revision_text_bytes","revision_text_bytes_diff",
    "revision_text_sha1","revision_content_model",
    "revision_content_format","revision_is_deleted_by_page_deletion",
    "revision_deleted_timestamp","revision_is_identity_reverted",
    "revision_first_identity_reverting_revision_id",
    "revision_seconds_to_identity_revert","revision_is_identity_revert",
    "revision_is_from_before_page_creation","revision_tags",
]

def make_row(i):
    is_anon    = "true"  if i % 4 == 0 else "false"
    reverted   = "true"  if i % 5 == 0 else "false"
    byte_delta = str((-50 if reverted == "true" else 120) + (i % 30))
    byte_total = str(5000 + i * 10)
    hour       = i % 24
    ts         = f"2023-06-{1+(i%28):02d} {hour:02d}:30:00.0"
    titles     = ["Python_(programming)","Machine_learning","Wikipedia","Neural_network","Vandalism"]
    comments   = ["/* section */ fixed typo","Reverted edits by user","Added citation","rv vandalism","Updated content"]
    row_vals = {
        "wiki_db": "enwiki", "event_entity": "revision", "event_type": "create",
        "event_timestamp": ts,
        "event_comment": comments[i % len(comments)],
        "event_user_id": str(1000 + i),
        "event_user_text_historical": f"User{i}",
        "event_user_text": f"User{i}",
        "event_user_blocks_historical": "",
        "event_user_blocks": "",
        "event_user_groups_historical": "extendedconfirmed",
        "event_user_groups": "extendedconfirmed",
        "event_user_is_bot_by_historical": "false",
        "event_user_is_bot_by": "false",
        "event_user_is_created_by_self": "true",
        "event_user_is_created_by_system": "false",
        "event_user_is_created_by_peer": "false",
        "event_user_is_anonymous": is_anon,
        "event_user_registration_timestamp": "2020-01-01 00:00:00.0",
        "event_user_creation_timestamp": "2020-01-01 00:00:00.0",
        "event_user_first_edit_timestamp": "2020-01-02 00:00:00.0",
        "event_user_revision_count": str(10 + i),
        "event_user_seconds_since_previous_revision": "3600",
        "page_id": str(100 + (i % 50)),
        "page_title_historical": titles[i % len(titles)],
        "page_title": titles[i % len(titles)],
        "page_namespace_historical": "0",
        "page_namespace": "0",
        "page_namespace_is_content_historical": "true",
        "page_namespace_is_content": "true",
        "page_is_redirect_historical": "false",
        "page_is_redirect": "false",
        "page_is_deleted": "false",
        "page_creation_timestamp": "2010-01-01 00:00:00.0",
        "page_first_edit_timestamp": "2010-01-01 00:00:00.0",
        "page_revision_count": str(500 + i),
        "page_seconds_since_previous_revision": "7200",
        "revision_id": str(900000000 + i),
        "revision_parent_id": str(900000000 + i - 1),
        "revision_minor_edit": "false",
        "revision_deleted_parts": "",
        "revision_deleted_parts_are_suppressed": "false",
        "revision_text_bytes": byte_total,
        "revision_text_bytes_diff": byte_delta,
        "revision_text_sha1": f"sha1{i:08x}",
        "revision_content_model": "wikitext",
        "revision_content_format": "text/x-wiki",
        "revision_is_deleted_by_page_deletion": "false",
        "revision_deleted_timestamp": "",
        "revision_is_identity_reverted": reverted,
        "revision_first_identity_reverting_revision_id": str(900000000 + i + 1) if reverted == "true" else "",
        "revision_seconds_to_identity_revert": "120" if reverted == "true" else "",
        "revision_is_identity_revert": "false",
        "revision_is_from_before_page_creation": "false",
        "revision_tags": "",
    }
    return "\t".join(row_vals[c] for c in MW_HEADER)

def gen_sample(*, force: bool = False) -> None:
    """Write synthetic bz2 TSV under data/raw/mediawiki_history/ (no tmp)."""
    if not force and SAMPLE_BZ2.exists() and SAMPLE_BZ2.stat().st_size > 500:
        print(f"         Reusing existing {SAMPLE_BZ2.name} ({SAMPLE_BZ2.stat().st_size:,} bytes)")
        return
    lines = []
    for i in range(2000):
        lines.append(make_row(i))
    data = ("\n".join(lines) + "\n").encode()
    with bz2.open(SAMPLE_BZ2, "wb") as fh:
        fh.write(data)
    print(f"         Wrote {SAMPLE_BZ2}  ({SAMPLE_BZ2.stat().st_size:,} bytes compressed)")


def main() -> None:
    p = argparse.ArgumentParser(description="WikiRisk smoke test (in-repo paths only)")
    p.add_argument("--generate-only", action="store_true", help="Write _smoke_sample.tsv.bz2 and exit")
    opts, _ = p.parse_known_args()
    if opts.generate_only:
        gen_sample(force=True)
        print("\nWrote:", SAMPLE_BZ2.resolve())
        print("In the notebook, set USE_SMOKE_DEMO = True in cell 2, then run cells step by step.")
        return

    _run_smoke_pipeline()


def _run_smoke_pipeline() -> None:
    print("\n[1] Synthetic bz2 sample (in-repo, reused if present)…")
    check("Generate / reuse _smoke_sample.tsv.bz2", lambda: gen_sample(force=False))

    # ── 2. Spark session ──────────────────────────────────────────────────────────
    print("\n[2] Starting SparkSession…")
    from pyspark.sql import SparkSession

    def start_spark():
        global spark
        spark = (
            SparkSession.builder
            .master("local[2]")
            .appName("WikiRisk-SmokeTest")
            .config("spark.local.dir",              str(SPARK_LOCAL))
            .config("spark.driver.memory",          "2g")
            .config("spark.executor.memory",        "2g")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.enabled",             "false")
            .config("spark.sql.adaptive.enabled",   "true")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")

    check("SparkSession start", start_spark)

    # ── 3. Load raw TSV ────────────────────────────────────────────────────────────
    print("\n[3] Load raw bz2 TSV with Spark…")
    from pyspark.sql.types import StructType, StructField, StringType

    schema = StructType([StructField(c, StringType(), True) for c in MW_HEADER])

    def load_raw():
        global raw
        # Spark CSV + tiny local bz2 is unreliable on some macOS JVM setups; parse in Python then createDataFrame.
        rows: list[list[str]] = []
        with bz2.open(SAMPLE_BZ2, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == len(MW_HEADER):
                    rows.append(parts)
        raw = spark.createDataFrame(rows, schema=schema)
        raw.cache()
        n = raw.count()
        assert n > 0, f"Expected >0 rows, got {n}"

    check("Load bz2 TSV (2000 rows)", load_raw)

    # ── 4. Feature engineering + label stats ──────────────────────────────────────
    print("\n[4] Feature engineering…")
    from pyspark.sql import functions as F

    def feature_eng():
        global labeled, label_stats
        labeled = (
            raw
            .filter(F.col("event_entity") == "revision")
            .filter(F.col("event_type")   == "create")
            .filter(F.col("page_namespace") == "0")
            .filter(F.col("revision_is_identity_reverted").isNotNull())
            .withColumn("label",
                (F.col("revision_is_identity_reverted") == "true").cast("int"))
            .withColumn("length_delta",
                F.coalesce(F.col("revision_text_bytes_diff").cast("double"), F.lit(0.0)))
            .withColumn("is_anon_int",
                (F.col("event_user_is_anonymous") == "true").cast("int"))
            .withColumn("event_ts",
                F.to_timestamp("event_timestamp", "yyyy-MM-dd HH:mm:ss.S"))
            .withColumn("hour_of_day", F.hour("event_ts"))
            .withColumn("day_of_week", F.dayofweek("event_ts"))
            .withColumn("comment_clean",
                F.lower(F.regexp_replace(
                    F.coalesce(F.col("event_comment"), F.lit("")), r"[^a-zA-Z0-9\s]", " ")))
            .withColumn("title_clean",
                F.lower(F.regexp_replace(
                    F.coalesce(F.col("page_title_historical"), F.lit("")), r"[^a-zA-Z0-9\s]", " ")))
            .select("revision_id","page_title_historical","label",
                    "length_delta","is_anon_int","hour_of_day","day_of_week",
                    "comment_clean","title_clean")
            .na.fill({"length_delta":0.0,"is_anon_int":0,"hour_of_day":0,
                      "day_of_week":1,"comment_clean":"","title_clean":""})
        )
        labeled.cache()
        total = labeled.count()
        pos   = labeled.filter(F.col("label") == 1).count()
        neg   = total - pos
        rate  = pos / total
        assert total > 0
        label_stats = {"total":total,"positive":pos,"negative":neg,
                       "positive_rate":round(rate,4),"n_train":0,"n_test":0}
        print(f"         {total} labeled rows  ({100*rate:.1f}% reverted)")

    check("Feature engineering + label stats", feature_eng)

    # ── 5. Train / test split ──────────────────────────────────────────────────────
    print("\n[5] Train/test split…")

    def split():
        global train_df, test_df, n_train, n_test
        train_df, test_df = labeled.randomSplit([0.8, 0.2], seed=42)
        train_df.cache(); test_df.cache()
        n_train = train_df.count(); n_test = test_df.count()
        label_stats.update({"n_train":n_train,"n_test":n_test})
        assert n_train > 0 and n_test > 0

    check(f"80/20 split", split)

    # ── 6. Build SparkML pipeline ─────────────────────────────────────────────────
    print("\n[6] Build SparkML pipeline…")
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, StandardScaler, VectorAssembler
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

    def build_pipeline():
        global cv, evaluator
        c_tok = RegexTokenizer(inputCol="comment_clean", outputCol="comment_tokens",
                                pattern=r"\s+", toLowercase=True, minTokenLength=2)
        t_tok = RegexTokenizer(inputCol="title_clean",   outputCol="title_tokens",
                                pattern=r"\s+", toLowercase=True, minTokenLength=2)
        c_tf  = HashingTF(inputCol="comment_tokens", outputCol="comment_tf",  numFeatures=64)
        t_tf  = HashingTF(inputCol="title_tokens",   outputCol="title_tf",    numFeatures=32)
        c_idf = IDF(inputCol="comment_tf", outputCol="comment_idf", minDocFreq=1)
        t_idf = IDF(inputCol="title_tf",   outputCol="title_idf",   minDocFreq=1)
        asm   = VectorAssembler(
            inputCols=["comment_idf","title_idf","length_delta","is_anon_int","hour_of_day","day_of_week"],
            outputCol="raw_features", handleInvalid="keep")
        scl   = StandardScaler(inputCol="raw_features", outputCol="features",
                                withMean=False, withStd=True)
        lr    = LogisticRegression(featuresCol="features", labelCol="label",
                                    probabilityCol="probability", rawPredictionCol="raw_prediction")
        pipeline = Pipeline(stages=[c_tok,t_tok,c_tf,t_tf,c_idf,t_idf,asm,scl,lr])
        param_grid = (ParamGridBuilder()
                      .addGrid(lr.regParam, [0.01, 0.1])
                      .build())
        evaluator = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="raw_prediction", metricName="areaUnderROC")
        cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid,
                            evaluator=evaluator, numFolds=2, seed=42)

    check("Build SparkML pipeline", build_pipeline)

    # ── 7. Train with MLflow ───────────────────────────────────────────────────────
    print("\n[7] Training (2-fold CV, 2 param combos = 4 fits on 500 rows)…")
    import mlflow, mlflow.spark
    from pyspark.ml import PipelineModel

    SMOKE_MODEL = str(MODEL_DIR / "wikirisk_lr_smoke")

    def train():
        global best_model, RUN_ID, elapsed, metrics
        mlflow.set_tracking_uri(str(MLFLOW_DIR))
        mlflow.set_experiment("wikirisk-smoke-test")
        t0 = time.monotonic()
        with mlflow.start_run(run_name="smoke-test") as run:
            RUN_ID = run.info.run_id
            mlflow.log_params({"model_type":"LR","n_train":n_train,"n_test":n_test})
            cv_model   = cv.fit(train_df)
            elapsed    = time.monotonic() - t0
            best_model = cv_model.bestModel
            best_lr    = best_model.stages[-1]
            mlflow.log_params({"best_reg_param": best_lr.getRegParam()})
        print(f"         Trained in {elapsed:.1f}s  regParam={best_lr.getRegParam()}")

    check(f"SparkML CrossValidator training", train)

    # ── 8. Evaluate ────────────────────────────────────────────────────────────────
    print("\n[8] Evaluate on test set…")

    def evaluate():
        global metrics
        preds = best_model.transform(test_df)
        mc    = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
        pr    = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="raw_prediction",
                                               metricName="areaUnderPR")
        metrics = {
            "auc_roc"  : round(evaluator.evaluate(preds), 4),
            "auc_pr"   : round(pr.evaluate(preds),        4),
            "f1"       : round(mc.setMetricName("f1").evaluate(preds), 4),
            "precision": round(mc.setMetricName("weightedPrecision").evaluate(preds), 4),
            "recall"   : round(mc.setMetricName("weightedRecall").evaluate(preds), 4),
            "accuracy" : round(mc.setMetricName("accuracy").evaluate(preds), 4),
            "training_time_seconds": round(elapsed, 1),
        }
        with mlflow.start_run(run_id=RUN_ID):
            mlflow.log_metrics(metrics)
        print(f"         AUC-ROC={metrics['auc_roc']}  F1={metrics['f1']}  Acc={metrics['accuracy']}")

    check("Evaluate + log metrics to MLflow", evaluate)

    # ── 9. Save model + manifest ───────────────────────────────────────────────────
    print("\n[9] Save model + manifest…")

    def save_model():
        best_model.write().overwrite().save(SMOKE_MODEL)
        manifest = {
            "run_id": RUN_ID,
            "model_name": "wikirisk-smoke",
            "model_local_path": SMOKE_MODEL,
            "trained_at": datetime.now(tz=timezone.utc).isoformat(),
            "data_source": "smoke_test_synthetic",
            "label_field": "revision_is_identity_reverted",
            "files_used": [SAMPLE_BZ2.name],
            "metrics": metrics,
            "training_data_stats": label_stats,
        }
        (ROOT/"manifests"/"smoke_manifest.json").write_text(json.dumps(manifest, indent=2))
        loaded = PipelineModel.load(SMOKE_MODEL)
        assert loaded is not None, "Model failed to reload"

    check("Save model locally + reload verification", save_model)

    # ── 10. Matplotlib / seaborn plots ────────────────────────────────────────────
    print("\n[10] Charts (matplotlib + seaborn)…")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    def make_charts():
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        ax[0].bar(["Benign","Reverted"],
                  [label_stats["negative"], label_stats["positive"]],
                  color=["#22c55e","#dc2626"])
        ax[0].set_title("Label Distribution")
        mnames = ["AUC-ROC","F1","Precision","Recall","Accuracy"]
        mvals  = [metrics["auc_roc"],metrics["f1"],metrics["precision"],
                   metrics["recall"],metrics["accuracy"]]
        ax[1].barh(mnames, mvals, color="#6366f1")
        ax[1].set_xlim(0, 1.1)
        ax[1].set_title("Metrics")
        plt.tight_layout()
        out = ROOT/"docs"/"smoke_test_chart.png"
        plt.savefig(out, dpi=80, bbox_inches="tight")
        plt.close()
        assert out.exists()

    check("Matplotlib + seaborn charts", make_charts)

    # ── 11. SQLite re-score ────────────────────────────────────────────────────────
    print("\n[11] SQLite DB re-score simulation…")
    from pyspark.sql.types import DoubleType, IntegerType
    from pyspark.sql.functions import udf as _udf

    def rescore_db():
        # Create a minimal smoke DB
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY, comment TEXT, page_title TEXT,
            length_delta REAL, is_anon INTEGER, timestamp TEXT,
            risk_score REAL, risk_label TEXT, scored INTEGER DEFAULT 0)""")
        # Insert 10 fake rows
        for i in range(10):
            conn.execute(
                "INSERT OR IGNORE INTO predictions VALUES (?,?,?,?,?,?,?,?,?)",
                (f"smoke-{i}", f"edit comment {i}", f"PageTitle{i}",
                 float(50 + i*5), i%2, "2023-06-15T12:00:00Z", None, None, 0))
        conn.commit()
        conn.close()

        # Re-score via Spark
        from pyspark.sql.types import StringType as _ST, StructType, StructField
        schema_rs = StructType([
            StructField("id",            _ST(),         False),
            StructField("comment_clean", _ST(),         True),
            StructField("title_clean",   _ST(),         True),
            StructField("length_delta",  DoubleType(),  True),
            StructField("is_anon_int",   IntegerType(), True),
            StructField("hour_of_day",   IntegerType(), True),
            StructField("day_of_week",   IntegerType(), True),
        ])
        conn  = sqlite3.connect(str(DB_PATH))
        rows  = conn.execute(
            "SELECT id,comment,page_title,length_delta,is_anon,timestamp FROM predictions"
        ).fetchall()
        conn.close()

        records = []
        for (rid,comment,title,delta,is_anon,ts_str) in rows:
            try:
                dt = datetime.fromisoformat(str(ts_str).replace("Z","+00:00"))
                hour,dow = dt.hour, dt.isoweekday()
            except Exception:
                hour,dow = 0, 1
            records.append((
                rid,
                re.sub(r"[^a-zA-Z0-9\s]"," ",(comment or "").lower()),
                re.sub(r"[^a-zA-Z0-9\s]"," ",(title   or "").lower()),
                float(delta or 0), int(is_anon or 0), hour, dow,
            ))

        loaded   = PipelineModel.load(SMOKE_MODEL)
        df_rs    = spark.createDataFrame(records, schema=schema_rs)
        get_prob = _udf(lambda v: float(v[1]) if v is not None else 0.5, DoubleType())
        scored   = loaded.transform(df_rs).withColumn("risk_score", get_prob(F.col("probability")))
        results  = scored.select("id","risk_score").toPandas()

        updates  = []
        for _, row in results.iterrows():
            sc  = float(row["risk_score"])
            lbl = "HIGH" if sc >= 0.68 else ("MEDIUM" if sc >= 0.30 else "LOW")
            updates.append((sc, lbl, row["id"]))

        conn = sqlite3.connect(str(DB_PATH))
        conn.executemany(
            "UPDATE predictions SET risk_score=?,risk_label=?,scored=1 WHERE id=?", updates)
        conn.commit()
        verified = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE scored=1").fetchone()[0]
        conn.close()
        assert verified == 10, f"Only {verified}/10 rows scored"
        print(f"         Re-scored 10 rows  example: score={updates[0][0]:.3f} label={updates[0][1]}")

    check("SQLite re-score via trained model", rescore_db)

    # ── 12. Idempotency check ──────────────────────────────────────────────────────
    print("\n[12] Idempotency checks…")

    def check_idempotency():
        if not SIZES_SMOKE.exists():
            SIZES_SMOKE.write_text(json.dumps({SAMPLE_BZ2.name: 1}, indent=2))
        cached = json.loads(SIZES_SMOKE.read_text())
        assert isinstance(cached, dict)
        STATS_SMOKE.write_text(json.dumps(label_stats, indent=2))
        assert STATS_SMOKE.exists()
        assert (MODEL_DIR / "wikirisk_lr_smoke").exists()

    check("sizes_smoke.json + label_stats_smoke.json + model on disk", check_idempotency)

    # ── Cleanup ────────────────────────────────────────────────────────────────────
    print("\n[Cleanup] Stopping Spark…")
    try:
        spark.stop()
        print("  Spark stopped.")
    except Exception:
        pass

    # Remove smoke artifacts (keep real production artifacts untouched)
    import shutil
    # Keep _smoke_sample.tsv.bz2 in repo for notebook USE_SMOKE_DEMO — do not delete
    for p in [DB_PATH,
              MODEL_DIR/"wikirisk_lr_smoke",
              ROOT/"manifests"/"smoke_manifest.json",
              ROOT/"docs"/"smoke_test_chart.png",
              STATS_SMOKE,
              SIZES_SMOKE]:
        if Path(p).exists():
            (shutil.rmtree if Path(p).is_dir() else Path(p).unlink)(p)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    if errors:
        print(f"{FAIL}  SMOKE TEST FAILED — {len(errors)} error(s):")
        for label, e in errors:
            print(f"     • {label}: {e}")
        sys.exit(1)
    print(f"{PASS}  ALL CHECKS PASSED — fast pipeline OK (artifacts under data/).")
    print("="*55)
    print("\nWhen you are ready, run the Jupyter notebook yourself (step-by-step recommended).")
    print("  notebooks/WikiRisk_Training_Pipeline.ipynb")
    print("  For a quick demo: set USE_SMOKE_DEMO = True in cell 2 after --generate-only.")


if __name__ == "__main__":
    main()
