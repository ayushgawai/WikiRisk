"""
WikiRisk – SparkML model trainer.

Trains a Logistic Regression classifier on feature-engineered Wikipedia
revision data.  Full experiment is tracked via MLflow, including:
  • Parameters (regularisation, elastic net, iterations)
  • Metrics (AUC, PR-AUC, F1, precision, recall)
  • Artefacts (confusion matrix, feature pipeline, model)
  • Model registration in MLflow Model Registry

CLI:
    python -m src.ml.trainer [--run-name <name>]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import click
import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from src.common.logger import configure_logging, get_logger
from src.config import get_settings

log = get_logger(__name__)


def _build_training_pipeline(cfg) -> Pipeline:
    """Assemble feature pipeline + LogisticRegression into one Pipeline."""
    from src.ml.features import build_feature_pipeline

    feature_pipeline = build_feature_pipeline()
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=cfg.ml_max_iter,
        regParam=cfg.ml_reg_param,
        elasticNetParam=cfg.ml_elastic_net_param,
        probabilityCol="probability",
        rawPredictionCol="raw_prediction",
    )
    return Pipeline(stages=[*feature_pipeline.getStages(), lr])


def train(run_name: str | None = None) -> str:
    """
    Full training run.  Returns MLflow run_id.
    """
    cfg = get_settings()

    from src.common.spark_session import get_spark
    from src.ml.evaluator import evaluate_model, log_confusion_matrix
    from src.ml.labeler import label_stats

    spark = get_spark()

    # ── Load labeled data ──────────────────────────────────────────────────
    labeled_path = str(cfg.processed_data_dir / "labeled")
    log.info("loading_labeled_data", path=labeled_path)

    df = spark.read.parquet(labeled_path)
    n_records = df.count()

    if n_records < cfg.ml_min_records_required:
        raise ValueError(
            f"Need at least {cfg.ml_min_records_required} records, "
            f"found {n_records}"
        )

    stats = label_stats(df)
    log.info("label_distribution", **stats)

    # ── Train / test split ─────────────────────────────────────────────────
    train_df, test_df = df.randomSplit(
        [cfg.ml_train_test_split, 1 - cfg.ml_train_test_split],
        seed=42,
    )
    log.info(
        "data_split",
        train=train_df.count(),
        test=test_df.count(),
    )

    # ── MLflow setup ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    experiment = mlflow.set_experiment(cfg.mlflow_experiment_name)
    log.info(
        "mlflow_experiment",
        name=cfg.mlflow_experiment_name,
        id=experiment.experiment_id,
    )

    with mlflow.start_run(run_name=run_name or "wikirisk-lr") as run:
        run_id = run.info.run_id
        log.info("mlflow_run_started", run_id=run_id)

        # ── Log parameters ─────────────────────────────────────────────────
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "max_iter": cfg.ml_max_iter,
                "reg_param": cfg.ml_reg_param,
                "elastic_net_param": cfg.ml_elastic_net_param,
                "train_size": train_df.count(),
                "test_size": test_df.count(),
                "positive_rate": stats.get("positive_rate", "unknown"),
                "spark_version": spark.version,
            }
        )
        mlflow.log_params(stats)

        # ── Build pipeline ─────────────────────────────────────────────────
        pipeline = _build_training_pipeline(cfg)

        # ── Cross-validation ───────────────────────────────────────────────
        lr_stage = pipeline.getStages()[-1]
        param_grid = (
            ParamGridBuilder()
            .addGrid(lr_stage.regParam, [0.001, 0.01, 0.1])
            .addGrid(lr_stage.maxIter, [50, 100])
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
            numFolds=cfg.ml_num_folds,
            seed=42,
        )

        log.info("cross_validation_start", num_folds=cfg.ml_num_folds)
        t0 = time.monotonic()
        cv_model = cv.fit(train_df)
        train_elapsed = time.monotonic() - t0
        log.info("cross_validation_done", elapsed_seconds=round(train_elapsed, 1))

        # Best model
        best_model = cv_model.bestModel

        # ── Evaluate on test set ───────────────────────────────────────────
        metrics = evaluate_model(best_model, test_df)
        log.info("test_metrics", **metrics)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_time_seconds", round(train_elapsed, 1))

        # ── Confusion matrix ───────────────────────────────────────────────
        predictions = best_model.transform(test_df)
        cm_path = log_confusion_matrix(predictions, run_id=run_id)
        if cm_path:
            mlflow.log_artifact(cm_path, "evaluation")

        # ── Log best CV params ─────────────────────────────────────────────
        best_lr = best_model.stages[-1]
        mlflow.log_params(
            {
                "best_reg_param": best_lr.getRegParam(),
                "best_max_iter": best_lr.getMaxIter(),
            }
        )

        # ── Save model ─────────────────────────────────────────────────────
        mlflow.spark.log_model(
            spark_model=best_model,
            artifact_path="spark-model",
            registered_model_name=cfg.mlflow_model_name,
        )
        log.info(
            "model_logged",
            registered_name=cfg.mlflow_model_name,
            run_id=run_id,
        )

        # ── Write model manifest ───────────────────────────────────────────
        _write_model_manifest(cfg, run_id, metrics, stats)

    return run_id


def _write_model_manifest(cfg, run_id: str, metrics: dict, stats: dict) -> None:
    from datetime import datetime, timezone

    manifest = {
        "run_id": run_id,
        "model_name": cfg.mlflow_model_name,
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "mlflow_tracking_uri": cfg.mlflow_tracking_uri,
        "metrics": metrics,
        "training_data_stats": stats,
    }
    path = cfg.manifests_dir / "model_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    log.info("model_manifest_written", path=str(path))


@click.command()
@click.option("--run-name", default=None, help="MLflow run name")
def main(run_name: str | None) -> None:
    """Train WikiRisk SparkML model and register in MLflow."""
    cfg = get_settings()
    configure_logging(cfg.log_level, cfg.log_format)
    run_id = train(run_name=run_name)
    log.info("training_complete", run_id=run_id)


if __name__ == "__main__":
    main()
