"""
WikiRisk – Model evaluation utilities.

Computes AUC, PR-AUC, F1, precision, recall for binary classification and
produces a confusion matrix artefact for MLflow logging.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.common.logger import get_logger

log = get_logger(__name__)


def evaluate_model(model: PipelineModel, test_df: DataFrame) -> dict[str, float]:
    """
    Compute evaluation metrics on *test_df* using *model*.

    Returns a dict suitable for mlflow.log_metrics().
    """
    predictions = model.transform(test_df)

    binary_eval = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="raw_prediction",
    )

    auc = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderROC"})
    pr_auc = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderPR"})

    multi_eval = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
    )

    f1 = multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})
    precision = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"})
    recall = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"})
    accuracy = multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"})

    metrics = {
        "auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
    }

    log.info("evaluation_metrics", **metrics)
    return metrics


def log_confusion_matrix(
    predictions: DataFrame,
    run_id: str,
    label_col: str = "label",
    pred_col: str = "prediction",
) -> Optional[str]:
    """
    Compute confusion matrix and save as JSON artefact.

    Returns path to the temporary JSON file (caller should log to MLflow).
    """
    try:
        cm_df = (
            predictions
            .select(label_col, pred_col)
            .groupBy(label_col, pred_col)
            .count()
            .orderBy(label_col, pred_col)
        )

        cm_data = [
            {
                "actual": int(row[label_col]),
                "predicted": int(row[pred_col]),
                "count": int(row["count"]),
            }
            for row in cm_df.collect()
        ]

        # Derive standard TP/FP/TN/FN from cm_data
        tp = sum(r["count"] for r in cm_data if r["actual"] == 1 and r["predicted"] == 1)
        fp = sum(r["count"] for r in cm_data if r["actual"] == 0 and r["predicted"] == 1)
        tn = sum(r["count"] for r in cm_data if r["actual"] == 0 and r["predicted"] == 0)
        fn = sum(r["count"] for r in cm_data if r["actual"] == 1 and r["predicted"] == 0)

        output = {
            "run_id": run_id,
            "confusion_matrix": cm_data,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

        tmp = tempfile.mktemp(suffix=".json", prefix="confusion_matrix_")
        Path(tmp).write_text(json.dumps(output, indent=2))
        log.info("confusion_matrix_saved", tp=tp, fp=fp, tn=tn, fn=fn)
        return tmp

    except Exception as exc:
        log.warning("confusion_matrix_failed", error=str(exc))
        return None


def load_production_model(cfg=None):
    """
    Load the latest 'Production' stage model from MLflow Model Registry.
    Falls back to 'Staging' if no Production model exists.
    """
    import mlflow.spark

    if cfg is None:
        from src.config import get_settings
        cfg = get_settings()

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    for stage in ("Production", "Staging", "None"):
        model_uri = f"models:/{cfg.mlflow_model_name}/{stage}"
        try:
            model = mlflow.spark.load_model(model_uri)
            log.info("model_loaded", uri=model_uri, stage=stage)
            return model
        except Exception as exc:
            log.debug("model_load_attempt_failed", stage=stage, error=str(exc))

    raise RuntimeError(
        f"No trained model found in registry: {cfg.mlflow_model_name}. "
        "Run 'make train' first."
    )
