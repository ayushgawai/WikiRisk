"""
WikiRisk – Model evaluation utilities.

Computes AUC, PR-AUC, F1, precision, recall for binary classification and
produces a confusion matrix artefact for MLflow logging.
"""
from __future__ import annotations

import json
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

        root = Path(__file__).resolve().parents[2]
        out_dir = root / "data" / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"confusion_matrix_{run_id}.json"
        out_path.write_text(json.dumps(output, indent=2))
        log.info("confusion_matrix_saved", tp=tp, fp=fp, tn=tn, fn=fn, path=str(out_path))
        return str(out_path)

    except Exception as exc:
        log.warning("confusion_matrix_failed", error=str(exc))
        return None


def load_production_model(cfg=None):
    """
    Load the trained SparkML model.

    Priority order:
      1. Local path saved by train_full_pipeline.py  (data/model/wikirisk_lr_model)
      2. MLflow Model Registry (Production → Staging → None stages)

    This means the model works even when MLflow server is not running.
    """
    from pyspark.ml import PipelineModel
    from pathlib import Path

    if cfg is None:
        from src.config import get_settings
        cfg = get_settings()

    # ── 1. Local model saved by full training pipeline ─────────────────────
    root = Path(__file__).resolve().parent.parent.parent
    local_path = root / "data" / "model" / "wikirisk_lr_model"
    if local_path.exists():
        try:
            model = PipelineModel.load(str(local_path))
            log.info("model_loaded_local", path=str(local_path))
            return model
        except Exception as exc:
            log.warning("local_model_load_failed", path=str(local_path), error=str(exc))

    # ── 2. MLflow Model Registry ───────────────────────────────────────────
    try:
        import mlflow.spark
    except Exception as exc:
        log.warning("mlflow_unavailable_for_model_load", error=str(exc))
    else:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

        for stage in ("Production", "Staging", "None"):
            model_uri = f"models:/{cfg.mlflow_model_name}/{stage}"
            try:
                model = mlflow.spark.load_model(model_uri)
                log.info("model_loaded_mlflow", uri=model_uri, stage=stage)
                return model
            except Exception as exc:
                log.debug("model_load_attempt_failed", stage=stage, error=str(exc))

    raise RuntimeError(
        f"No trained model found locally ({local_path}) or in MLflow registry "
        f"({cfg.mlflow_model_name}). Run scripts/train_full_pipeline.py first."
    )
