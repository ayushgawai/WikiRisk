"""
WikiRisk – Weak supervision labeller.

Creates binary "edit risk" labels (1 = risky, 0 = benign) from heuristics
applied to Wikipedia revision history.  These weak labels serve as
ground truth for supervised SparkML training.

Label = 1 (RISKY) if ANY of:
  • Comment contains revert/undo/vandal/spam/bot-revert keywords
  • Large content deletion  (length_delta < −500 bytes)
  • Anonymous user with large change (|delta| > 1 000 bytes)
  • Blank edit comment on article-space edit by anonymous user

Label = 0 (BENIGN) otherwise.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Keywords that strongly suggest the edit was reverted or is suspicious.
RISKY_COMMENT_PATTERNS = [
    r"(?i)\brevert",
    r"(?i)\brvv\b",
    r"(?i)\brv\b",               # common shorthand (e.g. "rv vandalism")
    r"(?i)\bvandaliz",
    r"(?i)\bvandalism\b",        # whole word (plain \bvandal\b misses "vandalism")
    r"(?i)\bvandal\b",
    r"(?i)\bspam\b",
    r"(?i)\bundo\b",
    r"(?i)\bundid\b",
    r"(?i)\btest edit",
    r"(?i)bot: reverting",
    r"(?i)\bcv\b",                 # copyright violation
    r"(?i)copyvio",
    r"(?i)off-?topic",
    r"(?i)not notable",
]


def _build_risky_comment_regex() -> str:
    return "|".join(RISKY_COMMENT_PATTERNS)


def apply_weak_labels(df: DataFrame, label_col: str = "label") -> DataFrame:
    """
    Apply heuristic-based weak labels to a Spark DataFrame of revisions.

    Expects columns: comment, is_anon_int (or is_anon), length_delta.
    """
    risky_regex = _build_risky_comment_regex()

    # ── Heuristic conditions ───────────────────────────────────────────────
    comment_col = F.col("comment") if "comment" in df.columns else F.col("comment_clean")
    is_anon_col = (
        F.col("is_anon_int").cast("boolean")
        if "is_anon_int" in df.columns
        else F.col("is_anon")
    )
    delta_col = F.col("length_delta")

    has_risky_comment = F.regexp_extract(comment_col, risky_regex, 0) != ""
    large_deletion = delta_col < -500
    anon_large_change = is_anon_col & (F.abs(delta_col) > 1000)
    anon_blank_comment = is_anon_col & (
        (comment_col.isNull()) | (F.trim(comment_col) == "")
    )

    label_expr = (
        has_risky_comment | large_deletion | anon_large_change | anon_blank_comment
    ).cast("int")

    return df.withColumn(label_col, label_expr)


def label_stats(df: DataFrame) -> dict:
    """Return dict with label distribution statistics."""
    from pyspark.sql.functions import count, when

    stats = df.agg(
        count("*").alias("total"),
        F.sum(F.col("label")).alias("positive"),
        (count("*") - F.sum(F.col("label"))).alias("negative"),
    ).first()

    if stats is None:
        return {}

    total = stats["total"] or 1
    return {
        "total": int(total),
        "positive": int(stats["positive"] or 0),
        "negative": int(stats["negative"] or 0),
        "positive_rate": round(int(stats["positive"] or 0) / total, 4),
    }
