#!/usr/bin/env python3
"""Lightweight demo scorer for WikiRisk.

This process keeps the live demo table fresh by scoring newly inserted raw
rows in the SQLite predictions store every few seconds. It is intentionally
simple and stable for demo use when the Spark streaming processor is too
fragile on the local machine.
"""
from __future__ import annotations

import re
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import get_settings
from src.common.logger import configure_logging, get_logger

log = get_logger(__name__)
_RUNNING = True


RISKY_COMMENT_PATTERNS = [
    r"(?i)\brevert",
    r"(?i)\brvv\b",
    r"(?i)\brv\b",
    r"(?i)\bvandaliz",
    r"(?i)\bvandalism\b",
    r"(?i)\bvandal\b",
    r"(?i)\bspam\b",
    r"(?i)\bundo\b",
    r"(?i)\bundid\b",
    r"(?i)\btest edit",
    r"(?i)bot: reverting",
    r"(?i)\bcv\b",
    r"(?i)copyvio",
    r"(?i)off-?topic",
    r"(?i)not notable",
]


def _handle_signal(signum, frame):  # noqa: ARG001
    global _RUNNING
    log.info("shutdown_signal_received", signum=signum)
    _RUNNING = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def _score_row(row: sqlite3.Row, cfg) -> tuple[float, str]:
    comment = (row["comment"] or "").strip()
    user = (row["user"] or "").strip()
    length_delta = float(row["length_delta"] or 0.0)
    is_anon = int(row["is_anon"] or 0)

    score = 0.05
    if is_anon:
        score += 0.18
    if length_delta < -500:
        score += 0.22
    if abs(length_delta) > 1000:
        score += 0.10
    if is_anon and not comment:
        score += 0.10
    if user.lower().startswith("bot"):
        score -= 0.03

    for pattern in RISKY_COMMENT_PATTERNS:
        if re.search(pattern, comment):
            score += 0.25
            break

    score = max(0.0, min(score, 0.99))
    if score >= cfg.risk_high_threshold:
        label = "HIGH"
    elif score >= cfg.risk_medium_threshold:
        label = "MEDIUM"
    else:
        label = "LOW"
    return score, label


def _process_pending_rows(conn: sqlite3.Connection, cfg) -> int:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, user, is_anon, comment, length_delta
        FROM predictions
        WHERE scored = 0
        ORDER BY created_at ASC
        LIMIT 1
        """
    ).fetchall()

    if not rows:
        return 0

    now_iso = datetime.now(tz=timezone.utc).isoformat()
    updates = []
    for row in rows:
        score, label = _score_row(row, cfg)
        updates.append((score, label, now_iso, row["id"]))

    conn.executemany(
        """
        UPDATE predictions
        SET risk_score = ?,
            risk_label = ?,
            scored = 1,
            created_at = ?
        WHERE id = ?
        """,
        updates,
    )
    conn.commit()
    log.info("demo_rows_scored", rows=len(updates))
    return len(updates)


def main() -> int:
    cfg = get_settings()
    configure_logging(cfg.log_level, cfg.log_format)
    cfg.ensure_dirs()

    db_path = Path(cfg.db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    log.info("demo_scoring_started", db=str(db_path))

    while _RUNNING:
        try:
            count = _process_pending_rows(conn, cfg)
            if count == 0:
                time.sleep(3)
            else:
                time.sleep(3)
        except Exception as exc:
            log.warning("demo_scoring_error", error=str(exc))
            time.sleep(3)

    conn.close()
    log.info("demo_scoring_stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
