"""
WikiRisk – Async SQLite database layer (aiosqlite).

Provides a thin async interface over the predictions SQLite database.
The database is shared between:
  • streaming/collector.py  (inserts raw events)
  • streaming/processor.py  (updates with risk scores)
  • serving/main.py          (reads via this module)
"""
from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import aiosqlite

from src.common.logger import get_logger
from src.config import get_settings

log = get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id            TEXT PRIMARY KEY,
    rev_id        TEXT,
    page_title    TEXT,
    namespace     INTEGER DEFAULT 0,
    user          TEXT,
    is_anon       INTEGER DEFAULT 0,
    comment       TEXT,
    length_delta  REAL DEFAULT 0,
    timestamp     TEXT,
    wiki          TEXT DEFAULT 'enwiki',
    risk_score    REAL,
    risk_label    TEXT,
    scored        INTEGER DEFAULT 0,
    created_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_created_at ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_score  ON predictions (risk_score   DESC);
CREATE INDEX IF NOT EXISTS idx_risk_label  ON predictions (risk_label);

CREATE TABLE IF NOT EXISTS explanations (
    edit_id       TEXT PRIMARY KEY,
    explanation   TEXT,
    model         TEXT,
    generated_at  TEXT
);

"""


async def init_db(db_path: str) -> None:
    """Create tables if they don't exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        for stmt in _DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                await db.execute(stmt)
        await db.commit()
    log.info("database_initialised", path=db_path)


@asynccontextmanager
async def get_db() -> AsyncIterator[aiosqlite.Connection]:
    """Async context manager that yields a DB connection."""
    cfg = get_settings()
    async with aiosqlite.connect(cfg.db_path) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def fetch_recent_edits(
    page: int = 1,
    page_size: int = 50,
    risk_label: Optional[str] = None,
    scored_only: bool = False,
    wiki: str = "enwiki",
    search: Optional[str] = None,
) -> tuple[list[dict], int]:
    """
    Return (items, total_count) for the recent edits query.
    """
    offset = (page - 1) * page_size
    conditions = ["wiki = ?"]
    params: list = [wiki]

    if risk_label:
        conditions.append("risk_label = ?")
        params.append(risk_label.upper())

    if scored_only:
        conditions.append("scored = 1")

    if search:
        conditions.append(
            "(page_title LIKE ? OR comment LIKE ? OR user LIKE ? OR rev_id LIKE ?)"
        )
        pattern = f"%{search.strip()}%"
        params.extend([pattern, pattern, pattern, pattern])

    where = "WHERE " + " AND ".join(conditions)

    async with get_db() as db:
        cur = await db.execute(
            f"SELECT COUNT(*) FROM predictions {where}", params
        )
        row = await cur.fetchone()
        total = row[0] if row else 0

        cur = await db.execute(
            f"""
            SELECT * FROM predictions
            {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            [*params, page_size, offset],
        )
        rows = await cur.fetchall()
        items = [dict(r) for r in rows]

    return items, total


async def fetch_edit_by_id(edit_id: str) -> Optional[dict]:
    async with get_db() as db:
        cur = await db.execute(
            "SELECT * FROM predictions WHERE id = ?", [edit_id]
        )
        row = await cur.fetchone()
        return dict(row) if row else None


async def fetch_stats() -> dict:
    async with get_db() as db:
        cur = await db.execute(
            """
            SELECT
                COUNT(*)                                   AS total_edits,
                SUM(CASE WHEN risk_label='HIGH'   THEN 1 ELSE 0 END) AS high_risk,
                SUM(CASE WHEN risk_label='MEDIUM' THEN 1 ELSE 0 END) AS medium_risk,
                SUM(CASE WHEN risk_label='LOW'    THEN 1 ELSE 0 END) AS low_risk,
                SUM(CASE WHEN scored=0            THEN 1 ELSE 0 END) AS unscored,
                AVG(risk_score)                            AS avg_score,
                MAX(created_at)                            AS last_updated
            FROM predictions
            """
        )
        row = await cur.fetchone()
        if row is None:
            return {}
        return dict(row)


async def fetch_explanation(edit_id: str) -> Optional[dict]:
    async with get_db() as db:
        cur = await db.execute(
            "SELECT * FROM explanations WHERE edit_id = ?", [edit_id]
        )
        row = await cur.fetchone()
        return dict(row) if row else None


async def upsert_explanation(
    edit_id: str, explanation: str, model: str, generated_at: str
) -> None:
    async with get_db() as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO explanations (edit_id, explanation, model, generated_at)
            VALUES (?, ?, ?, ?)
            """,
            [edit_id, explanation, model, generated_at],
        )
        await db.commit()
