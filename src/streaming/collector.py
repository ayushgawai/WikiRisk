"""
WikiRisk – Wikimedia EventStreams SSE Collector.

Reads the Wikimedia Recent Changes SSE stream and:
  1. Filters for English Wikipedia article edits (namespace=0)
  2. Writes micro-batches of JSON events to data/stream/incoming/
     as JSONL files (one file per flush interval)
  3. Also inserts raw events into the SQLite predictions table so the
     FastAPI backend can serve "live" edits immediately (pre-scoring)

This runs as a long-lived daemon process.

CLI:
    python -m src.streaming.collector [--wiki enwiki] [--limit N]
"""
from __future__ import annotations

import json
import signal
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import click
import requests
import sseclient

from src.common.logger import configure_logging, get_logger
from src.config import get_settings

log = get_logger(__name__)

_RUNNING = True


def _handle_signal(signum, frame):
    global _RUNNING
    log.info("shutdown_signal_received", signum=signum)
    _RUNNING = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def _db_ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id            TEXT PRIMARY KEY,
            rev_id        TEXT,
            page_title    TEXT,
            namespace     INTEGER,
            user          TEXT,
            is_anon       INTEGER,
            comment       TEXT,
            length_delta  REAL,
            timestamp     TEXT,
            wiki          TEXT,
            risk_score    REAL,
            risk_label    TEXT,
            scored        INTEGER DEFAULT 0,
            created_at    TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_created ON predictions (created_at DESC)"
    )
    conn.commit()


def _insert_event(conn: sqlite3.Connection, event: dict) -> str:
    """Insert a raw SSE event into the DB.  Returns the generated edit ID."""
    length = event.get("length") or {}
    old_len = length.get("old") or 0
    new_len = length.get("new") or 0
    delta = new_len - old_len

    import re
    user = event.get("user", "") or ""
    is_anon = int(bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$|^[0-9a-fA-F:]+:", user)))

    rev_info = event.get("revision") or {}
    rev_id = str(rev_info.get("new", "")) if rev_info else str(event.get("id", ""))

    edit_id = str(uuid.uuid4())
    now = datetime.now(tz=timezone.utc).isoformat()

    conn.execute(
        """
        INSERT OR IGNORE INTO predictions
            (id, rev_id, page_title, namespace, user, is_anon, comment,
             length_delta, timestamp, wiki, risk_score, risk_label,
             scored, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0, ?)
        """,
        (
            edit_id,
            rev_id,
            event.get("title", ""),
            int(event.get("namespace", 0) or 0),
            user,
            is_anon,
            event.get("comment", ""),
            float(delta),
            event.get("meta", {}).get("dt", now) if isinstance(event.get("meta"), dict) else now,
            event.get("wiki", ""),
            now,
        ),
    )
    return edit_id


def _flush_batch(
    events: list[dict],
    output_dir: Path,
    batch_id: str,
) -> Path:
    """Write a batch of events as a JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / f"batch_{batch_id}.jsonl"
    with fpath.open("w") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    log.debug("batch_flushed", file=fpath.name, events=len(events))
    return fpath


def stream_events(
    url: str,
    target_wiki: str = "enwiki",
) -> Iterator[dict]:
    """Yield filtered SSE events from the Wikimedia stream."""
    headers = {"Accept": "text/event-stream", "User-Agent": "WikiRisk/1.0"}

    while _RUNNING:
        try:
            response = requests.get(url, stream=True, headers=headers, timeout=60)
            response.raise_for_status()
            client = sseclient.SSEClient(response)

            for msg in client.events():
                if not _RUNNING:
                    return
                if not msg.data or msg.data.strip() == "":
                    continue
                try:
                    event = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                # Filter: only edits on target wiki, namespace 0
                if (
                    event.get("type") == "edit"
                    and event.get("wiki") == target_wiki
                    and event.get("namespace") == 0
                ):
                    yield event

        except (requests.ConnectionError, requests.Timeout) as exc:
            log.warning("stream_connection_error", error=str(exc), retry_in=5)
            time.sleep(5)
        except Exception as exc:
            log.error("stream_unexpected_error", error=str(exc))
            time.sleep(10)


def run_collector(
    target_wiki: str = "enwiki",
    limit: int | None = None,
    flush_interval: int = 10,
) -> None:
    cfg = get_settings()
    cfg.ensure_dirs()

    db_path = cfg.db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    _db_ensure_table(conn)

    log.info(
        "collector_start",
        wiki=target_wiki,
        output_dir=str(cfg.stream_input_dir),
        db=db_path,
    )

    buffer: list[dict] = []
    last_flush = time.monotonic()
    total = 0

    for event in stream_events(cfg.wikimedia_stream_url, target_wiki=target_wiki):
        buffer.append(event)
        _insert_event(conn, event)
        conn.commit()
        total += 1

        now = time.monotonic()
        should_flush = (
            len(buffer) >= cfg.stream_batch_size
            or (now - last_flush) >= flush_interval
        )

        if should_flush and buffer:
            batch_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            _flush_batch(buffer, cfg.stream_input_dir, batch_id)
            buffer = []
            last_flush = now
            log.info("collector_progress", total_events=total)

        if limit and total >= limit:
            log.info("collector_limit_reached", limit=limit)
            break

    # Final flush
    if buffer:
        batch_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        _flush_batch(buffer, cfg.stream_input_dir, batch_id)

    conn.close()
    log.info("collector_stopped", total_events=total)


@click.command()
@click.option("--wiki", default="enwiki", help="Target Wikipedia (e.g. enwiki)")
@click.option("--limit", default=None, type=int, help="Stop after N events")
@click.option("--flush-interval", default=10, help="Seconds between file flushes")
def main(wiki: str, limit: int | None, flush_interval: int) -> None:
    """Collect live Wikipedia edits and write to data/stream/incoming/."""
    cfg = get_settings()
    configure_logging(cfg.log_level, cfg.log_format)
    run_collector(target_wiki=wiki, limit=limit, flush_interval=flush_interval)


if __name__ == "__main__":
    main()
