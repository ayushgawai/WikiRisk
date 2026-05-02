"""
Tests for the FastAPI backend.

Uses httpx AsyncClient against the real app (with a test SQLite db).
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport


# ── App setup ─────────────────────────────────────────────────────────────────
# Override database URL before importing the app
_TMP_DB = tempfile.mktemp(suffix=".db", prefix="wikirisk_api_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"


def _seed_db(db_path: str, n: int = 10) -> list[str]:
    """Insert n test predictions into the SQLite DB.  Return list of IDs."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY, rev_id TEXT, page_title TEXT,
            namespace INTEGER, user TEXT, is_anon INTEGER,
            comment TEXT, length_delta REAL, timestamp TEXT,
            wiki TEXT, risk_score REAL, risk_label TEXT,
            scored INTEGER, created_at TEXT
        )
        """
    )
    ids = []
    now = datetime.now(tz=timezone.utc).isoformat()
    for i in range(n):
        eid = str(uuid.uuid4())
        ids.append(eid)
        label = ["HIGH", "MEDIUM", "LOW"][i % 3]
        score = [0.85, 0.55, 0.15][i % 3]
        conn.execute(
            """
            INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                eid, f"rev_{i}", f"Article {i}", 0,
                "TestUser" if i % 2 == 0 else "1.2.3.4",
                i % 2, f"edit comment {i}", float(i * 10 - 50),
                now, "enwiki", score, label, 1, now,
            ),
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS explanations
        (edit_id TEXT PRIMARY KEY, explanation TEXT, model TEXT, generated_at TEXT)
        """
    )
    conn.commit()
    conn.close()
    return ids


@pytest_asyncio.fixture
async def client():
    """Async HTTPX client connected to the test app."""
    from src.serving.main import app

    _seed_db(_TMP_DB, n=15)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ── Tests ──────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_root(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "WikiRisk" in r.json().get("service", "")


@pytest.mark.asyncio
async def test_health_returns_ok(client):
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "uptime_seconds" in data


@pytest.mark.asyncio
async def test_recent_edits_default(client):
    r = await client.get("/edits/recent")
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert "total" in data
    assert isinstance(data["items"], list)
    assert data["total"] >= 0


@pytest.mark.asyncio
async def test_recent_edits_filter_high(client):
    r = await client.get("/edits/recent", params={"risk_label": "HIGH"})
    assert r.status_code == 200
    data = r.json()
    for item in data["items"]:
        assert item["risk_label"] == "HIGH"


@pytest.mark.asyncio
async def test_recent_edits_pagination(client):
    r1 = await client.get("/edits/recent", params={"page": 1, "page_size": 5})
    assert r1.status_code == 200
    assert len(r1.json()["items"]) <= 5

    r2 = await client.get("/edits/recent", params={"page": 2, "page_size": 5})
    assert r2.status_code == 200


@pytest.mark.asyncio
async def test_get_edit_not_found(client):
    r = await client.get("/edits/nonexistent-id-xyz")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_edit_found(client):
    # Get an ID from the DB
    conn = sqlite3.connect(_TMP_DB)
    row = conn.execute("SELECT id FROM predictions LIMIT 1").fetchone()
    conn.close()

    if row:
        r = await client.get(f"/edits/{row[0]}")
        assert r.status_code == 200
        data = r.json()
        assert "id" in data
        assert "risk_label" in data


@pytest.mark.asyncio
async def test_stats_endpoint(client):
    r = await client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_edits" in data
    assert "high_risk" in data
    assert "avg_risk_score" in data


@pytest.mark.asyncio
async def test_explain_not_found(client):
    r = await client.get("/explain/nonexistent-id")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_explain_uses_fallback(client):
    """Explanation should work without OpenAI key via rule-based fallback."""
    conn = sqlite3.connect(_TMP_DB)
    row = conn.execute(
        "SELECT id FROM predictions WHERE risk_label='HIGH' LIMIT 1"
    ).fetchone()
    conn.close()

    if row:
        r = await client.get(f"/explain/{row[0]}")
        assert r.status_code == 200
        data = r.json()
        assert "explanation" in data
        assert len(data["explanation"]) > 10


@pytest.mark.asyncio
async def test_response_time_header(client):
    r = await client.get("/health")
    assert "x-response-time" in r.headers
