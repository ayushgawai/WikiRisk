"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Point all data paths to a temp dir during tests
_TMPDIR = tempfile.mkdtemp(prefix="wikirisk_test_")
os.environ.setdefault("DATA_DIR", _TMPDIR)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMPDIR}/test.db")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlruns/test.db")
os.environ.setdefault("LOG_FORMAT", "pretty")
os.environ.setdefault("LOG_LEVEL", "DEBUG")


@pytest.fixture(scope="session")
def tmp_data_dir() -> Path:
    return Path(_TMPDIR)


@pytest.fixture
def sample_edit() -> dict:
    return {
        "id": "test-edit-001",
        "rev_id": "987654321",
        "page_title": "Test Article",
        "namespace": 0,
        "user": "TestUser",
        "is_anon": 0,
        "comment": "Fixed typo in introduction",
        "length_delta": 15.0,
        "timestamp": "2024-01-01T12:00:00+00:00",
        "wiki": "enwiki",
        "risk_score": 0.12,
        "risk_label": "LOW",
        "scored": 1,
        "created_at": "2024-01-01T12:00:00+00:00",
    }


@pytest.fixture
def risky_edit() -> dict:
    return {
        "id": "test-edit-002",
        "rev_id": "111111111",
        "page_title": "Controversial Article",
        "namespace": 0,
        "user": "192.168.1.100",
        "is_anon": 1,
        "comment": "revert vandalism by user",
        "length_delta": -2500.0,
        "timestamp": "2024-01-01T03:00:00+00:00",
        "wiki": "enwiki",
        "risk_score": 0.85,
        "risk_label": "HIGH",
        "scored": 1,
        "created_at": "2024-01-01T03:00:00+00:00",
    }


@pytest.fixture
def raw_eventstream_event() -> dict:
    """A realistic Wikimedia EventStream edit event."""
    return {
        "schema": "mediawiki/recentchange/1.0.0",
        "meta": {
            "uri": "https://en.wikipedia.org/wiki/Test",
            "dt": "2024-01-01T12:00:00Z",
            "id": "abc123",
        },
        "id": 12345678,
        "type": "edit",
        "namespace": 0,
        "title": "Test Article",
        "comment": "/* Section */ clarified wording",
        "timestamp": 1704110400,
        "user": "WikiEditor123",
        "bot": False,
        "wiki": "enwiki",
        "server_name": "en.wikipedia.org",
        "length": {"old": 5000, "new": 5035},
        "revision": {"old": 987654320, "new": 987654321},
    }
