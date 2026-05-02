"""
Tests for feature engineering and weak labelling.

Marked with @pytest.mark.slow for tests that require a Spark session.
Quick tests (no Spark) run in the default test suite.
"""
from __future__ import annotations

import pytest


# ── Quick tests (no Spark) ─────────────────────────────────────────────────────

class TestPrepareStreamingRecord:
    """Unit tests for the streaming feature preparation function."""

    def test_basic_edit(self, raw_eventstream_event):
        from src.ml.features import prepare_streaming_record

        record = prepare_streaming_record(raw_eventstream_event)

        assert record["page_title"] == "Test Article"
        assert record["length_delta"] == 35.0
        assert record["is_anon"] is False
        assert record["is_anon_int"] == 0
        assert record["hour_of_day"] == 12
        assert isinstance(record["comment_clean"], str)
        assert isinstance(record["title_clean"], str)

    def test_anonymous_user(self):
        from src.ml.features import prepare_streaming_record

        event = {
            "type": "edit",
            "namespace": 0,
            "title": "Some Page",
            "comment": "",
            "user": "192.168.1.1",
            "wiki": "enwiki",
            "length": {"old": 1000, "new": 500},
            "meta": {"dt": "2024-01-01T15:30:00Z"},
            "id": 99,
            "revision": {"new": 111},
        }
        record = prepare_streaming_record(event)
        assert record["is_anon"] is True
        assert record["is_anon_int"] == 1
        assert record["length_delta"] == -500.0

    def test_missing_fields_handled(self):
        from src.ml.features import prepare_streaming_record

        record = prepare_streaming_record({})
        assert record["length_delta"] == 0.0
        assert record["is_anon_int"] in (0, 1)
        assert isinstance(record["comment_clean"], str)

    def test_text_cleaning(self):
        from src.ml.features import prepare_streaming_record

        event = {
            "type": "edit",
            "namespace": 0,
            "title": "Cat (animal)",
            "comment": "Fixed [[link]] & typo—123!",
            "user": "Editor",
            "wiki": "enwiki",
            "length": {"old": 100, "new": 105},
            "meta": {"dt": "2024-06-15T08:00:00Z"},
            "id": 1,
            "revision": {"new": 1},
        }
        record = prepare_streaming_record(event)
        assert "[[" not in record["comment_clean"]
        assert "((" not in record["title_clean"]


class TestWeakLabeler:
    """Unit tests for heuristic label assignment."""

    def _label(self, comment: str, delta: int = 0, is_anon: int = 0):
        """Helper: create minimal df and apply labels."""
        try:
            from pyspark.sql import SparkSession
            from src.ml.labeler import apply_weak_labels
        except ImportError:
            pytest.skip("PySpark not available")

        spark = SparkSession.builder.master("local").appName("test").getOrCreate()
        df = spark.createDataFrame(
            [{"comment": comment, "length_delta": float(delta), "is_anon_int": is_anon}]
        )
        labeled = apply_weak_labels(df)
        row = labeled.first()
        return row["label"]

    def test_revert_comment_flagged(self):
        assert self._label("Reverted edits by user") == 1

    def test_vandal_comment_flagged(self):
        assert self._label("rv vandalism") == 1

    def test_large_deletion_flagged(self):
        assert self._label("", delta=-600) == 1

    def test_normal_edit_not_flagged(self):
        assert self._label("Fixed grammar", delta=50, is_anon=0) == 0

    def test_anon_large_change_flagged(self):
        assert self._label("", delta=2000, is_anon=1) == 1


class TestAIExplainer:
    """Tests for the rule-based fallback explainer (no OpenAI key needed)."""

    @pytest.mark.asyncio
    async def test_fallback_high_risk(self, risky_edit):
        import os
        os.environ["OPENAI_API_KEY"] = ""
        from src.ai.explainer import _rule_based_explanation

        result = _rule_based_explanation(risky_edit)
        assert "explanation" in result
        assert "model" in result
        assert "HIGH" in result["explanation"]
        assert len(result["explanation"]) > 20

    @pytest.mark.asyncio
    async def test_fallback_low_risk(self, sample_edit):
        from src.ai.explainer import _rule_based_explanation

        result = _rule_based_explanation(sample_edit)
        assert "explanation" in result
        assert result["model"] == "rule-based-fallback"

    @pytest.mark.asyncio
    async def test_generate_uses_fallback_without_key(self, sample_edit):
        import os
        os.environ["OPENAI_API_KEY"] = ""
        # Clear settings cache
        from src.config.settings import get_settings
        get_settings.cache_clear()

        from src.ai.explainer import generate_explanation
        result = await generate_explanation(sample_edit)
        assert "explanation" in result
        assert len(result["explanation"]) > 0
