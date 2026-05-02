"""
WikiRisk – Pydantic models for the FastAPI serving layer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class EditRecord(BaseModel):
    """A single Wikipedia edit with risk scoring."""

    id: str
    rev_id: str = ""
    page_title: str
    namespace: int = 0
    user: str = ""
    is_anon: bool = False
    comment: str = ""
    length_delta: float = 0.0
    timestamp: str = ""
    wiki: str = "enwiki"
    risk_score: Optional[float] = None
    risk_label: Optional[str] = None
    scored: bool = False
    created_at: str = ""

    @field_validator("risk_score", mode="before")
    @classmethod
    def _round_score(cls, v):
        if v is None:
            return None
        return round(float(v), 4)

    @property
    def wiki_url(self) -> str:
        """Link to the Wikipedia revision diff."""
        if self.rev_id:
            return (
                f"https://en.wikipedia.org/w/index.php"
                f"?diff={self.rev_id}"
            )
        title = self.page_title.replace(" ", "_")
        return f"https://en.wikipedia.org/wiki/{title}"

    model_config = {"from_attributes": True}


class RecentEditsResponse(BaseModel):
    """Paginated list of recent edits."""

    items: list[EditRecord]
    total: int
    page: int
    page_size: int
    has_more: bool


class ExplanationRequest(BaseModel):
    """Request body for the AI explanation endpoint."""

    edit_id: str
    force_refresh: bool = False


class ExplanationResponse(BaseModel):
    """AI-generated natural language explanation of edit risk."""

    edit_id: str
    explanation: str
    model: str
    cached: bool = False
    generated_at: str = ""


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    version: str
    uptime_seconds: float
    services: dict[str, str]


class StatsResponse(BaseModel):
    """Aggregate statistics for the dashboard."""

    total_edits: int
    high_risk: int
    medium_risk: int
    low_risk: int
    unscored: int
    avg_risk_score: float
    last_updated: str
