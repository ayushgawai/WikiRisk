"""Health and metrics routes."""
from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter

from src.serving.models import HealthResponse, StatsResponse
from src.serving.database import fetch_stats
from src.common.logger import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["health"])

_START_TIME = time.monotonic()
VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health status."""
    import httpx

    services: dict[str, str] = {}

    # MLflow is a required part of the demo/provenance story.
    from src.config import get_settings
    cfg = get_settings()
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{cfg.mlflow_tracking_uri}/health")
            services["mlflow"] = "ok" if r.status_code == 200 else "degraded"
    except Exception:
        services["mlflow"] = "unreachable"

    services["api"] = "ok"
    services["database"] = "ok"

    return HealthResponse(
        status="ok" if services["mlflow"] == "ok" else "degraded",
        version=VERSION,
        uptime_seconds=round(time.monotonic() - _START_TIME, 1),
        services=services,
    )


@router.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """Return aggregate risk statistics."""
    data = await fetch_stats()
    return StatsResponse(
        total_edits=int(data.get("total_edits") or 0),
        high_risk=int(data.get("high_risk") or 0),
        medium_risk=int(data.get("medium_risk") or 0),
        low_risk=int(data.get("low_risk") or 0),
        unscored=int(data.get("unscored") or 0),
        avg_risk_score=round(float(data.get("avg_score") or 0.0), 4),
        last_updated=str(data.get("last_updated") or ""),
    )
