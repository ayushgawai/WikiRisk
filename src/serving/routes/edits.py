"""Edit query routes."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.serving.database import fetch_edit_by_id, fetch_recent_edits
from src.serving.models import EditRecord, RecentEditsResponse
from src.common.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/edits", tags=["edits"])


@router.get("/recent", response_model=RecentEditsResponse)
async def recent_edits(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    risk_label: Optional[str] = Query(
        None,
        description="Filter by risk label: HIGH | MEDIUM | LOW",
        pattern="^(HIGH|MEDIUM|LOW)$",
    ),
    scored_only: bool = Query(False, description="Only return scored edits"),
    wiki: str = Query("enwiki", description="Target wiki"),
    search: Optional[str] = Query(
        None,
        min_length=1,
        max_length=120,
        description="Search page title, edit summary, editor, or revision ID",
    ),
) -> RecentEditsResponse:
    """Retrieve recent Wikipedia edits with risk scores (newest first)."""
    items, total = await fetch_recent_edits(
        page=page,
        page_size=page_size,
        risk_label=risk_label,
        scored_only=scored_only,
        wiki=wiki,
        search=search,
    )

    records = []
    for item in items:
        try:
            records.append(
                EditRecord(
                    **{
                        **item,
                        "is_anon": bool(item.get("is_anon", 0)),
                        "scored": bool(item.get("scored", 0)),
                    }
                )
            )
        except Exception as exc:
            log.warning("record_parse_error", error=str(exc), item=str(item)[:200])

    return RecentEditsResponse(
        items=records,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.get("/{edit_id}", response_model=EditRecord)
async def get_edit(edit_id: str) -> EditRecord:
    """Retrieve a single edit by its internal ID."""
    row = await fetch_edit_by_id(edit_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Edit {edit_id!r} not found")

    return EditRecord(
        **{
            **row,
            "is_anon": bool(row.get("is_anon", 0)),
            "scored": bool(row.get("scored", 0)),
        }
    )
