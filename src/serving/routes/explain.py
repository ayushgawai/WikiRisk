"""AI explanation route."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.serving.database import (
    fetch_edit_by_id,
    fetch_explanation,
    upsert_explanation,
)
from src.serving.models import ExplanationRequest, ExplanationResponse
from src.common.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/explain", tags=["explain"])


@router.post("", response_model=ExplanationResponse)
@router.get("/{edit_id}", response_model=ExplanationResponse)
async def explain_edit(
    edit_id: str | None = None,
    body: ExplanationRequest | None = None,
) -> ExplanationResponse:
    """
    Generate or retrieve an AI explanation for why an edit is risky.

    Checks the cache first; calls OpenAI only on cache miss or forced refresh.
    """
    # Resolve edit_id from path or body
    resolved_id = edit_id or (body.edit_id if body else None)
    force = (body.force_refresh if body else False) if body else False

    if not resolved_id:
        raise HTTPException(status_code=422, detail="edit_id is required")

    # ── Check cache ────────────────────────────────────────────────────────
    if not force:
        cached = await fetch_explanation(resolved_id)
        if cached:
            log.debug("explanation_cache_hit", edit_id=resolved_id)
            return ExplanationResponse(
                edit_id=resolved_id,
                explanation=cached["explanation"],
                model=cached["model"],
                cached=True,
                generated_at=cached["generated_at"],
            )

    # ── Load edit ──────────────────────────────────────────────────────────
    edit = await fetch_edit_by_id(resolved_id)
    if edit is None:
        raise HTTPException(status_code=404, detail=f"Edit {resolved_id!r} not found")

    # ── Generate explanation ───────────────────────────────────────────────
    from src.ai.explainer import generate_explanation

    result = await generate_explanation(edit)

    await upsert_explanation(
        edit_id=resolved_id,
        explanation=result["explanation"],
        model=result["model"],
        generated_at=result["generated_at"],
    )

    return ExplanationResponse(
        edit_id=resolved_id,
        explanation=result["explanation"],
        model=result["model"],
        cached=False,
        generated_at=result["generated_at"],
    )
