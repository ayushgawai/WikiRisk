"""Manual alert notification route."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.common.logger import get_logger
from src.serving.database import fetch_edit_by_id
from src.serving.notifier import send_email_alert_sync, send_slack_alert

log = get_logger(__name__)
router = APIRouter(prefix="/notify", tags=["notify"])


class NotifyResponse(BaseModel):
    edit_id: str
    delivered: bool
    channels: list[dict]
    message: str


@router.post("/{edit_id}", response_model=NotifyResponse)
async def notify_edit(edit_id: str) -> NotifyResponse:
    """Send a Slack/email alert for a selected edit."""
    edit = await fetch_edit_by_id(edit_id)
    if edit is None:
        raise HTTPException(status_code=404, detail=f"Edit {edit_id!r} not found")

    channels: list[dict] = []

    try:
        channels.append(await send_slack_alert(edit))
    except Exception as exc:
        log.warning("slack_alert_failed", edit_id=edit_id, error=str(exc))
        channels.append(
            {
                "channel": "slack",
                "configured": True,
                "delivered": False,
                "error": str(exc),
            }
        )

    try:
        channels.append(await asyncio.to_thread(send_email_alert_sync, edit))
    except Exception as exc:
        log.warning("email_alert_failed", edit_id=edit_id, error=str(exc))
        channels.append(
            {
                "channel": "email",
                "configured": True,
                "delivered": False,
                "error": str(exc),
            }
        )

    delivered = any(c.get("delivered") for c in channels)
    configured = any(c.get("configured") for c in channels)
    if delivered:
        message = "Alert delivered."
    elif configured:
        message = "Notification channel configured, but delivery failed."
    else:
        message = (
            "No notification channel configured. Set SLACK_WEBHOOK_URL or SMTP "
            "environment variables to send real alerts."
        )

    return NotifyResponse(
        edit_id=edit_id,
        delivered=delivered,
        channels=channels,
        message=message,
    )
