"""Slack and email notification helpers for WikiRisk."""
from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Any

import httpx

from src.config import get_settings


def build_alert_message(edit: dict[str, Any]) -> tuple[str, str]:
    """Return a compact subject and body for a risky-edit alert."""
    label = edit.get("risk_label") or "UNKNOWN"
    score = edit.get("risk_score")
    score_text = f"{float(score):.1%}" if score is not None else "not scored"
    title = edit.get("page_title") or "Unknown page"
    rev_id = edit.get("rev_id") or ""
    delta = int(float(edit.get("length_delta") or 0))
    comment = edit.get("comment") or "(no edit summary)"
    diff_url = (
        f"https://en.wikipedia.org/wiki/Special:Diff/{rev_id}"
        if rev_id
        else f"https://en.wikipedia.org/wiki/{str(title).replace(' ', '_')}"
    )

    subject = f"WikiRisk {label} alert: {title}"
    body = (
        f"WikiRisk flagged an edit on {title} as {label} risk "
        f"(score {score_text}).\n\n"
        f"Size change: {delta:+d} bytes\n"
        f"Edit summary: {comment[:500]}\n"
        f"Diff/page: {diff_url}"
    )
    return subject, body


async def send_slack_alert(edit: dict[str, Any]) -> dict[str, Any]:
    """Send a Slack alert when SLACK_WEBHOOK_URL is configured."""
    cfg = get_settings()
    if not cfg.slack_webhook_url:
        return {"channel": "slack", "configured": False, "delivered": False}

    subject, body = build_alert_message(edit)
    payload = {"text": f"*{subject}*\n{body}"}
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(cfg.slack_webhook_url, json=payload)
        response.raise_for_status()
    return {"channel": "slack", "configured": True, "delivered": True}


def send_email_alert_sync(edit: dict[str, Any]) -> dict[str, Any]:
    """Send an SMTP email alert when SMTP settings are configured."""
    cfg = get_settings()
    required = [
        cfg.smtp_host,
        cfg.smtp_username,
        cfg.smtp_password,
        cfg.smtp_from_email,
        cfg.alert_to_email,
    ]
    if not all(required):
        return {"channel": "email", "configured": False, "delivered": False}

    subject, body = build_alert_message(edit)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.smtp_from_email
    msg["To"] = cfg.alert_to_email
    msg.set_content(body)

    with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=10) as server:
        server.starttls()
        server.login(cfg.smtp_username, cfg.smtp_password)
        server.send_message(msg)

    return {"channel": "email", "configured": True, "delivered": True}
