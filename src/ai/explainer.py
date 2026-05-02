"""
WikiRisk – AI Explainability Layer (OpenAI).

Generates natural-language explanations for why a Wikipedia edit was
flagged as risky.  Explanations are persisted in the SQLite database
to avoid redundant API calls.

Key design:
  • Async OpenAI client (non-blocking in FastAPI context)
  • Graceful degradation if OPENAI_API_KEY is not set
  • Token-efficient prompt with structured context
  • No PII in prompts (username redacted)
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.common.logger import get_logger
from src.config import get_settings

log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are WikiRisk, an expert Wikipedia content integrity analyst.

Your task is to explain in 2-3 clear, concise sentences why a Wikipedia edit
was flagged as potentially risky by an automated ML classifier.

Focus on:
- Observable edit characteristics (size, comment, user type)
- Patterns associated with vandalism or low-quality edits
- What specifically increased the risk score

Be specific, factual, and educational.  Do NOT be alarmist.
Do NOT reveal or reproduce any username.  Keep the explanation under 100 words.
"""


def _build_prompt(edit: dict) -> str:
    title = edit.get("page_title", "Unknown Article")
    comment = edit.get("comment", "") or "(no comment)"
    delta = int(edit.get("length_delta") or 0)
    is_anon = bool(edit.get("is_anon", 0))
    risk_score = edit.get("risk_score")
    risk_label = edit.get("risk_label", "UNKNOWN")

    delta_desc = (
        f"added {delta:+d} bytes"
        if delta >= 0
        else f"removed {abs(delta)} bytes"
    )
    user_type = "an anonymous (IP) user" if is_anon else "a registered user"
    score_str = f"{float(risk_score):.1%}" if risk_score is not None else "unknown"

    return f"""Wikipedia edit details:
- Article: "{title}"
- Risk label: {risk_label} (score: {score_str})
- Edit size: {delta_desc}
- Made by: {user_type}
- Edit comment: "{comment[:200]}"

Explain why this edit was flagged as {risk_label} risk in 2-3 sentences."""


async def generate_explanation(edit: dict) -> dict:
    """
    Generate an AI explanation for the given edit dict.

    Returns a dict with keys: explanation, model, generated_at.
    Falls back to a rule-based explanation if OpenAI is unavailable.
    """
    cfg = get_settings()

    if not cfg.openai_api_key:
        log.info("openai_key_missing_using_fallback", edit_id=edit.get("id"))
        return _rule_based_explanation(edit)

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=cfg.openai_api_key)
        prompt = _build_prompt(edit)

        response = await client.chat.completions.create(
            model=cfg.openai_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=cfg.openai_max_tokens,
            temperature=0.3,
        )

        text = response.choices[0].message.content.strip()
        model_used = response.model
        log.info(
            "explanation_generated",
            edit_id=edit.get("id"),
            model=model_used,
            tokens=response.usage.total_tokens if response.usage else None,
        )

        return {
            "explanation": text,
            "model": model_used,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    except Exception as exc:
        log.warning(
            "openai_error_falling_back",
            error=str(exc),
            edit_id=edit.get("id"),
        )
        return _rule_based_explanation(edit)


def _rule_based_explanation(edit: dict) -> dict:
    """
    Generate a rule-based explanation when OpenAI is unavailable.
    Covers the most common risky edit patterns.
    """
    reasons: list[str] = []
    delta = float(edit.get("length_delta") or 0)
    is_anon = bool(edit.get("is_anon", 0))
    comment = (edit.get("comment") or "").lower()
    score = float(edit.get("risk_score") or 0)
    label = edit.get("risk_label", "UNKNOWN")

    risky_words = ["revert", "undo", "vandal", "spam", "rv", "copyvio"]
    if any(w in comment for w in risky_words):
        reasons.append(
            "the edit comment contains keywords associated with reverts or vandalism"
        )

    if delta < -500:
        reasons.append(f"a large amount of content was removed ({abs(int(delta))} bytes deleted)")
    elif delta < 0 and is_anon:
        reasons.append("an anonymous user removed content without explanation")

    if is_anon and abs(delta) > 1000:
        reasons.append("an anonymous user made a large content change")

    if not comment.strip() and is_anon:
        reasons.append("the anonymous editor provided no edit summary")

    if not reasons:
        if score >= 0.7:
            reasons.append(
                "multiple statistical features (edit size, user type, comment) "
                "combined to produce a high risk score"
            )
        else:
            reasons.append(
                "the combination of edit characteristics slightly elevated the risk score"
            )

    joined = "; ".join(reasons[:2])
    explanation = (
        f"This edit was flagged as **{label}** risk because {joined}. "
        f"The automated classifier assigned a risk score of {score:.1%} based on "
        f"TF-IDF analysis of the edit comment and numerical features including "
        f"content size change and user anonymity."
    )

    return {
        "explanation": explanation,
        "model": "rule-based-fallback",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
