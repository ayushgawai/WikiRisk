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

Your task is to explain in 2-3 clear, concise sentences WHY the automated ML
classifier assigned this particular risk score to a Wikipedia edit.

Critical rules:
- Explain the SIGNALS the classifier detected, NOT a verdict on the editor's intent.
- A "revert" comment often means the EDITOR IS FIXING someone else's vandalism — 
  flag this nuance when relevant (it raises the classifier score but is often benign).
- An "undo" comment usually means a good-faith rollback — note this.
- Be factual and measured. Do NOT say the edit IS vandalism — say the signals RESEMBLE patterns the classifier associates with risk.
- Focus on: edit size, user type (anon vs registered), comment keywords, content delta.
- Do NOT be alarmist. Keep the explanation under 90 words.
- Do NOT reproduce or reference any username.
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
        result = _rule_based_explanation(edit)
        try:
            _mlflow_rest_log_explanation(edit, result, cfg)
        except Exception:
            pass
        return result

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=cfg.openai_api_key)
        prompt = _build_prompt(edit)

        # Try configured model first, then fall back to a safer widely-available model.
        candidates = [cfg.openai_model, "gpt-3.5-turbo"]
        last_exc = None
        for m in candidates:
            if not m:
                continue
            try:
                response = await client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=cfg.openai_max_tokens,
                    temperature=0.3,
                )

                text = response.choices[0].message.content.strip()
                model_used = getattr(response, "model", m)
                tokens = getattr(getattr(response, "usage", None), "total_tokens", None)
                log.info(
                    "explanation_generated",
                    edit_id=edit.get("id"),
                    model=model_used,
                    tokens=tokens,
                )
                # Best-effort: log explanation to MLflow tracking (if available).
                try:
                    import mlflow
                    import tempfile

                    try:
                        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
                        mlflow.set_experiment(cfg.mlflow_experiment_name)
                    except Exception:
                        # Non-fatal: continue even if setting experiment/URI fails
                        pass

                    try:
                        with mlflow.start_run():
                            mlflow.log_param("edit_id", edit.get("id"))
                            mlflow.log_param("model", model_used)
                            if edit.get("risk_label"):
                                mlflow.log_param("risk_label", edit.get("risk_label"))
                            if edit.get("risk_score") is not None:
                                try:
                                    mlflow.log_metric("risk_score", float(edit.get("risk_score")))
                                except Exception:
                                    pass
                            # write explanation to a temp file and log as artifact
                            with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
                                tf.write(text)
                                tf.flush()
                                mlflow.log_artifact(tf.name, artifact_path="explanations")
                    except Exception as e:
                        log.warning("mlflow_log_failed", error=str(e), edit_id=edit.get("id"))
                except Exception:
                    # mlflow package not available; attempt REST API fallback to server
                    try:
                        import requests
                        base = cfg.mlflow_tracking_uri.rstrip("/")

                        # get experiment id by name
                        exp_resp = requests.post(
                            f"{base}/api/2.0/mlflow/experiments/get-by-name",
                            json={"name": cfg.mlflow_experiment_name},
                            timeout=5,
                        )
                        if exp_resp.status_code == 200:
                            exp_id = exp_resp.json().get("experiment", {}).get("experiment_id")
                        else:
                            # try create experiment
                            create_resp = requests.post(
                                f"{base}/api/2.0/mlflow/experiments/create",
                                json={"name": cfg.mlflow_experiment_name},
                                timeout=5,
                            )
                            exp_id = create_resp.json().get("experiment_id")

                        if exp_id:
                            # create a run
                            run_resp = requests.post(
                                f"{base}/api/2.0/mlflow/runs/create",
                                json={
                                    "experiment_id": exp_id,
                                    "start_time": int(datetime.now(tz=timezone.utc).timestamp() * 1000),
                                    "tags": [{"key": "edit_id", "value": str(edit.get("id"))}],
                                },
                                timeout=5,
                            )
                            run = run_resp.json().get("run", {})
                            run_id = run.get("info", {}).get("run_id")
                            if run_id:
                                # log params (explanation may be long; we store truncated param and model)
                                try:
                                    requests.post(
                                        f"{base}/api/2.0/mlflow/runs/log-parameter",
                                        json={"run_id": run_id, "key": "model", "value": str(model_used)},
                                        timeout=3,
                                    )
                                    if edit.get("risk_label"):
                                        requests.post(
                                            f"{base}/api/2.0/mlflow/runs/log-parameter",
                                            json={"run_id": run_id, "key": "risk_label", "value": str(edit.get("risk_label"))},
                                            timeout=3,
                                        )
                                    if edit.get("risk_score") is not None:
                                        try:
                                            requests.post(
                                                f"{base}/api/2.0/mlflow/runs/log-metric",
                                                json={"run_id": run_id, "key": "risk_score", "value": float(edit.get("risk_score"))},
                                                timeout=3,
                                            )
                                        except Exception:
                                            pass
                                    # store explanation as a parameter (truncated to 2000 chars)
                                    expl_text = text[:2000]
                                    requests.post(
                                        f"{base}/api/2.0/mlflow/runs/log-parameter",
                                        json={"run_id": run_id, "key": "explanation", "value": expl_text},
                                        timeout=5,
                                    )
                                except Exception as e:
                                    log.warning("mlflow_rest_log_failed", error=str(e), edit_id=edit.get("id"))
                    except Exception:
                        # final fallback: ignore mlflow logging errors
                        log.debug("mlflow_rest_unavailable", edit_id=edit.get("id"))

                return {
                    "explanation": text,
                    "model": model_used,
                    "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                }

            except Exception as exc:
                # Capture and try next model; log details for debugging.
                last_exc = exc
                log.warning(
                    "openai_attempt_failed",
                    model=m,
                    error=str(exc),
                    edit_id=edit.get("id"),
                )

        # If we exit the loop, all model attempts failed — log final error
        log.warning(
            "openai_all_models_failed",
            error=str(last_exc),
            edit_id=edit.get("id"),
        )
        result = _rule_based_explanation(edit)
        try:
            _mlflow_rest_log_explanation(edit, result, cfg)
        except Exception:
            pass
        return result

    except Exception as exc:
        log.warning(
            "openai_error_falling_back",
            error=str(exc),
            edit_id=edit.get("id"),
        )
        result = _rule_based_explanation(edit)
        try:
            _mlflow_rest_log_explanation(edit, result, cfg)
        except Exception:
            pass
        return result


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


def _mlflow_rest_log_explanation(edit: dict, result: dict, cfg) -> None:
    """Best-effort: log explanation to MLflow REST API if available."""
    try:
        import requests
        base = cfg.mlflow_tracking_uri.rstrip("/")
        # get or create experiment
        exp_resp = requests.post(
            f"{base}/api/2.0/mlflow/experiments/get-by-name",
            json={"name": cfg.mlflow_experiment_name},
            timeout=5,
        )
        if exp_resp.status_code == 200:
            exp_id = exp_resp.json().get("experiment", {}).get("experiment_id")
        else:
            create_resp = requests.post(
                f"{base}/api/2.0/mlflow/experiments/create",
                json={"name": cfg.mlflow_experiment_name},
                timeout=5,
            )
            exp_id = create_resp.json().get("experiment_id")

        if not exp_id:
            return

        run_resp = requests.post(
            f"{base}/api/2.0/mlflow/runs/create",
            json={
                "experiment_id": exp_id,
                "start_time": int(datetime.now(tz=timezone.utc).timestamp() * 1000),
                "tags": [{"key": "edit_id", "value": str(edit.get("id"))}],
            },
            timeout=5,
        )
        run = run_resp.json().get("run", {})
        run_id = run.get("info", {}).get("run_id")
        if not run_id:
            return

        # log a few params/metrics
        try:
            requests.post(f"{base}/api/2.0/mlflow/runs/log-parameter", json={"run_id": run_id, "key": "model", "value": str(result.get("model"))}, timeout=3)
            if edit.get("risk_label"):
                requests.post(f"{base}/api/2.0/mlflow/runs/log-parameter", json={"run_id": run_id, "key": "risk_label", "value": str(edit.get("risk_label"))}, timeout=3)
            if edit.get("risk_score") is not None:
                try:
                    requests.post(f"{base}/api/2.0/mlflow/runs/log-metric", json={"run_id": run_id, "key": "risk_score", "value": float(edit.get("risk_score"))}, timeout=3)
                except Exception:
                    pass
            # store explanation truncated
            expl_text = result.get("explanation", "")[:2000]
            requests.post(f"{base}/api/2.0/mlflow/runs/log-parameter", json={"run_id": run_id, "key": "explanation", "value": expl_text}, timeout=5)
        except Exception:
            return
    except Exception:
        return
