"""WikiRisk Streamlit dashboard."""
from __future__ import annotations

import os
import time
from html import escape
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_BASE = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
STREAM_URL = os.getenv(
    "WIKIMEDIA_STREAM_URL",
    "https://stream.wikimedia.org/v2/stream/recentchange",
)
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

RISK_COLORS = {
    "HIGH": "#b00020",
    "MEDIUM": "#d99b12",
    "LOW": "#116329",
    "UNSCORED": "#72777d",
}


st.set_page_config(
    page_title="WikiRisk Control Room",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #202122;
            --muted: #54595d;
            --blue: #3366cc;
            --gold: #d99b12;
            --paper: #f8f9fa;
            --line: #d8dde6;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }
        h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            box-shadow: 0 1px 2px rgba(32,33,34,0.04);
            min-height: 92px;
        }
        div[data-testid="stMetric"] label {
            color: var(--muted);
            font-size: 0.78rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
            line-height: 1.15;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 0.72rem;
        }
        .wikirisk-hero {
            border-bottom: 1px solid var(--line);
            padding-bottom: 0.75rem;
            margin-bottom: 0.9rem;
        }
        .wikirisk-title {
            font-size: 1.8rem;
            font-weight: 750;
            margin: 0;
            color: var(--ink);
        }
        .wikirisk-subtitle {
            color: var(--muted);
            margin-top: 0.25rem;
            font-size: 0.95rem;
        }
        .risk-high { color: #b00020; font-weight: 700; }
        .risk-medium { color: #946200; font-weight: 700; }
        .risk-low { color: #116329; font-weight: 700; }
        .status-pill {
            display: inline-block;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            background: #eef3ff;
            border: 1px solid #c8d7ff;
            color: #1f4aa8;
            font-size: 0.78rem;
            font-weight: 650;
            margin: 0.25rem 0.35rem 0 0;
        }
        .compact-facts {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem;
            margin: 0.65rem 0;
        }
        .fact {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #fff;
            padding: 0.6rem 0.7rem;
            min-width: 0;
        }
        .fact-label {
            color: var(--muted);
            font-size: 0.72rem;
            margin-bottom: 0.2rem;
        }
        .fact-value {
            color: var(--ink);
            font-size: 0.95rem;
            font-weight: 650;
            overflow-wrap: anywhere;
        }
        .stream-text {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #fff;
            padding: 0.7rem;
            max-height: 120px;
            overflow-y: auto;
            font-size: 0.88rem;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }
        .small-muted {
            color: var(--muted);
            font-size: 0.78rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def api_get(path: str, **params: Any) -> dict[str, Any]:
    response = requests.get(f"{API_BASE}{path}", params=params, timeout=8)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.post(f"{API_BASE}{path}", json=payload or {}, timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=3, show_spinner=False)
def load_stats(refresh_key: int) -> dict[str, Any]:
    return api_get("/stats")


@st.cache_data(ttl=3, show_spinner=False)
def load_health(refresh_key: int) -> dict[str, Any]:
    return api_get("/health")


@st.cache_data(ttl=3, show_spinner=False)
def load_edits(
    page_size: int,
    risk_label: str | None,
    scored_only: bool,
    search: str,
    refresh_key: int,
) -> dict[str, Any]:
    params = {
        "page": 1,
        "page_size": page_size,
        "risk_label": risk_label,
        "scored_only": scored_only,
    }
    if search.strip():
        params["search"] = search.strip()
    return api_get("/edits/recent", **params)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def time_ago(value: str | None) -> str:
    dt = parse_dt(value)
    if dt is None:
        return ""
    now = datetime.now(UTC)
    seconds = max(0, int((now - dt.astimezone(UTC)).total_seconds()))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


def diff_url(edit: dict[str, Any]) -> str:
    rev_id = str(edit.get("rev_id") or "").strip()
    title = str(edit.get("page_title") or "Wikipedia").replace(" ", "_")
    if rev_id:
        return f"https://en.wikipedia.org/wiki/Special:Diff/{rev_id}"
    return f"https://en.wikipedia.org/wiki/{title}"


def risk_class(label: str | None) -> str:
    return {
        "HIGH": "risk-high",
        "MEDIUM": "risk-medium",
        "LOW": "risk-low",
    }.get(label or "", "")


def render_header(health: dict[str, Any]) -> None:
    services = health.get("services", {})
    pills = "".join(
        f'<span class="status-pill">{name}: {status}</span>'
        for name, status in services.items()
    )
    st.markdown(
        f"""
        <div class="wikirisk-hero">
          <p class="wikirisk-title">WikiRisk Control Room</p>
          <div class="wikirisk-subtitle">
            Live Wikipedia edit integrity monitoring powered by Spark streaming,
            SparkML, MLflow, FastAPI, AI explanations, and alerting.
          </div>
          <div style="margin-top:0.55rem">{pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(stats: dict[str, Any]) -> None:
    total = int(stats.get("total_edits") or 0)
    high = int(stats.get("high_risk") or 0)
    med = int(stats.get("medium_risk") or 0)
    low = int(stats.get("low_risk") or 0)
    unscored = int(stats.get("unscored") or 0)
    avg = float(stats.get("avg_risk_score") or 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total edits", f"{total:,}")
    c2.metric("High risk", f"{high:,}", delta=f"{(high / total * 100):.1f}%" if total else None)
    c3.metric("Medium risk", f"{med:,}")
    c4.metric("Low risk", f"{low:,}")
    c5.metric("Avg / unscored", f"{avg:.3f}", delta=f"{unscored:,} unscored")


def edits_to_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        ts = item.get("timestamp") or item.get("created_at")
        rows.append(
            {
                "id": item.get("id"),
                "risk": item.get("risk_label") or "UNSCORED",
                "score": item.get("risk_score"),
                "page": item.get("page_title"),
                "delta": item.get("length_delta"),
                "editor": "anonymous" if item.get("is_anon") else "registered",
                "edit_summary": item.get("comment") or "",
                "age": time_ago(ts),
                "timestamp": ts or "",
                "rev_id": item.get("rev_id") or "",
            }
        )
    return pd.DataFrame(rows)


def render_charts(df: pd.DataFrame) -> None:
    left, right = st.columns([0.9, 1.1])
    if df.empty:
        left.info("Waiting for scored edits.")
        right.info("Start the collector and streaming processor to populate live data.")
        return

    order = ["HIGH", "MEDIUM", "LOW", "UNSCORED"]
    counts = df["risk"].value_counts().reindex(order).fillna(0).reset_index()
    counts.columns = ["risk", "count"]
    fig_donut = px.pie(
        counts[counts["count"] > 0],
        names="risk",
        values="count",
        hole=0.58,
        color="risk",
        color_discrete_map=RISK_COLORS,
        height=285,
    )
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    fig_donut.update_layout(
        showlegend=False,
        margin=dict(l=8, r=8, t=18, b=8),
        uniformtext_minsize=11,
        uniformtext_mode="hide",
    )
    left.plotly_chart(fig_donut, use_container_width=True)

    scored = df.dropna(subset=["score"]).copy()
    if scored.empty:
        right.info("No scored rows in the current filter.")
        return
    fig_scatter = px.scatter(
        scored.head(100),
        x="delta",
        y="score",
        color="risk",
        hover_name="page",
        hover_data=["editor", "edit_summary", "age"],
        color_discrete_map=RISK_COLORS,
        height=285,
    )
    fig_scatter.update_layout(margin=dict(l=8, r=8, t=18, b=8))
    right.plotly_chart(fig_scatter, use_container_width=True)


def render_detail(selected: dict[str, Any]) -> None:
    label = selected.get("risk_label") or "UNSCORED"
    score = selected.get("risk_score")
    score_text = f"{float(score):.1%}" if score is not None else "not scored"
    title = selected.get("page_title") or "Unknown page"
    ts = selected.get("timestamp") or selected.get("created_at")
    summary = selected.get("comment") or "No edit summary provided by the editor."

    st.markdown(
        f"### {title}\n"
        f'<span class="{risk_class(label)}">{label}</span> risk - {score_text} - '
        f"[open Wikipedia diff]({diff_url(selected)})",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="compact-facts">
          <div class="fact"><div class="fact-label">Byte delta</div><div class="fact-value">{float(selected.get("length_delta") or 0):+.0f}</div></div>
          <div class="fact"><div class="fact-label">Editor type</div><div class="fact-value">{"Anonymous" if selected.get("is_anon") else "Registered"}</div></div>
          <div class="fact"><div class="fact-label">Wiki</div><div class="fact-value">{selected.get("wiki") or "enwiki"}</div></div>
        </div>
        <div class="small-muted">{time_ago(ts)} - {ts or ""}</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Edit summary from Wikipedia stream**")
    st.markdown(f'<div class="stream-text">{escape(summary)}</div>', unsafe_allow_html=True)

    b1, b2 = st.columns([1, 1])
    if b1.button("Generate explanation", type="primary", use_container_width=True):
        with st.spinner("Generating explanation..."):
            out = api_post("/explain", {"edit_id": selected["id"]})
        st.session_state["explanation"] = out

    if b2.button("Send alert", use_container_width=True):
        with st.spinner("Sending alert..."):
            out = api_post(f"/notify/{selected['id']}")
        st.session_state["notify"] = out

    if "explanation" in st.session_state:
        out = st.session_state["explanation"]
        st.info(out.get("explanation", ""))
        st.caption(f"Model: {out.get('model')} - cached: {out.get('cached')}")

    if "notify" in st.session_state:
        out = st.session_state["notify"]
        (st.success if out.get("delivered") else st.warning)(out.get("message", ""))


def main() -> None:
    inject_css()
    st.session_state.setdefault("refresh_key", 0)
    st.session_state.setdefault("last_refresh_at", datetime.now(UTC))

    with st.sidebar:
        st.header("Demo Controls")
        if st.button("Refresh now", type="primary", use_container_width=True):
            st.session_state["refresh_key"] += 1
            st.session_state["last_refresh_at"] = datetime.now(UTC)
            st.cache_data.clear()
            st.rerun()

        st.caption(f"Last refresh: {time_ago(st.session_state['last_refresh_at'].isoformat())}")
        auto_refresh = st.toggle("Auto refresh", value=True)
        refresh_seconds = st.slider("Refresh seconds", 3, 30, 6)
        page_size = st.slider("Rows", 25, 200, 100, step=25)
        risk_choice = st.radio(
            "Risk filter",
            options=["All", "HIGH", "MEDIUM", "LOW"],
            index=0,
            horizontal=True,
        )
        scored_only = st.checkbox("Scored edits only", value=True)
        search = st.text_input("Search pages", placeholder="Article, editor, summary, rev id")
        st.divider()
        st.link_button("Wikimedia live stream", STREAM_URL, use_container_width=True)
        st.link_button("API docs", f"{API_BASE}/docs", use_container_width=True)
        st.link_button("MLflow UI", MLFLOW_URL, use_container_width=True)
        st.caption("Run `make live` to keep collector and Spark streaming active.")

    try:
        key = st.session_state["refresh_key"]
        health = load_health(key)
        stats = load_stats(key)
        edits = load_edits(
            page_size=page_size,
            risk_label=None if risk_choice == "All" else risk_choice,
            scored_only=scored_only,
            search=search,
            refresh_key=key,
        )
    except requests.RequestException as exc:
        st.error(f"Cannot reach WikiRisk API at {API_BASE}: {exc}")
        st.stop()

    st.session_state["last_refresh_at"] = datetime.now(UTC)
    render_header(health)
    render_metrics(stats)

    df = edits_to_frame(edits.get("items", []))
    st.subheader("Live Risk Feed")
    render_charts(df)

    table_col, detail_col = st.columns([1.35, 1])
    with table_col:
        if df.empty:
            st.info("No edits match the current filters.")
        else:
            display = df[
                [
                    "risk",
                    "score",
                    "page",
                    "delta",
                    "editor",
                    "age",
                    "timestamp",
                    "edit_summary",
                ]
            ].copy()
            display["score"] = display["score"].map(
                lambda v: "" if pd.isna(v) else f"{float(v):.3f}"
            )
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                height=430,
            )

    with detail_col:
        if df.empty:
            st.info("Search or loosen filters to inspect an edit.")
        else:
            labels = [
                f"{row.risk} - {row.page} - {row.age} - {row.score if pd.notna(row.score) else 'unscored'}"
                for row in df.itertuples()
            ]
            selected_label = st.selectbox("Inspect edit", labels)
            selected_index = labels.index(selected_label)
            selected_id = df.iloc[selected_index]["id"]
            selected = next(item for item in edits.get("items", []) if item.get("id") == selected_id)
            render_detail(selected)

    st.caption(f"Rendered at {datetime.now().strftime('%H:%M:%S')} - API: {API_BASE}")
    if auto_refresh:
        time.sleep(refresh_seconds)
        st.session_state["refresh_key"] += 1
        st.rerun()


if __name__ == "__main__":
    main()
