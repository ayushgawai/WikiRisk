"""
WikiRisk – Streamlit Dashboard

Production-grade, clean white/light UI with:
  • Real-time edit feed with risk badges
  • Risk-level filter  (ALL / HIGH / MEDIUM / LOW)
  • Aggregate KPI metrics (auto-refreshes every 15 s)
  • Per-edit detail panel with AI explanation
  • Risk score distribution chart
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WikiRisk · Edit Intelligence",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Custom CSS (minimal, professional) ───────────────────────────────────────
st.markdown(
    """
    <style>
    /* Clean sans-serif body */
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Risk badges */
    .badge-high   { background:#fee2e2; color:#991b1b; border-radius:4px;
                    padding:2px 8px; font-size:0.75rem; font-weight:600; }
    .badge-medium { background:#fef3c7; color:#92400e; border-radius:4px;
                    padding:2px 8px; font-size:0.75rem; font-weight:600; }
    .badge-low    { background:#dcfce7; color:#166534; border-radius:4px;
                    padding:2px 8px; font-size:0.75rem; font-weight:600; }
    .badge-na     { background:#f1f5f9; color:#64748b; border-radius:4px;
                    padding:2px 8px; font-size:0.75rem; font-weight:600; }

    /* KPI card */
    .kpi-card { background:#f8fafc; border:1px solid #e2e8f0;
                border-radius:8px; padding:16px 20px; text-align:center; }
    .kpi-value { font-size:2rem; font-weight:700; color:#0f172a; }
    .kpi-label { font-size:0.8rem; color:#64748b; margin-top:2px; }

    /* Explanation box */
    .explain-box { background:#f0f9ff; border-left:3px solid #0ea5e9;
                   padding:12px 16px; border-radius:0 6px 6px 0;
                   font-size:0.9rem; line-height:1.6; }

    /* Table row hover */
    [data-testid="stDataFrame"] tr:hover td { background:#f0f9ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── API helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def api_get(path: str, params: dict | None = None) -> dict | list | None:
    try:
        with httpx.Client(timeout=8) as client:
            r = client.get(f"{API_URL}{path}", params=params)
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        st.warning(f"API error ({path}): {exc}", icon="⚠")
        return None


def api_explain(edit_id: str) -> Optional[str]:
    try:
        with httpx.Client(timeout=30) as client:
            r = client.get(f"{API_URL}/explain/{edit_id}")
            r.raise_for_status()
            return r.json().get("explanation", "")
    except Exception as exc:
        return f"Could not generate explanation: {exc}"


def _badge(label: Optional[str]) -> str:
    if not label:
        return '<span class="badge-na">UNSCORED</span>'
    cls = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(
        label.upper(), "badge-na"
    )
    return f'<span class="{cls}">{label.upper()}</span>'


def _score_color(score: Optional[float]) -> str:
    if score is None:
        return "#94a3b8"
    if score >= 0.7:
        return "#dc2626"
    if score >= 0.4:
        return "#d97706"
    return "#16a34a"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### WikiRisk")
    st.caption("Real-Time Edit Intelligence")
    st.divider()

    risk_filter = st.radio(
        "Filter by Risk",
        options=["ALL", "HIGH", "MEDIUM", "LOW"],
        index=0,
        horizontal=False,
    )
    scored_only = st.toggle("Scored edits only", value=False)
    page_size = st.select_slider("Edits per page", options=[25, 50, 100], value=50)

    st.divider()
    auto_refresh = st.toggle("Auto-refresh (15 s)", value=True)

    st.divider()
    st.markdown("**Links**")
    st.markdown(f"[FastAPI Docs]({API_URL}/docs)")
    st.markdown("[Wikimedia EventStreams](https://stream.wikimedia.org)")

# ── Main header ───────────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([4, 1])
with col_title:
    st.markdown("## Wikipedia Edit Risk Monitor")
    st.caption(
        "Live risk scoring of English Wikipedia edits · "
        f"Powered by SparkML + MLflow"
    )
with col_refresh:
    if st.button("↺ Refresh", use_container_width=True):
        st.cache_data.clear()

# ── KPI row ───────────────────────────────────────────────────────────────────
stats = api_get("/stats") or {}

total = int(stats.get("total_edits") or 0)
high = int(stats.get("high_risk") or 0)
medium = int(stats.get("medium_risk") or 0)
low = int(stats.get("low_risk") or 0)
unscored = int(stats.get("unscored") or 0)
avg_score = float(stats.get("avg_risk_score") or 0.0)

k1, k2, k3, k4, k5 = st.columns(5)

def _kpi(col, value, label, color=None):
    with col:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-value" style="color:{color or '#0f172a'}">{value}</div>
            <div class="kpi-label">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

_kpi(k1, f"{total:,}", "Total Edits")
_kpi(k2, f"{high:,}", "High Risk", "#dc2626")
_kpi(k3, f"{medium:,}", "Medium Risk", "#d97706")
_kpi(k4, f"{low:,}", "Low Risk", "#16a34a")
_kpi(k5, f"{avg_score:.2%}", "Avg Risk Score")

st.divider()

# ── Charts row ────────────────────────────────────────────────────────────────
chart_col, _ = st.columns([2, 1])

with chart_col:
    if total > 0:
        fig = go.Figure(
            go.Pie(
                labels=["High", "Medium", "Low", "Unscored"],
                values=[high, medium, low, unscored],
                hole=0.6,
                marker_colors=["#dc2626", "#f59e0b", "#22c55e", "#cbd5e1"],
                textinfo="percent+label",
                showlegend=False,
            )
        )
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            height=220,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12),
            annotations=[
                dict(
                    text=f"<b>{total:,}</b><br>edits",
                    x=0.5, y=0.5,
                    font_size=14,
                    showarrow=False,
                )
            ],
        )
        st.plotly_chart(fig, use_container_width=True, key="pie_chart")

# ── Edit feed ─────────────────────────────────────────────────────────────────
st.markdown("### Live Edit Feed")

params: dict = {"page_size": page_size, "scored_only": scored_only}
if risk_filter != "ALL":
    params["risk_label"] = risk_filter

data = api_get("/edits/recent", params=params) or {}
items = data.get("items", [])
total_items = int(data.get("total") or 0)

st.caption(
    f"Showing {len(items)} of {total_items:,} edits "
    + (f"filtered to **{risk_filter}**" if risk_filter != "ALL" else "")
)

if not items:
    st.info("No edits yet. The collector starts populating data in real time.")
    st.stop()

# ── Table ─────────────────────────────────────────────────────────────────────
for edit in items:
    score = edit.get("risk_score")
    label = edit.get("risk_label")
    title = edit.get("page_title", "Unknown")
    user = edit.get("user", "—")
    comment = edit.get("comment", "") or "—"
    delta = int(edit.get("length_delta") or 0)
    ts = edit.get("created_at", "")[:16].replace("T", " ")
    edit_id = edit.get("id", "")

    score_str = f"{score:.2%}" if score is not None else "—"
    delta_str = (f"+{delta}" if delta >= 0 else str(delta)) + " B"

    with st.expander(
        f"{_badge(label)} &nbsp; **{title}** &nbsp; · &nbsp; "
        f"`{score_str}` &nbsp; {delta_str} &nbsp; · &nbsp; _{ts}_",
        expanded=False,
    ):
        col_l, col_r = st.columns([3, 1])

        with col_l:
            st.markdown(f"**User:** `{user}` {'_(anon)_' if edit.get('is_anon') else ''}")
            st.markdown(f"**Comment:** {comment[:300] or '—'}")
            rev_id = edit.get("rev_id", "")
            if rev_id:
                diff_url = f"https://en.wikipedia.org/w/index.php?diff={rev_id}"
                st.markdown(f"[View diff on Wikipedia ↗]({diff_url})")

        with col_r:
            if score is not None:
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=round(score * 100, 1),
                        number={"suffix": "%", "font": {"size": 18}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 0},
                            "bar": {"color": _score_color(score)},
                            "bgcolor": "#f1f5f9",
                            "steps": [
                                {"range": [0, 40], "color": "#dcfce7"},
                                {"range": [40, 70], "color": "#fef3c7"},
                                {"range": [70, 100], "color": "#fee2e2"},
                            ],
                            "threshold": {
                                "line": {"color": "#0f172a", "width": 2},
                                "thickness": 0.75,
                                "value": round(score * 100, 1),
                            },
                        },
                    )
                )
                gauge.update_layout(
                    height=160,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11),
                )
                st.plotly_chart(gauge, use_container_width=True, key=f"gauge_{edit_id}")

        # ── AI explanation ─────────────────────────────────────────────────
        st.markdown("---")
        if st.button(
            "✦ Explain with AI",
            key=f"explain_{edit_id}",
            use_container_width=False,
        ):
            with st.spinner("Generating explanation…"):
                text = api_explain(edit_id)
            st.markdown(
                f'<div class="explain-box">{text}</div>',
                unsafe_allow_html=True,
            )

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(15)
    st.cache_data.clear()
    st.rerun()
