"""
MuleWatch analyst dashboard (Streamlit).

Addresses C10 (Decision-support prototype): presents findings through a functional
dashboard that supports analyst decisions, not just displays a ranked list -- every
alert shows WHY it was flagged (a component-level breakdown of the three fusion
signals), and reconstructed evidence timelines are shown where available.

Run locally with:
    streamlit run app.py
from inside 07_dashboard_or_prototype/, after analyst_dashboard_full.csv (and any
incident_timeline_*.csv files) have been generated and placed alongside this file.

Also works when deployed on Streamlit Community Cloud, where the app's working
directory is the repository root rather than this file's folder -- paths below are
resolved relative to this script's own location, not the working directory.
"""
import glob
import os

import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DASH_CSV = os.path.join(APP_DIR, "analyst_dashboard_full.csv")

st.set_page_config(page_title="MuleWatch Analyst Dashboard", layout="wide")
st.title("MuleWatch — Analyst Decision-Support Dashboard")
st.caption("T17 · Real IBM AML transaction/graph data + synthetic auth/device/KYC/text layers "
           "(MetroTrust Bank is fictional). Every recommendation below is advisory — "
           "account actions remain a human analyst/MLRO decision (Charter §2.4).")

if not os.path.exists(DASH_CSV):
    st.error(
        f"Could not find analyst_dashboard_full.csv next to app.py.\n\n"
        f"Generate it (see build_dashboard_data.py) and commit it to the repository in "
        f"the same folder as app.py: {APP_DIR}"
    )
    st.stop()

dash = pd.read_csv(DASH_CSV)

# ---------------------------------------------------------------------------
# KPI summary row — gives an analyst the shape of today's alert queue at a glance
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accounts reviewed", f"{len(dash):,}")
c2.metric("Escalate to MLRO", int((dash["recommended_action"].str.startswith("ESCALATE")).sum()))
c3.metric("Investigate", int((dash["recommended_action"] == "Investigate + enhanced monitoring").sum()))
c4.metric("Monitor only", int((dash["recommended_action"] == "Monitor").sum()))

st.divider()

# ---------------------------------------------------------------------------
# Sidebar filters — an analyst triaging a queue needs to narrow it, not scroll it
# ---------------------------------------------------------------------------
st.sidebar.header("Filter the alert queue")
action_filter = st.sidebar.multiselect(
    "Recommended action", options=sorted(dash["recommended_action"].unique()),
    default=sorted(dash["recommended_action"].unique()))
risk_filter = st.sidebar.multiselect(
    "KYC risk rating", options=sorted(dash["risk_rating"].dropna().unique()),
    default=sorted(dash["risk_rating"].dropna().unique()))
min_score = st.sidebar.slider("Minimum fused risk score", 0, 100, 0)

filtered = dash[
    dash["recommended_action"].isin(action_filter)
    & dash["risk_rating"].isin(risk_filter)
    & (dash["fused_risk_score"] >= min_score)
].sort_values("fused_risk_score", ascending=False)

st.subheader(f"Ranked alert queue ({len(filtered):,} of {len(dash):,} accounts match your filters)")
st.dataframe(
    filtered[["account_id", "risk_rating", "fused_risk_score", "recommended_action", "rationale"]]
    .head(200).style.format({"fused_risk_score": "{:.1f}"}),
    use_container_width=True, height=320,
)

st.divider()

# ---------------------------------------------------------------------------
# Per-account detail panel — this is the part that actually explains a decision
# ---------------------------------------------------------------------------
st.subheader("Alert detail — why was this account flagged?")
options = filtered["account_id"].tolist() or dash["account_id"].tolist()
selected = st.selectbox("Select an account to review", options)
row = dash[dash["account_id"] == selected].iloc[0]

d1, d2 = st.columns([1, 2])
with d1:
    st.metric("Fused risk score", f"{row.fused_risk_score:.1f} / 100")
    st.write("**KYC risk rating:**", row.risk_rating)
    st.write("**Recommended action:**", row.recommended_action)
    if row.has_incident_timeline:
        st.warning("Corroborated by both the access-anomaly model AND the transaction "
                   "classifier — high-confidence incident.")
    st.write("**Rationale:**", row.rationale)

with d2:
    st.write("**Signal breakdown** — which of the three fused signals drove this score:")
    breakdown = pd.DataFrame({
        "signal": ["Transaction model (50% weight)", "Graph cluster membership (30% weight)",
                   "Case-note typology (20% weight)"],
        "value": [row.model_score, row.graph_score, row.text_score],
    }).set_index("signal")
    st.bar_chart(breakdown, height=220)

# ---------------------------------------------------------------------------
# Reconstructed incident timeline, where one exists for the selected account
# ---------------------------------------------------------------------------
timeline_path = os.path.join(APP_DIR, f"incident_timeline_{selected}.csv")
st.subheader("Reconstructed incident timeline")
if os.path.exists(timeline_path):
    tl = pd.read_csv(timeline_path)
    st.dataframe(tl, use_container_width=True)
else:
    st.info("No reconstructed evidence timeline is bundled for this specific account in this "
            "deployment. Timelines are generated on demand from the auth, beneficiary-change and "
            "transaction evidence in the notebook (Section 7/8) — see the notebook to reconstruct "
            "one for any account on request.")

st.divider()
st.caption("Every account action — hold, freeze, escalate, notify — remains a human analyst or "
           "MLRO decision. This dashboard ranks and explains; it does not act.")
