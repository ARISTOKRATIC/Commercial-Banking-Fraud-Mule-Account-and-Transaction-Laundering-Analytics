"""
MuleWatch analyst dashboard (Streamlit).

Addresses C10 (Decision-support prototype): a functional dashboard that supports
analyst decisions through three guided views -- an Overview separating trend
signal from raw noise, a filterable Alert Queue, and an Account Detail view that
explains a specific decision (signal breakdown, network neighborhood, evidence
timeline) rather than just displaying a ranked list.

Run locally with:
    streamlit run app.py
from inside 07_dashboard_or_prototype/, with analyst_dashboard_full.csv,
daily_trend.csv, graph_edges.csv, graph_scores.csv and any incident_timeline_*.csv
files placed alongside this file.

Paths below are resolved relative to this script location (not the working
directory), so this also works when deployed on Streamlit Community Cloud, where
the working directory is the repository root.
"""
import os

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _path(name):
    return os.path.join(APP_DIR, name)


def _require(name):
    p = _path(name)
    if not os.path.exists(p):
        st.error(f"Missing {name} next to app.py (expected at {p}). "
                 f"Generate it from the notebook and commit it alongside app.py.")
        st.stop()
    return p


st.set_page_config(page_title="MuleWatch Analyst Dashboard", layout="wide")
st.title("MuleWatch — Analyst Decision-Support Dashboard")
st.caption("T17 · Real IBM AML transaction/graph data + synthetic auth/device/KYC/text layers "
           "(MetroTrust Bank is fictional). Every recommendation below is advisory — "
           "account actions remain a human analyst/MLRO decision (Charter §2.4).")

dash = pd.read_csv(_require("analyst_dashboard_full.csv"))
daily = pd.read_csv(_require("daily_trend.csv"), parse_dates=["date"])
edges = pd.read_csv(_require("graph_edges.csv"))
graph_scores = pd.read_csv(_require("graph_scores.csv"))

ACTION_COLOR = {
    "ESCALATE to MLRO + hold outbound transfers": "#C0392B",
    "Investigate + enhanced monitoring": "#D68910",
    "Monitor": "#7F8C8D",
}

tab_overview, tab_queue, tab_detail = st.tabs(
    ["Overview", "Alert Queue", "Account Detail"])

# ===========================================================================
# TAB 1 — Overview: trend signal separated from the raw, noisy account list
# ===========================================================================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts reviewed", f"{len(dash):,}")
    c2.metric("Escalate to MLRO", int((dash["recommended_action"].str.startswith("ESCALATE")).sum()))
    c3.metric("Investigate", int((dash["recommended_action"] == "Investigate + enhanced monitoring").sum()))
    c4.metric("Monitor only", int((dash["recommended_action"] == "Monitor").sum()))

    st.subheader("Daily transaction volume vs. flagged laundering activity")
    st.caption("Separates the overall transaction trend (noise) from the flagged-activity trend "
               "(signal) across the real data window — a spike in flagged volume that does not "
               "track the overall volume trend is what an analyst should look at first.")
    trend = daily.set_index("date")[["total_transactions", "flagged_transactions"]]
    st.line_chart(trend, height=280)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Risk-score distribution")
        st.caption("Where the population actually sits — most accounts cluster near zero; "
                   "the escalation-worthy tail is small and should stay small.")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.hist(dash["fused_risk_score"], bins=40, color="#2C5C9E")
        ax.set_xlabel("Fused risk score"); ax.set_ylabel("Accounts (log scale)")
        ax.set_yscale("log")
        st.pyplot(fig, width="stretch")
    with col_b:
        st.subheader("Recommended action breakdown")
        counts = dash["recommended_action"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        ax2.bar(counts.index, counts.values,
                color=[ACTION_COLOR.get(k, "#7F8C8D") for k in counts.index])
        ax2.set_ylabel("Accounts"); ax2.set_yscale("log")
        plt.setp(ax2.get_xticklabels(), rotation=15, ha="right", fontsize=8)
        st.pyplot(fig2, width="stretch")

# ===========================================================================
# TAB 2 — Alert Queue: filterable, sortable, the operational triage list
# ===========================================================================
with tab_queue:
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
        .head(300).style.format({"fused_risk_score": "{:.1f}"})
        .background_gradient(subset=["fused_risk_score"], cmap="Reds"),
        width="stretch", height=420,
    )
    st.session_state["queue_options"] = filtered["account_id"].tolist()

# ===========================================================================
# TAB 3 — Account Detail: explains one specific decision end-to-end
# ===========================================================================
with tab_detail:
    options = st.session_state.get("queue_options") or dash["account_id"].tolist()
    selected = st.selectbox("Select an account to review", options, key="detail_select")
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
            "signal": ["Transaction model (50%)", "Graph cluster (30%)", "Case-note typology (20%)"],
            "value": [row.model_score, row.graph_score, row.text_score],
        }).set_index("signal")
        st.bar_chart(breakdown, height=200)

    st.subheader("Account network neighborhood")
    st.caption("Direct counterparty, device and beneficiary links for this account in the "
               "transaction graph — the same structure Section 7's Louvain clustering runs over.")
    neighbor_edges = edges[(edges["source"] == selected) | (edges["target"] == selected)]
    if len(neighbor_edges) == 0:
        st.info("This account has no graph edges recorded (isolated node).")
    else:
        G = nx.Graph()
        for _, e in neighbor_edges.iterrows():
            G.add_edge(e["source"], e["target"], edge_type=e["edge_type"])
        node_colors = []
        for n in G.nodes():
            if n == selected:
                node_colors.append("#1A1A1A")
            else:
                match = dash[dash["account_id"] == n]
                if len(match) and match.iloc[0]["recommended_action"].startswith("ESCALATE"):
                    node_colors.append("#C0392B")
                elif len(match) and match.iloc[0]["recommended_action"] == "Investigate + enhanced monitoring":
                    node_colors.append("#D68910")
                else:
                    node_colors.append("#AAB7B8")
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        pos = nx.spring_layout(G, seed=42, k=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax3, alpha=0.4)
        nx.draw_networkx_nodes(G, pos, ax=ax3, node_color=node_colors, node_size=260)
        labels = {n: (n if n == selected else "") for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax3, font_size=7)
        ax3.set_axis_off()
        st.pyplot(fig3, width="stretch")
        st.caption("Black = selected account · red = other escalation-priority accounts in its "
                   "neighborhood · orange = investigate-priority · grey = monitor-only.")

    timeline_path = _path(f"incident_timeline_{selected}.csv")
    st.subheader("Reconstructed incident timeline")
    if os.path.exists(timeline_path):
        st.dataframe(pd.read_csv(timeline_path), width="stretch")
    else:
        st.info("No reconstructed evidence timeline is bundled for this specific account in this "
                "deployment. Timelines are generated on demand from the auth, beneficiary-change "
                "and transaction evidence in the notebook (Section 7/8).")

st.divider()
st.caption("Every account action — hold, freeze, escalate, notify — remains a human analyst or "
           "MLRO decision. This dashboard ranks and explains; it does not act.")
