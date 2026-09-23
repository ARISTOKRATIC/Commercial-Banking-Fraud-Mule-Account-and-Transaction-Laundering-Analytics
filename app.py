"""
MuleWatch analyst dashboard (Streamlit). Run with: streamlit run app.py
from inside 07_dashboard_or_prototype/, after Sections 12/14 have produced the CSV snapshots.
"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MuleWatch Analyst Dashboard", layout="wide")
st.title("MuleWatch — Ranked Alert Queue")
st.caption("T17 · Real IBM AML transaction/graph data + synthetic auth/device/KYC/text layers "
           "(MetroTrust Bank is fictional)")

dash = pd.read_csv("analyst_dashboard_snapshot.csv")
st.dataframe(dash, use_container_width=True)

st.subheader("Alert detail")
selected = st.selectbox("Select an account to review", dash["account_id"])
row = dash[dash["account_id"] == selected].iloc[0]
st.metric("Fused risk score", f"{row.fused_risk_score:.1f}")
st.write("**Rationale:**", row.rationale)
st.write("**Recommended action:**", row.recommended_action)
st.info("Every account action remains a human analyst/MLRO decision (Charter Section 2.4).")
