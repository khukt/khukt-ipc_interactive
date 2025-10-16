# streamlit_app.py
# TRUST DEMO — Persona-aware Incident Tab
# ------------------------------------------------------------
# Requirements (see requirements.txt):
#   streamlit, pandas, numpy, plotly
# Optional (installed in your reqs): scikit-learn, lightgbm, shap, pydeck
# ------------------------------------------------------------

import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ----------------------------
# Demo data seeding utilities
# ----------------------------
DEVICE_POOL = [f"edge-{i:03d}" for i in range(1, 21)]
SCENARIOS = [
    "Urban Jamming (Wi-Fi/5G)",
    "Access Breach Attempt",
    "GPS Spoofing (GNSS)",
    "Data Integrity Tamper",
    "Benign Fluctuation",
]
TYPE_LABELS = ["Jamming", "Access Breach", "GPS Spoof", "Data Tamper", "Unknown"]
SEVERITY_ORDER = ["Low", "Medium", "High"]


def seed_demo_incidents(n=60, start=None):
    """Create semi-realistic demo incidents for the session."""
    rng = random.Random(42)
    start = start or (datetime.utcnow() - timedelta(minutes=30))
    rows = []
    for i in range(n):
        t = start + timedelta(seconds=30 * i)
        device = rng.choice(DEVICE_POOL)
        scenario = rng.choices(SCENARIOS, weights=[3, 2, 2, 2, 1], k=1)[0]
        if scenario == "Urban Jamming (Wi-Fi/5G)":
            label = "Jamming"
            sev = rng.choices(SEVERITY_ORDER, weights=[1, 2, 4])[0]
            prob = min(1.0, rng.uniform(0.7, 0.98))
            pval = rng.uniform(0.0001, 0.07)
        elif scenario == "Access Breach Attempt":
            label = "Access Breach"
            sev = rng.choices(SEVERITY_ORDER, weights=[2, 3, 3])[0]
            prob = min(1.0, rng.uniform(0.6, 0.92))
            pval = rng.uniform(0.001, 0.12)
        elif scenario == "GPS Spoofing (GNSS)":
            label = "GPS Spoof"
            sev = rng.choices(SEVERITY_ORDER, weights=[1, 3, 3])[0]
            prob = min(1.0, rng.uniform(0.65, 0.95))
            pval = rng.uniform(0.0005, 0.09)
        elif scenario == "Data Integrity Tamper":
            label = "Data Tamper"
            sev = rng.choices(SEVERITY_ORDER, weights=[2, 3, 2])[0]
            prob = min(1.0, rng.uniform(0.55, 0.9))
            pval = rng.uniform(0.005, 0.2)
        else:
            label = "Unknown"
            sev = rng.choices(SEVERITY_ORDER, weights=[4, 2, 1])[0]
            prob = rng.uniform(0.1, 0.5)
            pval = rng.uniform(0.05, 0.5)

        rows.append(
            dict(
                id=f"INC-{i:05d}",
                tick=t,
                device_id=device,
                scenario=scenario,
                type_label=label,
                severity=sev,
                prob=round(prob, 3),
                p_value=round(pval, 4),
            )
        )
    return rows


# ----------------------------
# Persona helpers
# ----------------------------
def _sev_color(sev: str) -> str:
    return {"High": "#e03131", "Medium": "#f08c00", "Low": "#2f9e44"}.get(sev, "#868e96")


def _sev_badge(sev: str) -> str:
    color = _sev_color(sev)
    return (
        f"<span style='background:{color};color:white;"
        "padding:2px 10px;border-radius:999px;font-weight:700'>"
        f"{sev}</span>"
    )


def plain_english_summary(inc: dict) -> str:
    """
    One human sentence, no jargon — for End Users.
    """
    sev = inc.get("severity", "—")
    scen = inc.get("scenario", "—").split(" (")[0]
    tlabel = inc.get("type_label", "Unknown")
    pv = inc.get("p_value")
    ptxt = " with strong evidence" if (pv is not None and pv <= 0.05) else ""
    return (
        f"**{inc['device_id']}** likely has a **{tlabel if tlabel!='Unknown' else 'network issue'}** "
        f"during **{scen}**. Risk is **{sev}**{ptxt}."
    )


def quick_actions_for_end_user(inc: dict) -> list[str]:
    scen = inc.get("scenario", "")
    if "Jamming" in scen:
        return [
            "Move 50–100 m away and retry.",
            "If possible, switch to a different channel/network.",
        ]
    if "Access Breach" in scen:
        return [
            "Avoid unknown SSIDs/PLMNs.",
            "Connect only to approved networks and report suspicious prompts.",
        ]
    if "GPS Spoofing" in scen or "GNSS" in scen:
        return [
            "Use fallback navigation (IMU/UWB) if available.",
            "Reduce speed until GPS stabilizes.",
        ]
    if "Data Integrity" in scen or "Tamper" in scen:
        return [
            "Resend the last message once.",
            "If it repeats, notify operations with the incident ID.",
        ]
    return ["Retry once.", "If it repeats, notify operations."]


# ----------------------------
# Rendering: common sections
# ----------------------------
def build_anomaly_explanation(inc: dict) -> str:
    """Compact, readable explanation stub."""
    reason = (
        "We compared the device’s recent signals to its normal baseline. "
        "Several metrics deviated at the same time window, which is unlikely by chance."
    )
    return reason


def build_type_explanation(inc: dict) -> str:
    """Compact classification explanation stub."""
    label = inc.get("type_label", "Unknown")
    msg = {
        "Jamming": "High interference near the device with abrupt SNR and throughput drops.",
        "Access Breach": "Authentication anomalies and unusual association attempts.",
        "GPS Spoof": "Inconsistent satellite geometry and sudden heading jumps.",
        "Data Tamper": "Checksum mismatches and replay-like timing patterns.",
        "Unknown": "Patterns don’t match a known threat family with high confidence.",
    }[label]
    return msg


def render_device_inspector_from_incident(inc: dict, topk: int = 8, scope: str = "local"):
    """Placeholder for a deeper expert/AI view."""
    st.markdown("**Top contributing signals (pseudo)**")
    rng = random.Random(inc["id"])
    feats = [f"f{i}" for i in range(1, 21)]
    vals = [rng.uniform(-1.5, 1.5) for _ in feats]
    df = (
        pd.DataFrame({"feature": feats, "impact": vals})
        .assign(rank=lambda d: d["impact"].abs().rank(ascending=False, method="first"))
        .sort_values("rank")
        .head(topk)
        .drop(columns="rank")
    )
    fig = px.bar(df, x="impact", y="feature", orientation="h", title="Signal impacts")
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Persona-specific renderers
# ----------------------------
def render_end_user_card(inc: dict):
    st.markdown("#### What’s happening (simple)")
    st.markdown(plain_english_summary(inc))

    sev = inc.get("severity", "—")
    pv = inc.get("p_value")
    pv_str = f"{pv:.3f}" if isinstance(pv, (int, float)) else "—"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Severity**<br/>{_sev_badge(sev)}", unsafe_allow_html=True)
    with c2:
        st.metric("Confidence (prob.)", f"{inc.get('prob', float('nan')):.2f}")
    with c3:
        st.metric("Evidence (p-value)", pv_str)

    st.markdown("#### What to do now")
    for step in quick_actions_for_end_user(inc):
        st.write(f"• {step}")

    with st.expander("Why this is flagged (optional)"):
        st.markdown(build_anomaly_explanation(inc))
        st.markdown(build_type_explanation(inc))


def render_executive_overview(incidents: list[dict]):
    if not incidents:
        st.info("No incidents yet. When they appear, this panel summarizes risk at a glance.")
        return
    df = pd.DataFrame(incidents)
    if "type_label" not in df.columns:
        df["type_label"] = "Unknown"
    if "severity" not in df.columns:
        df["severity"] = "—"

    st.subheader("Executive overview")

    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        trend = df.groupby(pd.Grouper(key="tick", freq="2T")).size().reset_index(name="count")
        if trend.empty:
            trend = pd.DataFrame({"tick": [df["tick"].min()], "count": [len(df)]})
        fig_trend = px.line(trend, x="tick", y="count", markers=True, title="Incident Trend")
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        by_type = df["type_label"].value_counts().reset_index()
        by_type.columns = ["Type", "Incidents"]
        fig_type = px.bar(by_type, x="Type", y="Incidents", title="By Type")
        st.plotly_chart(fig_type, use_container_width=True)

    c3, c4 = st.columns([1, 1], gap="large")
    with c3:
        by_sev = df["severity"].value_counts().reset_index()
        by_sev.columns = ["Severity", "Incidents"]
        fig_sev = px.bar(by_sev, x="Severity", y="Incidents", title="By Severity")
        st.plotly_chart(fig_sev, use_container_width=True)
    with c4:
        top_dev = (
            df.groupby("device_id").size().sort_values(ascending=False).head(5).reset_index(name="Incidents")
        )
        fig_dev = px.bar(top_dev, x="device_id", y="Incidents", title="Top Devices")
        st.plotly_chart(fig_dev, use_container_width=True)

    k1, k2, k3 = st.columns(3)
    k1.metric("Incidents (session)", len(df))
    k2.metric("Unique devices", df["device_id"].nunique())
    k3.metric("Active scenario", str(df.tail(1)["scenario"].values[0]) if not df.empty else "—")


def render_domain_expert(inc: dict):
    st.markdown("#### Signals & Playbook")
    st.write("- Verify channel conditions and retry with alternate band.")
    st.write("- Correlate with neighbor devices in 100 m radius.")
    st.write("- If persistent, escalate to spectrum analysis.")
    st.markdown("#### Local model factors")
    render_device_inspector_from_incident(inc, topk=8, scope="local")


def render_regulator(inc: dict):
    st.markdown("#### Assurance, evidence & export")
    st.write(
        "This incident is backed by statistical deviation checks and a supervised classifier. "
        "You can export the packet of evidence below."
    )
    meta = {
        "Incident ID": inc["id"],
        "Device": inc["device_id"],
        "Scenario": inc["scenario"],
        "Type": inc["type_label"],
        "Severity": inc["severity"],
        "Probability": inc["prob"],
        "p-value": inc["p_value"],
        "Timestamp (UTC)": inc["tick"].strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.json(meta)
    st.download_button(
        "Export incident JSON",
        data=pd.Series(meta).to_json(indent=2),
        file_name=f"{inc['id']}.json",
        mime="application/json",
    )


def render_ai_builder(inc: dict):
    st.markdown("#### Model factors (local)")
    render_device_inspector_from_incident(inc, topk=10, scope="local")
    with st.expander("Standardized feature vector (z-scores)"):
        rng = np.random.default_rng(abs(hash(inc["id"])) % (2**32))
        feats = [f"f{i}" for i in range(1, 31)]
        vals = rng.normal(loc=0, scale=1, size=len(feats))
        df = pd.DataFrame({"feature": feats, "z": vals}).sort_values("feature")
        st.dataframe(df, use_container_width=True)


# ----------------------------
# Incident list + detail
# ----------------------------
def incident_filter_bar(df: pd.DataFrame) -> pd.DataFrame:
    c1, c2, c3 = st.columns(3)
    with c1:
        sev_sel = st.multiselect("Severity", SEVERITY_ORDER, default=SEVERITY_ORDER)
    with c2:
        types = sorted(df["type_label"].dropna().unique().tolist())
        type_sel = st.multiselect("Type", types, default=types)
    with c3:
        devs = sorted(df["device_id"].dropna().unique().tolist())
        dev_sel = st.multiselect("Device", devs, default=devs)

    mask = df["severity"].isin(sev_sel) & df["type_label"].isin(type_sel) & df["device_id"].isin(dev_sel)
    return df.loc[mask].copy()


def render_incident_list(incidents: list[dict]) -> pd.DataFrame:
    if not incidents:
        st.info("No incidents recorded yet.")
        return pd.DataFrame()
    df = pd.DataFrame(incidents)
    df = df.sort_values("tick", ascending=False)
    df_disp = df.assign(
        when=df["tick"].dt.strftime("%H:%M:%S"),
        sev=lambda d: d["severity"].map(lambda s: f"🟥 {s}" if s == "High" else ("🟧 Medium" if s == "Medium" else "🟩 Low")),
    )[
        ["id", "when", "device_id", "scenario", "type_label", "sev", "prob", "p_value"]
    ].rename(
        columns={
            "id": "Incident",
            "when": "Time (UTC)",
            "device_id": "Device",
            "scenario": "Scenario",
            "type_label": "Type",
            "sev": "Severity",
            "prob": "Prob.",
            "p_value": "p-value",
        }
    )
    st.dataframe(df_disp, use_container_width=True, height=420)
    return df.sort_values("tick")


def render_incident_detail(role: str, inc: dict):
    st.markdown(f"### {inc['id']} — {inc['device_id']}")
    st.caption(inc["tick"].strftime("%Y-%m-%d %H:%M:%S UTC"))

    if role == "End User":
        render_end_user_card(inc)
    elif role == "Executive":
        # Executives see the overview at top of tab; detail shows compact facts
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Severity", inc["severity"])
        c2.metric("Type", inc["type_label"])
        c3.metric("Prob.", f"{inc['prob']:.2f}")
        c4.metric("p-value", f"{inc['p_value']:.3f}")
        with st.expander("Summary"):
            st.write(plain_english_summary(inc))
            st.write(build_type_explanation(inc))
    elif role == "Domain Expert":
        render_domain_expert(inc)
    elif role == "Regulator":
        render_regulator(inc)
    elif role == "AI Builder":
        render_ai_builder(inc)
    else:
        st.write("Unsupported role.")


# ----------------------------
# App layout
# ----------------------------
st.set_page_config(page_title="TRUST DEMO", layout="wide")

# Seed session incidents (demo)
if "incidents" not in st.session_state:
    st.session_state.incidents = seed_demo_incidents()

st.sidebar.title("TRUST DEMO")
role = st.sidebar.selectbox(
    "Viewer role",
    ["End User", "Executive", "Domain Expert", "Regulator", "AI Builder"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.write("Use the filters in the Incident tab to narrow down the list.")

tabs = st.tabs(["🧯 Incident", "🖥️ Devices", "⚙️ Settings"])

# ----------------------------
# Incident tab
# ----------------------------
with tabs[0]:
    st.header("Incident Center")

    # Executive overview at the very top (only for Execs)
    if role == "Executive":
        render_executive_overview(st.session_state.incidents)
        st.markdown("---")

    df_all = pd.DataFrame(st.session_state.incidents)
    df_all = incident_filter_bar(df_all)
    df_sorted = render_incident_list(df_all.to_dict(orient="records"))

    # Select an incident to view detail (newest by default)
    if not df_sorted.empty:
        latest_id = df_sorted.tail(1)["id"].values[0]
        sel = st.selectbox(
            "Open incident",
            df_sorted["id"].tolist(),
            index=df_sorted.index.get_loc(df_sorted.index[-1]),
        )
        inc = df_sorted.loc[df_sorted["id"] == sel].iloc[0].to_dict()
        st.markdown("---")
        render_incident_detail(role, inc)
    else:
        st.info("No incidents match the current filters.")

# ----------------------------
# Devices tab (minimal demo)
# ----------------------------
with tabs[1]:
    st.header("Devices")
    dev_counts = (
        pd.DataFrame(st.session_state.incidents)
        .groupby("device_id")
        .size()
        .reset_index(name="Incidents")
        .sort_values("Incidents", ascending=False)
    )
    st.dataframe(dev_counts, use_container_width=True, height=420)

# ----------------------------
# Settings tab (minimal demo)
# ----------------------------
with tabs[2]:
    st.header("Settings")
    st.write("Persona-aware UI is on by default for the Incident tab.")
    st.checkbox("Enable demo incident seeding", value=True, disabled=True)
    st.caption("This is a demo build; connect your real pipeline to replace the seeded data.")
