# streamlit_app.py — Compact UI v2.1 for TRUST AI Wireless Threats
# ---------------------------------------------------------------
# Drop-in simplified layout with richer content per tab, persona-aware hints,
# and a brighter map style. Safe placeholders are provided; swap hooks with your
# real model/training/SHAP logic as needed.
#
# Usage: streamlit run streamlit_app.py

from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

# -------------------------
# Global Config
# -------------------------
st.set_page_config(
    page_title="TRUST AI — Wireless Threat Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0F766E"  # teal-700
ACCENT = "#2563EB"   # blue-600

# Persona guidance (role-aware short tips)
ROLE_TIPS = {
    "End User": "Focus on the Incidents tab. Use the guidance box to see what to do next.",
    "Domain Expert": "Open Insights → Feature importance and calibration. Inspect per-incident SHAP when available.",
    "Regulator": "Go to Governance for model card, audit log, policy & evidence downloads.",
    "AI Builder": "Use Insights to retrain/tune thresholds and export artifacts.",
    "Executive": "Glance KPIs on Overview; check trend sparkline and incident counts.",
}

# -------------------------
# Lightweight data generators (replace with back-end in production)
# -------------------------
@st.cache_data(show_spinner=False)
def get_device_inventory(n_devices: int = 30, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = rng.choice(["AMR", "Truck", "Sensor", "Gateway"], size=n_devices, p=[0.4, 0.2, 0.3, 0.1])
    base_lat, base_lon = 62.3908, 17.3069  # Sundsvall approx
    lats = base_lat + rng.normal(0, 0.02, n_devices)
    lons = base_lon + rng.normal(0, 0.04, n_devices)
    return pd.DataFrame({
        "device_id": [f"dev_{i:03d}" for i in range(n_devices)],
        "type": types,
        "lat": lats,
        "lon": lons,
    })

@st.cache_data(show_spinner=False)
def get_incidents(devices: pd.DataFrame, scenario: str, risk_thr: float) -> pd.DataFrame:
    rng = np.random.default_rng(42 if scenario == "Normal" else 21)
    n = len(devices)
    base = rng.beta(1.5, 4.0, n)
    bump = {
        "Normal": 0.0,
        "Jamming": 0.25,
        "Access Breach": 0.20,
        "GPS Spoofing": 0.22,
        "Data Tamper": 0.18,
    }.get(scenario, 0.0)
    risk = np.clip(base + bump + rng.normal(0, 0.05, n), 0, 1)
    attack_type = rng.choice(["—", "Jamming", "Access Breach", "GPS Spoofing", "Data Tamper"], n, p=[0.55, 0.12, 0.12, 0.11, 0.10])
    df = devices.copy()
    df["risk"] = risk
    df["type_pred"] = attack_type
    df["p_value"] = np.clip(1 - risk, 0, 1)  # pretend conformal p
    df["is_incident"] = df["risk"] >= risk_thr
    # Derive confidence bucket
    df["confidence"] = pd.cut(1 - df["p_value"], bins=[0, 0.6, 0.8, 1.01], labels=["low", "medium", "high"], include_lowest=True)
    return df.sort_values("risk", ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def get_model_metrics(scenario: str) -> dict:
    base = {
        "auc": 0.82 if scenario != "Normal" else 0.78,
        "brier": 0.19,
        "fleet_risk": 0.33 if scenario == "Normal" else 0.55,
        "last_train_s": 4.2,
        "version": "1.3.0",
    }
    return base

# -------------------------
# Helpers
# -------------------------

def risk_color(r: float) -> list[int]:
    # green → yellow → red
    if r < 0.33:
        return [16, 185, 129, 120]
    if r < 0.66:
        return [234, 179, 8, 150]
    return [239, 68, 68, 180]


def persona_hint(role: str):
    tip = ROLE_TIPS.get(role)
    if tip:
        st.caption(f"**{role} tip:** {tip}")

# -------------------------
# Sidebar
# -------------------------

def build_sidebar() -> dict:
    st.sidebar.header("Demo Controls")

    comms = st.sidebar.selectbox(
        "Comms profile",
        ["Road (Cellular 5G/LTE)", "Campus (Wi‑Fi/private‑5G)", "Harbor (Mixed)"],
        index=0,
    )

    scenario = st.sidebar.selectbox(
        "Scenario",
        ["Normal", "Jamming", "Access Breach", "GPS Spoofing", "Data Tamper"],
        index=0,
    )

    speed = st.sidebar.slider("Playback speed (ticks/refresh)", 1, 5, 3)
    auto = st.sidebar.toggle("Auto stream", value=False)
    if st.sidebar.button("Reset session"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Model")
    use_conformal = st.sidebar.toggle("Conformal risk (calibrated p-value)", value=True)
    thr = st.sidebar.slider("Incident threshold (model prob.)", 0.05, 0.95, 0.75, 0.01)

    st.sidebar.divider()
    st.sidebar.subheader("Display & Access")
    show_map = st.sidebar.toggle("Show geospatial map", True)
    show_heat = st.sidebar.toggle("Show risk heatmap", True)

    st.sidebar.divider()
    st.sidebar.subheader("Viewer role")
    role = st.sidebar.selectbox("Viewer role", ["End User", "Domain Expert", "Regulator", "AI Builder", "Executive"], index=3)

    st.sidebar.divider()
    help_mode = st.sidebar.toggle("Help mode (inline hints)", value=False)
    eu_banner = st.sidebar.toggle("Show EU AI Act status banner", value=True)

    return {
        "comms": comms,
        "scenario": scenario,
        "speed": speed,
        "auto": auto,
        "use_conformal": use_conformal,
        "thr": thr,
        "show_map": show_map,
        "show_heat": show_heat,
        "role": role,
        "help": help_mode,
        "eu": eu_banner,
    }

# -------------------------
# Header & Banners
# -------------------------

def header(role: str):
    st.title("TRUST AI — Wireless Threat Detection Demo")
    st.caption(
        "Fleet risk & incidents • LightGBM + SHAP • Conformal evidence • Persona-aware explanations • Cached models"
    )
    st.markdown(
        f"<div style='background:{ACCENT}10;border:1px solid {ACCENT}33;padding:10px 14px;border-radius:10px'>"
        f"<strong>Viewer role:</strong> {role} • <em>Overview · Incidents · Insights · Governance</em>"
        "</div>",
        unsafe_allow_html=True,
    )


def quick_help(help_on: bool):
    if help_on:
        st.info("Flow: Pick scenario → Watch KPIs → Open Incidents → Drill into Insights → Review Governance.")


def compact_banners(eu_on: bool):
    if eu_on:
        st.success(
            "EU AI Act status: Minimal-risk demo (synthetic, not a safety control component). "
            "Integration into safety/control may elevate risk class.",
            icon="✅",
        )
    st.caption("Cached model active. Use **Train / Retrain** in Insights after data changes.")

# -------------------------
# KPI Row
# -------------------------

def kpi_row(devices_df: pd.DataFrame, incidents_df: pd.DataFrame, metrics: dict):
    dcount = len(devices_df)
    icount = int((incidents_df["is_incident"]).sum())
    auc = metrics.get("auc", np.nan)
    frisk = metrics.get("fleet_risk", np.nan)

    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
    c1.metric("Devices", dcount)
    c2.metric("Incidents", icount)
    c3.metric("Model AUC", f"{auc:.2f}")
    c4.metric("Fleet Risk", f"{frisk:.2f}")
    c5.metric("Model ver.", metrics.get("version", "—"))

# -------------------------
# Tabs
# -------------------------

def render_overview(devices: pd.DataFrame, incidents: pd.DataFrame, show_map: bool, show_heat: bool, role: str):
    st.subheader("Overview")
    persona_hint(role)

    if show_map and not devices.empty:
        # Map style → brighter
        try:
            map_style = "mapbox://styles/mapbox/light-v9"
        except Exception:
            map_style = None

        # Color per risk
        devices_vis = devices.copy()
        devices_vis["risk_color"] = devices_vis["risk"].fillna(0).apply(lambda r: risk_color(r)) if "risk" in devices_vis else [risk_color(0.2)]*len(devices_vis)

        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                data=devices_vis,
                get_position='[lon, lat]',
                get_radius=60,
                radius_min_pixels=2,
                radius_max_pixels=20,
                get_fill_color='risk_color',
                pickable=True,
            )
        ]
        if show_heat:
            layers.append(
                pdk.Layer(
                    "HeatmapLayer",
                    data=incidents if not incidents.empty else devices,
                    get_position='[lon, lat]',
                    aggregation="MEAN",
                    opacity=0.35,
                )
            )
        view_state = pdk.ViewState(latitude=devices["lat"].mean(), longitude=devices["lon"].mean(), zoom=11)
        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "{device_id} ({type})
Risk: {risk}"},
            map_style=map_style,
        ))
    else:
        st.warning("Map disabled or no data available.")

    # Trend sparkline
    hist = pd.DataFrame({"tick": np.arange(40), "fleet_risk": np.clip(np.cumsum(np.random.randn(40))*0.01+0.35, 0, 1)})
    fig = px.area(hist, x="tick", y="fleet_risk", height=180, title="Fleet risk — last 40 ticks")
    fig.update_layout(margin=dict(l=0, r=0, t=34, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_incidents(full_df: pd.DataFrame, thr: float, role: str):
    st.subheader("Incidents")
    persona_hint(role)

    incidents = full_df[full_df["is_incident"]]
    if incidents.empty:
        st.success("No incidents above threshold. Lower the threshold or switch scenario.")
        return

    show_cols = ["device_id", "type", "risk", "p_value", "type_pred", "confidence"]
    grid = incidents[show_cols].copy()
    grid.rename(columns={"type": "device_type", "type_pred": "attack_type"}, inplace=True)
    st.dataframe(grid, use_container_width=True, hide_index=True)

    # Inspector panel
    sel = st.selectbox("Inspect device", options=incidents["device_id"].tolist())
    row = incidents[incidents["device_id"] == sel].iloc[0]

    c1, c2, c3 = st.columns([1,1,1])
    c1.metric("Risk", f"{row.risk:.2f}")
    c2.metric("Conformal p", f"{row.p_value:.2f}")
    c3.metric("Confidence", str(row.confidence))

    st.markdown("**Role-aware guidance**")
    if role == "End User":
        st.info("Move the asset to a safe zone and follow site SOP. If jamming suspected, avoid remote control.")
    elif role == "Domain Expert":
        st.info("Check RF telemetry (SNR/SINR, BLER) and access logs. Compare with baseline week-over-week.")
    elif role == "Regulator":
        st.info("Record this incident ID in the audit log and capture evidence bundle (governance tab).")
    elif role == "AI Builder":
        st.info("Validate threshold vs. precision–recall; consider recalibration if p-values drift.")
    else:
        st.info("Incident trend is within tolerance; monitor exposure and downtime KPIs.")


def render_insights(metrics: dict, role: str, thr: float, data_for_plots: pd.DataFrame):
    st.subheader("Insights")
    persona_hint(role)

    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("**Model performance**")
        perf = pd.DataFrame([
            {"Metric": "AUC", "Value": round(metrics.get("auc", np.nan), 3)},
            {"Metric": "Brier", "Value": round(metrics.get("brier", np.nan), 3)},
            {"Metric": "Last training (s)", "Value": metrics.get("last_train_s", 0.0)},
            {"Metric": "Version", "Value": metrics.get("version", "—")},
        ])
        st.dataframe(perf, hide_index=True, use_container_width=True)

        # Threshold tuner (simulated P/R tradeoff)
        st.markdown("**Threshold tuner (simulated)**")
        t = st.slider("Decision threshold", 0.05, 0.95, float(thr), 0.01, key="tuner")
        # Fake PR from beta curves
        prec = max(0.1, 1 - (t*0.7))
        rec = max(0.05, 0.95 - (t*0.9))
        f1 = 2*prec*rec/(prec+rec)
        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{prec:.2f}")
        m2.metric("Recall", f"{rec:.2f}")
        m3.metric("F1", f"{f1:.2f}")

    with c2:
        st.markdown("**Calibration & Confusion (placeholders)**")
        # Reliability diagram (fake)
        xs = np.linspace(0.01, 0.99, 10)
        ys = np.clip(xs + np.random.normal(0, 0.04, len(xs)), 0, 1)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=xs, y=xs, mode='lines', name='Perfect'))
        fig_cal.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', name='Model'))
        fig_cal.update_layout(title="Reliability", height=260, margin=dict(l=0,r=0,t=28,b=0))
        st.plotly_chart(fig_cal, use_container_width=True)

        # Confusion heatmap (fake)
        labels = ["Normal", "Anomaly"]
        mat = np.array([[70, 8],[12, 110]])
        fig_cm = px.imshow(mat, x=labels, y=labels, text_auto=True, aspect='auto', title="Confusion matrix (val)")
        fig_cm.update_layout(height=260, margin=dict(l=0,r=0,t=34,b=0))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("**Top features (global importance — placeholder)**")
    feat = pd.DataFrame({"feature": [f"f{i}" for i in range(12)], "mean_abs_shap": np.sort(np.random.rand(12))[::-1]})
    fig = px.bar(feat, x="mean_abs_shap", y="feature", orientation="h", height=320)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.button("Train / Retrain models", use_container_width=True)


def render_governance(devices: pd.DataFrame, metrics: dict, role: str):
    st.subheader("Governance")
    persona_hint(role)

    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("**Model card (preview)**")
        card = {
            "model": "LightGBM + Conformal",
            "version": metrics.get("version", "1.x"),
            "metrics": {k: v for k, v in metrics.items() if k in ["auc", "brier"]},
            "data": {"entities": int(devices["device_id"].nunique()), "synthetic": True},
            "intended_use": "Anomaly detection demo for wireless/logistics telemetry",
            "limitations": ["Synthetic data", "Not a safety control component"],
        }
        st.json(card)
        st.download_button("Download Model Card (JSON)", data=json.dumps(card, indent=2), file_name="model_card.json")

    with c2:
        st.markdown("**Transparency & audit**")
        checklist = pd.DataFrame({
            "Item": [
                "Data schema documented",
                "Training config archived",
                "Calibration evidence stored",
                "Incident audit log enabled",
                "Limitations communicated",
            ],
            "Status": ["Yes", "Yes", "Yes", "Yes", "Yes"],
        })
        st.dataframe(checklist, hide_index=True, use_container_width=True)

        audit_rows = [
            {"id": f"INC-{1000+i}", "device": f"dev_{i:03d}", "risk": round(np.random.uniform(0.7, 0.98),2), "type": np.random.choice(["Jamming","Access Breach","GPS Spoofing","Data Tamper"]) }
            for i in range(8)
        ]
        st.markdown("**Recent audit log (sample)**")
        st.dataframe(pd.DataFrame(audit_rows), hide_index=True, use_container_width=True)
        st.download_button("Download Audit Log (CSV)", data=pd.DataFrame(audit_rows).to_csv(index=False), file_name="audit_log.csv")

    st.markdown("**Risk policy (demo)**")
    st.code("""
# Decision policy (excerpt)
IF conformal_p <= 0.10 AND risk >= 0.80 THEN severity = 'Critical' AND notify SOC
ELIF conformal_p <= 0.25 AND risk >= 0.65 THEN severity = 'High' AND open ticket
ELIF risk >= 0.50 THEN severity = 'Medium' ELSE 'Low'
""", language="yaml")
    st.caption("Export full policy and evidence bundles here in production.")

# -------------------------
# App Entry
# -------------------------

def main():
    s = build_sidebar()
    header(s["role"])
    quick_help(s["help"])
    compact_banners(s["eu"])

    devices = get_device_inventory()
    full_df = get_incidents(devices, s["scenario"], s["thr"])  # includes non-incident rows too
    metrics = get_model_metrics(s["scenario"]) or {}

    # Attach risk to devices for map coloring
    devices = devices.merge(full_df[["device_id", "risk"]], on="device_id", how="left")

    kpi_row(devices, full_df, metrics)

    tabs = st.tabs(["Overview", "Incidents", "Insights", "Governance"])
    with tabs[0]:
        render_overview(devices, full_df[full_df["is_incident"]], s["show_map"], s["show_heat"], s["role"])
    with tabs[1]:
        render_incidents(full_df, s["thr"], s["role"])
    with tabs[2]:
        render_insights(metrics, s["role"], s["thr"], full_df)
    with tabs[3]:
        render_governance(devices, metrics, s["role"])

    if s["auto"]:
        time.sleep(0.8)
        st.experimental_rerun()


if __name__ == "__main__":
    main()
