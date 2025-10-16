# streamlit_app.py — Compact UI v2 for TRUST AI Wireless Threats
# ---------------------------------------------------------------
# Drop-in simplified layout that reduces scrolling and cognitive load.
# It is **runnable as-is** (uses lightweight synthetic data + placeholders),
# but exposes clear hooks so you can plug your existing training/SHAP logic.
#
# Usage: streamlit run streamlit_app.py
#
# Dependencies: streamlit, pandas, numpy, plotly, pydeck, shap, scikit-learn, lightgbm

from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
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

# Minimal theming helpers
PRIMARY = "#0F766E"  # teal-700
MUTED = "#6B7280"    # gray-500
ACCENT = "#2563EB"   # blue-600

# -------------------------
# Mock / lightweight data generators (safe defaults for demo)
# Replace these with your real back-end functions if available.
# -------------------------
@st.cache_data(show_spinner=False)
def get_device_inventory(n_devices: int = 30, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = rng.choice(["AMR", "Truck", "Sensor", "Gateway"], size=n_devices, p=[0.4, 0.2, 0.3, 0.1])
    base_lat, base_lon = 62.3908, 17.3069  # Sundsvall approx
    lats = base_lat + rng.normal(0, 0.02, n_devices)
    lons = base_lon + rng.normal(0, 0.04, n_devices)
    df = pd.DataFrame({
        "device_id": [f"dev_{i:03d}" for i in range(n_devices)],
        "type": types,
        "lat": lats,
        "lon": lons,
    })
    return df

@st.cache_data(show_spinner=False)
def get_incidents(devices: pd.DataFrame, scenario: str, risk_thr: float) -> pd.DataFrame:
    rng = np.random.default_rng(42 if scenario == "Normal" else 21)
    n = len(devices)
    # risk ~ baseline + scenario bump
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
    df["p_value"] = 1 - risk  # pretend conformal p
    df["is_incident"] = df["risk"] >= risk_thr
    return df[df["is_incident"]].sort_values("risk", ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def get_model_metrics(scenario: str) -> dict:
    # Placeholder metrics
    base = {
        "auc": 0.78,
        "fleet_risk": 0.33 if scenario == "Normal" else 0.55,
        "last_train_s": 4.2,
    }
    return base

# -------------------------
# UI Builders
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
    show_heat = st.sidebar.toggle("Show fleet heatmap (z-scores)", False)

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


def header(role: str):
    st.title("TRUST AI — Wireless Threat Detection Demo")
    st.caption(
        "AMR & logistics fleet • RF/network realism • LightGBM + SHAP • Conformal • Persona-aware explanations • Cached models"
    )
    # Compact role strip
    st.markdown(
        f"<div style='background:{ACCENT}10;border:1px solid {ACCENT}33;padding:8px 12px;border-radius:10px'>"
        f"<strong>Viewer role:</strong> {role} • <em>Overview, incidents, insights, governance</em>"
        "</div>",
        unsafe_allow_html=True,
    )


def quick_help(help_on: bool):
    if help_on:
        st.info("How to use: Pick a scenario → Watch KPIs → Open Incidents → See Insights → Check Governance.")


def compact_banners(eu_on: bool, scenario: str):
    if eu_on:
        st.success(
            "EU AI Act status: Limited/Minimal risk demo (synthetic telemetry; no safety control loop). "
            "If integrated as a safety component or for critical infrastructure control, it may become High‑risk.",
            icon="✅",
        )
    st.caption(
        "Loaded cached model — no retraining needed. Use the **Train / Retrain** button in Insights if you change data.")


def kpi_row(devices_df: pd.DataFrame, incidents_df: pd.DataFrame, metrics: dict):
    dcount = len(devices_df)
    icount = len(incidents_df)
    auc = metrics.get("auc", np.nan)
    frisk = metrics.get("fleet_risk", np.nan)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Devices", dcount)
    c2.metric("Incidents", icount)
    c3.metric("Model AUC", f"{auc:.2f}")
    c4.metric("Fleet Risk", f"{frisk:.2f}")


# -------------------------
# Tab renderers
# -------------------------

def tab_overview(devices: pd.DataFrame, incidents: pd.DataFrame, show_map: bool):
    st.subheader("Overview")
    if show_map:
        # Map with incidents overlay
        if not devices.empty:
            layer_all = pdk.Layer(
                "ScatterplotLayer",
                data=devices,
                get_position='[lon, lat]',
                get_radius=60,
                radius_min_pixels=2,
                radius_max_pixels=20,
                get_fill_color=[120, 120, 120, 80],
                pickable=True,
            )
            layer_inc = pdk.Layer(
                "ScatterplotLayer",
                data=incidents,
                get_position='[lon, lat]',
                get_radius=80,
                radius_min_pixels=3,
                radius_max_pixels=30,
                get_fill_color=[220, 50, 47, 160],
                pickable=True,
            )
            view_state = pdk.ViewState(latitude=devices["lat"].mean(), longitude=devices["lon"].mean(), zoom=11)
            st.pydeck_chart(pdk.Deck(layers=[layer_all, layer_inc], initial_view_state=view_state, tooltip={"text": "{device_id}\nRisk: {risk}"}))
        else:
            st.warning("No device data available.")

    # Compact KPI sparkline (dummy)
    hist = pd.DataFrame({"tick": np.arange(30), "fleet_risk": np.clip(np.cumsum(np.random.randn(30))*0.01+0.35, 0, 1)})
    fig = px.area(hist, x="tick", y="fleet_risk", height=180, title="Fleet risk (last 30 ticks)")
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)


def tab_incidents(incidents: pd.DataFrame):
    st.subheader("Incidents")
    if incidents.empty:
        st.success("No incidents above threshold. Adjust the threshold or switch scenario.")
        return
    # Compact table
    show_cols = ["device_id", "type", "risk", "p_value", "type_pred"]
    grid = incidents[show_cols].copy()
    grid.rename(columns={"type": "device_type", "type_pred": "attack_type"}, inplace=True)
    st.dataframe(grid, use_container_width=True, hide_index=True)


def tab_insights(metrics: dict):
    st.subheader("Insights")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Model performance**")
        k = pd.DataFrame([
            {"Metric": "AUC", "Value": round(metrics.get("auc", np.nan), 3)},
            {"Metric": "Brier", "Value": 0.21},
            {"Metric": "Last training (s)", "Value": metrics.get("last_train_s", 0.0)},
        ])
        st.dataframe(k, hide_index=True, use_container_width=True)
        st.button("Train / Retrain models", use_container_width=True)
    with c2:
        st.markdown("**Top features (global SHAP — placeholder)**")
        # Simple bar placeholder
        feat = pd.DataFrame({"feature": [f"f{i}" for i in range(10)], "mean_abs_shap": np.sort(np.random.rand(10))[::-1]})
        fig = px.bar(feat, x="mean_abs_shap", y="feature", orientation="h", height=320)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)


def tab_governance(devices: pd.DataFrame, metrics: dict):
    st.subheader("Governance")
    card = {
        "model": "LightGBM + Conformal",
        "metrics": metrics,
        "data": {
            "entities": int(devices["device_id"].nunique()),
            "synthetic": True,
        },
        "intended_use": "Anomaly detection demo for wireless/logistics telemetry",
        "limitations": ["Synthetic data", "Not a safety control component"],
    }
    st.download_button("Download Model Card (JSON)", data=json.dumps(card, indent=2), file_name="model_card.json")
    st.caption("Audit log and transparency artifacts can be added here.")


# -------------------------
# App Entry
# -------------------------

def main():
    s = build_sidebar()
    header(s["role"])
    quick_help(s["help"])
    compact_banners(s["eu"], s["scenario"])

    devices = get_device_inventory()
    incidents = get_incidents(devices, s["scenario"], s["thr"]) if s["thr"] is not None else pd.DataFrame()
    metrics = get_model_metrics(s["scenario"]) or {}

    kpi_row(devices, incidents, metrics)

    tabs = st.tabs(["Overview", "Incidents", "Insights", "Governance"])
    with tabs[0]:
        tab_overview(devices, incidents, s["show_map"])
    with tabs[1]:
        tab_incidents(incidents)
    with tabs[2]:
        tab_insights(metrics)
    with tabs[3]:
        tab_governance(devices, metrics)

    # Optional auto stream tick (lightweight visual effect)
    if s["auto"]:
        time.sleep(0.8)
        st.experimental_rerun()


if __name__ == "__main__":
    main()
