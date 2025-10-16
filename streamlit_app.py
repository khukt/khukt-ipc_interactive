# streamlit_app.py
# TRUST DEMO — Map + Attack Injection + Persona-Aware Incidents
# ------------------------------------------------------------
# Requires: streamlit, pandas, numpy, plotly, pydeck
# ------------------------------------------------------------

from __future__ import annotations
import math, time, random
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import pydeck as pdk


# =========================
# Config
# =========================
@dataclass
class CFG:
    seed: int = 42
    n_devices: int = 36
    max_history: int = 1200
    map_center_lat: float = 59.3326     # Stockholm-ish
    map_center_lon: float = 18.0649
    site_radius_m: int = 400
    threshold: float = 0.7              # anomaly prob threshold
    jam_radius_m: int = 220
    breach_radius_m: int = 180
    spoof_radius_m: int = 220

random.seed(CFG.seed)
np.random.seed(CFG.seed)


# =========================
# Helper math
# =========================
def meters_to_latlon_offset(d_north_m: float, d_east_m: float, lat0: float):
    """Rough meter->deg conversion."""
    dlat = d_north_m / 111_111.0
    dlon = d_east_m / (111_111.0 * math.cos(math.radians(lat0)))
    return dlat, dlon

def random_point_in_disc(lat0: float, lon0: float, r_m: int):
    """Uniform random point within a disc of radius r_m."""
    rho = r_m * math.sqrt(random.random())
    theta = random.random() * 2*math.pi
    dN, dE = rho * math.cos(theta), rho * math.sin(theta)
    dlat, dlon = meters_to_latlon_offset(dN, dE, lat0)
    return lat0 + dlat, lon0 + dlon


# =========================
# Synthetic fleet
# =========================
TYPE_COLORS = {
    "AMR": [0,128,255,180],
    "Truck": [255,165,0,180],
    "Sensor": [34,197,94,180],
    "Gateway": [147,51,234,200],
}

def seed_devices(n=CFG.n_devices):
    """Create a mixed fleet with fixed positions around a site center."""
    devs = []
    for i in range(n):
        t = random.choices(["AMR","Truck","Sensor","Gateway"], weights=[5,3,4,2])[0]
        lat, lon = random_point_in_disc(CFG.map_center_lat, CFG.map_center_lon, CFG.site_radius_m)
        devs.append(dict(
            device_id=f"dev-{i:03d}",
            type=t, lat=lat, lon=lon,
        ))
    return pd.DataFrame(devs)

def inject_attack_risk(row, scenario, jam_mode, breach_mode, spoof_scope, radii):
    """Return anomaly probability and p-value-ish signal driven by attack shape."""
    base = {
        "AMR": np.random.uniform(0.10, 0.25),
        "Truck": np.random.uniform(0.10, 0.25),
        "Sensor": np.random.uniform(0.05, 0.20),
        "Gateway": np.random.uniform(0.05, 0.15),
    }[row["type"]]

    # distance to attack center (site center for simplicity)
    dist_m = math.hypot(
        (row.lat - CFG.map_center_lat) * 111_111.0,
        (row.lon - CFG.map_center_lon) * 111_111.0 * math.cos(math.radians(CFG.map_center_lat)),
    )

    prob, pval = base, None

    if scenario.startswith("Jamming"):
        R = radii["jam"]
        core_boost = 0.55 if jam_mode == "Broadband noise" else (0.6 if jam_mode == "Reactive" else 0.5)
        if dist_m <= R:
            prob = base + core_boost*(1 - dist_m/R) + np.random.uniform(0, 0.1)
            pval = np.clip(0.02 + 0.03*(dist_m/R) + np.random.uniform(-0.01, 0.02), 0.0005, 0.2)

    elif scenario.startswith("Access Breach"):
        R = radii["breach"]
        lure = 0.50 if breach_mode == "Evil Twin" else (0.40 if breach_mode == "Rogue Open AP" else 0.35)
        # AMR/Truck more exposed than fixed sensors
        expo = 1.0 if row["type"] in ["AMR","Truck"] else 0.7
        if dist_m <= R:
            prob = base + expo*lure*(1 - 0.6*(dist_m/R)) + np.random.uniform(0, 0.08)
            pval = np.clip(0.03 + 0.05*(dist_m/R), 0.001, 0.3)

    elif scenario.startswith("GPS Spoofing"):
        R = radii["spoof"]
        scope_mul = 1.0 if spoof_scope == "Site-wide" else (0.85 if spoof_scope == "Localized area" else 0.6)
        mob_only = st.session_state.get("spoof_mobile_only", True)
        mobile = row["type"] in ["AMR","Truck"]
        if (not mob_only) or mobile:
            if dist_m <= R:
                prob = base + scope_mul*0.55*(1 - 0.5*(dist_m/R)) + np.random.uniform(0, 0.08)
                pval = np.clip(0.02 + 0.05*(dist_m/R), 0.0005, 0.25)

    elif scenario.startswith("Data Tamper"):
        # not strongly spatial — slight bump for gateways & sensors
        mul = 0.55 if row["type"] in ["Gateway","Sensor"] else 0.4
        prob = base + mul*np.random.uniform(0.4, 0.9)
        pval = np.clip(0.05 + np.random.uniform(-0.03, 0.1), 0.005, 0.4)

    prob = float(np.clip(prob, 0, 1))
    return prob, pval


# =========================
# Simple typing head
# =========================
def classify_type(scenario, prob):
    if prob < 0.35: return "Unknown", None
    if scenario.startswith("Jamming"): return "Jamming", np.interp(prob, [0.35,1.0], [0.55,0.98])
    if scenario.startswith("Access Breach"): return "Access Breach", np.interp(prob, [0.35,1.0], [0.55,0.95])
    if scenario.startswith("GPS Spoofing"): return "GPS Spoof", np.interp(prob, [0.35,1.0], [0.55,0.96])
    if scenario.startswith("Data Tamper"): return "Data Tamper", np.interp(prob, [0.35,1.0], [0.55,0.94])
    return "Unknown", None

def severity(prob, pval):
    if prob >= 0.85 or (pval is not None and pval <= 0.05): return "High", "red"
    if prob >= 0.70 or (pval is not None and pval <= 0.20): return "Medium", "orange"
    return "Low", "green"


# =========================
# Persona helpers (Incidents)
# =========================
def _sev_color(sev: str) -> str:
    return {"High": "#e03131", "Medium": "#f08c00", "Low": "#2f9e44"}.get(sev, "#868e96")

def _sev_badge(sev: str) -> str:
    c = _sev_color(sev)
    return f"<span style='background:{c};color:white;padding:2px 10px;border-radius:999px;font-weight:700'>{sev}</span>"

def plain_english_summary(inc: dict) -> str:
    sev = inc.get("severity","—")
    scen = inc.get("scenario","—").split(" (")[0]
    tlabel = inc.get("type_label","Unknown")
    pv = inc.get("p_value")
    ptxt = " with strong evidence" if (pv is not None and pv <= 0.05) else ""
    return f"**{inc['device_id']}** likely has a **{tlabel if tlabel!='Unknown' else 'network issue'}** during **{scen}**. Risk is **{sev}**{ptxt}."

def quick_actions_for_end_user(inc: dict) -> list[str]:
    scen = inc.get("scenario","")
    if "Jamming" in scen: return ["Move 50–100 m away and retry.", "If possible, switch channel/network."]
    if "Access Breach" in scen: return ["Avoid unknown SSIDs/PLMNs.", "Use only approved networks."]
    if "GPS Spoof" in scen or "GNSS" in scen: return ["Use IMU/UWB fallback if available.", "Reduce speed until GPS stabilizes."]
    if "Tamper" in scen: return ["Resend once.", "If repeats, notify ops with incident ID."]
    return ["Retry once.", "If repeats, notify ops."]

def build_anomaly_explanation(_inc):  # compact stub
    return ("We compare each device’s recent signals to its learned baseline. "
            "Multiple metrics deviate together, which is unlikely by chance.")

def build_type_explanation(inc):
    label = inc.get("type_label","Unknown")
    msg = {
        "Jamming": "Abrupt SNR drop, noise floor rise, throughput collapse.",
        "Access Breach": "Auth anomalies and suspicious association patterns.",
        "GPS Spoof": "Inconsistent satellite geometry and sudden heading jumps.",
        "Data Tamper": "Checksum/sequence mismatches and replay-like timing.",
        "Unknown": "Doesn’t match a known threat family with high confidence.",
    }[label]
    return msg

def render_device_impacts_stub(inc: dict, topk=8):
    rng = np.random.default_rng(abs(hash(inc["id"]))%(2**32))
    feats = [f"f{i}" for i in range(1,21)]
    vals  = rng.uniform(-1.5, 1.5, size=len(feats))
    df = (pd.DataFrame({"feature":feats,"impact":vals})
          .assign(rank=lambda d: d["impact"].abs().rank(ascending=False, method="first"))
          .sort_values("rank").head(topk).drop(columns="rank"))
    fig = px.bar(df, x="impact", y="feature", orientation="h", title="Signal impacts")
    st.plotly_chart(fig, use_container_width=True)


# =========================
# App state
# =========================
def init_state():
    if "devices" not in st.session_state:
        st.session_state.devices = seed_devices()
    if "tick" not in st.session_state:
        st.session_state.tick = 0
    if "fleet_records" not in st.session_state:
        st.session_state.fleet_records = deque(maxlen=CFG.max_history)
    if "incidents" not in st.session_state:
        st.session_state.incidents = []
    if "latest_probs" not in st.session_state:
        st.session_state.latest_probs = {}

init_state()


# =========================
# Sidebar (scenario & inject)
# =========================
st.set_page_config(page_title="TRUST DEMO", layout="wide")
st.sidebar.title("TRUST DEMO — Live Site")

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Benign / Normal",
     "Jamming — RF interference",
     "Access Breach Attempt",
     "GPS Spoofing (GNSS)",
     "Data Integrity Tamper"],
    index=0
)

st.sidebar.markdown("**Attack injection (if applicable)**")
jam_mode = breach_mode = None
spoof_scope = st.sidebar.radio("Spoofing scope", ["Single device","Localized area","Site-wide"], index=1) if scenario.startswith("GPS") else None
st.sidebar.checkbox("Affect mobile (AMR/Truck) only", True, key="spoof_mobile_only") if scenario.startswith("GPS") else None

if scenario.startswith("Jamming"):
    jam_mode = st.sidebar.radio("Jamming type", ["Broadband noise","Reactive","Mgmt (deauth)"], index=0)
    CFG.jam_radius_m = st.sidebar.slider("Jam coverage (m)", 50, 500, CFG.jam_radius_m, 10)
if scenario.startswith("Access Breach"):
    breach_mode = st.sidebar.radio("Breach mode", ["Evil Twin","Rogue Open AP","Credential hammer"], index=0)
    CFG.breach_radius_m = st.sidebar.slider("Rogue node lure radius (m)", 50, 400, CFG.breach_radius_m, 10)
if scenario.startswith("GPS"):
    CFG.spoof_radius_m = st.sidebar.slider("Spoof coverage (m)", 50, 500, CFG.spoof_radius_m, 10)

st.sidebar.divider()
CFG.threshold = st.sidebar.slider("Incident threshold (model prob.)", 0.30, 0.95, CFG.threshold, 0.01)
help_mode = st.sidebar.checkbox("Help mode (inline hints)", True)

# small info banner
if help_mode:
    st.info("Use the left panel to inject attacks. The **map** shows live risk. "
            "Open **Incidents** for persona-aware explanations; **Insights** gives trends.")


# =========================
# One site tick (generate + score)
# =========================
radii = {"jam": CFG.jam_radius_m, "breach": CFG.breach_radius_m, "spoof": CFG.spoof_radius_m}

records = []
for _, row in st.session_state.devices.iterrows():
    prob, pval = inject_attack_risk(row, scenario, jam_mode, breach_mode, spoof_scope, radii)
    tlabel, tconf = classify_type(scenario, prob)
    sev, _ = severity(prob, pval)

    st.session_state.latest_probs[row.device_id] = prob
    rec = {
        "tick": datetime.utcnow(),
        "device_id": row.device_id,
        "type": row.type,
        "lat": row.lat,
        "lon": row.lon,
        "scenario": scenario,
        "prob": prob,
        "p_value": pval,
        "type_label": tlabel,
        "type_conf": tconf,
        "severity": sev,
        "risk": prob,
    }
    records.append(rec)

# append to fleet history & sample incidents
now_records = pd.DataFrame(records)
st.session_state.fleet_records.extend(now_records.to_dict(orient="records"))

# Produce incidents from devices above threshold
new_incidents = now_records[now_records["prob"] >= CFG.threshold]
for _, r in new_incidents.iterrows():
    st.session_state.incidents.append(dict(
        id=f"INC-{len(st.session_state.incidents)+1:05d}",
        tick=r["tick"], device_id=r["device_id"], scenario=r["scenario"], type_label=r["type_label"],
        prob=float(r["prob"]), p_value=(None if pd.isna(r["p_value"]) else float(r["p_value"])),
        severity=r["severity"]
    ))

st.session_state.tick += 1


# =========================
# Header KPIs
# =========================
k1,k2,k3,k4 = st.columns(4)
with k1: st.metric("Active scenario", scenario.split(" — ")[0])
with k2: st.metric("Devices", len(st.session_state.devices))
with k3:
    probs = list(st.session_state.latest_probs.values())
    st.metric("Fleet risk (mean prob)", f"{(np.mean(probs) if probs else 0):.2f}")
with k4:
    st.metric("Incidents (session)", len(st.session_state.incidents))


# =========================
# Tabs
# =========================
tab_overview, tab_incidents, tab_insights, tab_governance = st.tabs(
    ["Overview", "Incidents", "Insights", "Governance"]
)

# ---------- Overview (Map + Fleet table)
with tab_overview:
    st.header("Live Site Overview")

    # Map (pydeck)
    df_map = now_records.copy()
    df_map["fill_color"] = df_map["type"].map(TYPE_COLORS)
    df_map["radius"] = 6 + (df_map["risk"] * 16)
    df_map["label"] = df_map.apply(lambda r: f"{r.device_id} ({r.type})", axis=1)

    # Attack discs
    disc_layers = []
    if scenario.startswith("Jamming"):
        disc_layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=[{"lat":CFG.map_center_lat, "lon":CFG.map_center_lon}],
            get_position='[lon, lat]', get_radius=CFG.jam_radius_m,
            get_fill_color=[255,0,0,28], get_line_color=[255,0,0,160], get_line_width=2,
        ))
    if scenario.startswith("Access Breach"):
        disc_layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=[{"lat":CFG.map_center_lat, "lon":CFG.map_center_lon}],
            get_position='[lon, lat]', get_radius=CFG.breach_radius_m,
            get_fill_color=[255,140,0,26], get_line_color=[255,140,0,160], get_line_width=2,
        ))
    if scenario.startswith("GPS"):
        disc_layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=[{"lat":CFG.map_center_lat, "lon":CFG.map_center_lon}],
            get_position='[lon, lat]', get_radius=CFG.spoof_radius_m,
            get_fill_color=[0,128,255,22], get_line_color=[0,128,255,160], get_line_width=2,
        ))

    layers = [
        pdk.Layer("ScatterplotLayer", data=df_map,
                  get_position='[lon, lat]', get_fill_color='fill_color',
                  get_radius='radius', get_line_color=[0,0,0,100], get_line_width=1, pickable=True),
        pdk.Layer("TextLayer", data=df_map,
                  get_position='[lon, lat]', get_text='label',
                  get_color=[20,20,20,255], get_size=12, get_alignment_baseline="top", get_pixel_offset=[0,10]),
        *disc_layers
    ]

    view_state = pdk.ViewState(latitude=CFG.map_center_lat, longitude=CFG.map_center_lon, zoom=14.5, pitch=35)
    tooltip = {"text": "{label}\nprob={risk}"}
    st.pydeck_chart(
        pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None, tooltip=tooltip),
        use_container_width=True
    )

    # Fleet snapshot
    st.markdown("### Fleet snapshot")
    st.dataframe(
        now_records[["device_id","type","prob","p_value","type_label","severity"]]
        .sort_values("prob", ascending=False)
        .reset_index(drop=True),
        use_container_width=True, height=420
    )


# ---------- Incidents (Persona-aware)
def render_end_user_card(inc: dict):
    st.markdown("#### What’s happening (simple)")
    st.markdown(plain_english_summary(inc))
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(f"**Severity**<br/>{_sev_badge(inc['severity'])}", unsafe_allow_html=True)
    with c2: st.metric("Confidence (prob.)", f"{inc['prob']:.2f}")
    with c3: st.metric("Evidence (p-value)", "—" if inc.get("p_value") is None else f"{inc['p_value']:.3f}")

    st.markdown("#### What to do now")
    for step in quick_actions_for_end_user(inc):
        st.write(f"• {step}")

    with st.expander("Why this is flagged (optional)"):
        st.markdown(build_anomaly_explanation(inc))
        st.markdown(build_type_explanation(inc))

def render_executive_overview(incidents: list[dict]):
    if not incidents:
        st.info("No incidents yet.")
        return
    df = pd.DataFrame(incidents)
    st.subheader("Executive overview")
    c1,c2 = st.columns([2,1], gap="large")
    with c1:
        trend = df.groupby(pd.Grouper(key="tick", freq="2T")).size().reset_index(name="count")
        if trend.empty: trend = pd.DataFrame({"tick":[df["tick"].min()], "count":[len(df)]})
        st.plotly_chart(px.line(trend, x="tick", y="count", markers=True, title="Incident Trend"), use_container_width=True)
    with c2:
        bt = df["type_label"].value_counts().reset_index(); bt.columns=["Type","Incidents"]
        st.plotly_chart(px.bar(bt, x="Type", y="Incidents", title="By Type"), use_container_width=True)
    c3,c4 = st.columns(2, gap="large")
    with c3:
        bs = df["severity"].value_counts().reset_index(); bs.columns=["Severity","Incidents"]
        st.plotly_chart(px.bar(bs, x="Severity", y="Incidents", title="By Severity"), use_container_width=True)
    with c4:
        td = df.groupby("device_id").size().sort_values(ascending=False).head(5).reset_index(name="Incidents")
        st.plotly_chart(px.bar(td, x="device_id", y="Incidents", title="Top Devices"), use_container_width=True)
    k1,k2,k3 = st.columns(3)
    k1.metric("Incidents (session)", len(df))
    k2.metric("Unique devices", df["device_id"].nunique())
    k3.metric("Active scenario", str(df.tail(1)["scenario"].values[0]) if not df.empty else "—")

with tab_incidents:
    st.header("Incident Center")

    # Persona selector (you can wire this to auth/roles)
    role = st.selectbox("Viewer role", ["End User","Executive","Domain Expert","Regulator","AI Builder"], index=0)

    # Executive overview first
    if role == "Executive":
        render_executive_overview(st.session_state.incidents)
        st.markdown("---")

    # Filters + list
    df_all = pd.DataFrame(st.session_state.incidents)
    if df_all.empty:
        st.info("No incidents recorded yet.")
    else:
        c1,c2,c3 = st.columns(3)
        with c1:
            sev_sel = st.multiselect("Severity", ["High","Medium","Low"], default=["High","Medium","Low"])
        with c2:
            types = sorted(df_all["type_label"].unique().tolist())
            type_sel = st.multiselect("Type", types, default=types)
        with c3:
            devs = sorted(df_all["device_id"].unique().tolist())
            dev_sel = st.multiselect("Device", devs, default=devs)

        mask = df_all["severity"].isin(sev_sel) & df_all["type_label"].isin(type_sel) & df_all["device_id"].isin(dev_sel)
        df_f = df_all.loc[mask].sort_values("tick")

        st.dataframe(
            df_f.assign(when=df_f["tick"].dt.strftime("%H:%M:%S")).rename(columns={
                "id":"Incident","when":"Time (UTC)","device_id":"Device","type_label":"Type","prob":"Prob."
            })[["Incident","Time (UTC)","Device","scenario","Type","severity","Prob.","p_value"]],
            use_container_width=True, height=360
        )

        if not df_f.empty:
            sel = st.selectbox("Open incident", df_f["id"].tolist(), index=len(df_f)-1)
            inc = df_f[df_f["id"]==sel].iloc[0].to_dict()
            st.markdown("---")
            st.markdown(f"### {inc['id']} — {inc['device_id']}")
            st.caption(inc["tick"].strftime("%Y-%m-%d %H:%M:%S UTC"))

            if role == "End User":
                render_end_user_card(inc)
            elif role == "Executive":
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Severity", inc["severity"])
                c2.metric("Type", inc["type_label"])
                c3.metric("Prob.", f"{inc['prob']:.2f}")
                c4.metric("p-value", "—" if inc.get("p_value") is None else f"{inc['p_value']:.3f}")
                with st.expander("Summary"):
                    st.write(plain_english_summary(inc))
                    st.write(build_type_explanation(inc))
            elif role == "Domain Expert":
                st.markdown("#### Signals & Playbook")
                st.write("- Verify channel conditions; retry or switch band.")
                st.write("- Correlate with neighbor devices in ~100 m radius.")
                st.write("- If persistent, escalate to spectrum capture.")
                st.markdown("#### Local model factors")
                render_device_impacts_stub(inc, topk=8)
            elif role == "Regulator":
                st.markdown("#### Assurance, evidence & export")
                meta = {
                    "Incident ID": inc["id"], "Device": inc["device_id"], "Scenario": inc["scenario"],
                    "Type": inc["type_label"], "Severity": inc["severity"], "Probability": inc["prob"],
                    "p-value": inc["p_value"], "Timestamp (UTC)": inc["tick"].strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.json(meta)
                st.download_button("Export incident JSON",
                                   data=pd.Series(meta).to_json(indent=2),
                                   file_name=f"{inc['id']}.json", mime="application/json")
            else:  # AI Builder
                st.markdown("#### Model factors (local)")
                render_device_impacts_stub(inc, topk=10)
                with st.expander("Standardized feature vector (z-scores)"):
                    rng = np.random.default_rng(abs(hash(inc["id"]))%(2**32))
                    feats = [f"f{i}" for i in range(1,31)]
                    vals  = rng.normal(0,1,size=len(feats))
                    st.dataframe(pd.DataFrame({"feature":feats,"z":vals}), use_container_width=True, height=320)


# ---------- Insights (simple trends)
with tab_insights:
    st.header("Site Insights")
    fr = pd.DataFrame(list(st.session_state.fleet_records))
    if fr.empty:
        st.info("Not enough data yet.")
    else:
        # Trend of mean risk
        sub = fr.groupby(pd.Grouper(key="tick", freq="1T"))["prob"].mean().reset_index(name="mean_prob")
        st.plotly_chart(px.line(sub, x="tick", y="mean_prob", title="Mean fleet risk over time", markers=True),
                        use_container_width=True)

        # Type and severity mix from incidents
        di = pd.DataFrame(st.session_state.incidents)
        if not di.empty:
            c1,c2 = st.columns(2)
            with c1:
                bt = di["type_label"].value_counts().reset_index(); bt.columns=["Type","Incidents"]
                st.plotly_chart(px.bar(bt, x="Type", y="Incidents", title="Incident mix by type"),
                                use_container_width=True)
            with c2:
                bs = di["severity"].value_counts().reset_index(); bs.columns=["Severity","Incidents"]
                st.plotly_chart(px.bar(bs, x="Severity", y="Incidents", title="Incident mix by severity"),
                                use_container_width=True)
        else:
            st.info("No incidents yet for mix charts.")

# ---------- Governance (placeholder)
with tab_governance:
    st.header("Governance & Transparency")
    st.markdown(
        "- **Model scope**: site-level anomaly screening with simple type hints.\n"
        "- **Data**: synthetic telemetry approximating RF/GNSS/access/integrity signals.\n"
        "- **Fair use**: human-in-the-loop confirmation for any enforcement.\n"
        "- **Evidence**: each incident includes probability, p-value (if any), and a plain-English rationale."
    )
