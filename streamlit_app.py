# streamlit_app.py
# TRUST DEMO — Map + Attack Injection + Persona-Aware Incidents + EU Governance/Audit
# -----------------------------------------------------------------------------------
# Requires: streamlit, pandas, numpy, plotly, pydeck
# -----------------------------------------------------------------------------------

from __future__ import annotations
import math, random, hashlib, io, json
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
    max_history: int = 2000
    map_center_lat: float = 59.3326     # Stockholm-ish
    map_center_lon: float = 18.0649
    site_radius_m: int = 400
    threshold: float = 0.7              # anomaly prob threshold
    jam_radius_m: int = 220
    breach_radius_m: int = 180
    spoof_radius_m: int = 220
    model_version: str = "demo-0.3.1"   # surfaced in governance/audit


random.seed(CFG.seed)
np.random.seed(CFG.seed)


# =========================
# Helper math
# =========================
def meters_to_latlon_offset(d_north_m: float, d_east_m: float, lat0: float):
    dlat = d_north_m / 111_111.0
    dlon = d_east_m / (111_111.0 * math.cos(math.radians(lat0)))
    return dlat, dlon

def random_point_in_disc(lat0: float, lon0: float, r_m: int):
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
    devs = []
    for i in range(n):
        t = random.choices(["AMR","Truck","Sensor","Gateway"], weights=[5,3,4,2])[0]
        lat, lon = random_point_in_disc(CFG.map_center_lat, CFG.map_center_lon, CFG.site_radius_m)
        devs.append(dict(device_id=f"dev-{i:03d}", type=t, lat=lat, lon=lon))
    return pd.DataFrame(devs)

def inject_attack_risk(row, scenario, jam_mode, breach_mode, spoof_scope, radii):
    base = {
        "AMR": np.random.uniform(0.10, 0.25),
        "Truck": np.random.uniform(0.10, 0.25),
        "Sensor": np.random.uniform(0.05, 0.20),
        "Gateway": np.random.uniform(0.05, 0.15),
    }[row["type"]]

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

    elif scenario.startswith("Data Integrity"):
        mul = 0.55 if row["type"] in ["Gateway","Sensor"] else 0.4
        prob = base + mul*np.random.uniform(0.4, 0.9)
        pval = np.clip(0.05 + np.random.uniform(-0.03, 0.1), 0.005, 0.4)

    prob = float(np.clip(prob, 0, 1))
    return prob, pval


# =========================
# Simple classifier head
# =========================
def classify_type(scenario, prob):
    if prob < 0.35: return "Unknown", None
    if scenario.startswith("Jamming"): return "Jamming", np.interp(prob, [0.35,1.0], [0.55,0.98])
    if scenario.startswith("Access Breach"): return "Access Breach", np.interp(prob, [0.35,1.0], [0.55,0.95])
    if scenario.startswith("GPS Spoofing"): return "GPS Spoof", np.interp(prob, [0.35,1.0], [0.55,0.96])
    if scenario.startswith("Data Integrity"): return "Data Tamper", np.interp(prob, [0.35,1.0], [0.55,0.94])
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

def build_anomaly_explanation(_inc):
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


# =========================
# Audit helpers (hashing & exports)
# =========================
def incident_json(inc: dict) -> str:
    payload = {
        "incident_id": inc["id"],
        "device_id": inc["device_id"],
        "created_utc": inc["tick"].strftime("%Y-%m-%d %H:%M:%S"),
        "scenario": inc["scenario"],
        "type_label": inc["type_label"],
        "severity": inc["severity"],
        "probability": float(inc["prob"]),
        "p_value": None if inc.get("p_value") is None else float(inc["p_value"]),
        "model_version": CFG.model_version,
        "explanation": {
            "plain": plain_english_summary(inc),
            "type_reason": build_type_explanation(inc),
        },
    }
    return json.dumps(payload, indent=2)

def incident_md_summary(inc: dict) -> str:
    j = json.loads(incident_json(inc))
    blob = json.dumps(j, sort_keys=True).encode()
    digest = hashlib.sha256(blob).hexdigest()
    lines = [
        f"# Incident {j['incident_id']}",
        "",
        f"- **Device**: {j['device_id']}",
        f"- **Created (UTC)**: {j['created_utc']}",
        f"- **Scenario**: {j['scenario']}",
        f"- **Type**: {j['type_label']}",
        f"- **Severity**: {j['severity']}",
        f"- **Probability**: {j['probability']:.2f}",
        f"- **p-value**: {j['p_value']}",
        f"- **Model version**: {j['model_version']}",
        "",
        "## Explanation",
        f"- {j['explanation']['plain']}",
        f"- Detail: {j['explanation']['type_reason']}",
        "",
        f"---",
        f"Audit signature (SHA-256 of JSON): `{digest}`",
    ]
    return "\n".join(lines).strip()


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
    if "audit_log" not in st.session_state:
        # each entry: {incident_id, decision, role, decided_at, created_at}
        st.session_state.audit_log = []

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

if help_mode:
    st.info("Use the left panel to inject attacks. The **map** shows live risk. "
            "Open **Incidents** for persona-aware explanations; **Insights** gives both trends and compliance metrics.")


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
        "model_version": CFG.model_version,
    }
    records.append(rec)

now_records = pd.DataFrame(records)
st.session_state.fleet_records.extend(now_records.to_dict(orient="records"))

# Produce incidents from devices above threshold
new_incidents = now_records[now_records["prob"] >= CFG.threshold]
for _, r in new_incidents.iterrows():
    st.session_state.incidents.append(dict(
        id=f"INC-{len(st.session_state.incidents)+1:05d}",
        tick=r["tick"], device_id=r["device_id"], scenario=r["scenario"],
        type_label=r["type_label"], prob=float(r["prob"]),
        p_value=(None if pd.isna(r["p_value"]) else float(r["p_value"])),
        severity=r["severity"], model_version=r["model_version"]
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
st.caption("🛈 All device data in this demo is synthetic and anonymized. "
           "Live deployments must ensure GDPR lawful basis, data minimization, and NIS2-grade telemetry security.")


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
        now_records[["device_id","type","prob","p_value","type_label","severity","model_version"]]
        .sort_values("prob", ascending=False)
        .reset_index(drop=True),
        use_container_width=True, height=420
    )


# ---------- Incidents (Persona-aware + Audit)
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
        st.plotly_chart(px.line(trend, x="tick", y="count", markers=True, title="Incident Trend"),
                        use_container_width=True)
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

    # Persona selector (wire to auth in production)
    role = st.selectbox("Viewer role", ["End User","Executive","Domain Expert","Regulator","AI Builder"], index=0)

    if role == "Executive":
        render_executive_overview(st.session_state.incidents)
        st.markdown("---")

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
            })[["Incident","Time (UTC)","Device","scenario","Type","severity","Prob.","p_value","model_version"]],
            use_container_width=True, height=360
        )

        if not df_f.empty:
            sel = st.selectbox("Open incident", df_f["id"].tolist(), index=len(df_f)-1)
            inc = df_f[df_f["id"]==sel].iloc[0].to_dict()
            st.markdown("---")
            st.markdown(f"### {inc['id']} — {inc['device_id']}")
            st.caption(inc["tick"].strftime("%Y-%m-%d %H:%M:%S UTC"))

            # Decision buttons → audit
            dc1, dc2, dc3 = st.columns([1,1,2])
            with dc1:
                if st.button("✅ Confirm", key=f"confirm_{inc['id']}"):
                    st.session_state.audit_log.append({
                        "incident_id": inc["id"],
                        "decision": "confirm",
                        "role": role,
                        "decided_at": datetime.utcnow(),
                        "created_at": inc["tick"],
                    })
                    st.success("Decision logged: confirm")
            with dc2:
                if st.button("✖️ Dismiss", key=f"dismiss_{inc['id']}"):
                    st.session_state.audit_log.append({
                        "incident_id": inc["id"],
                        "decision": "dismiss",
                        "role": role,
                        "decided_at": datetime.utcnow(),
                        "created_at": inc["tick"],
                    })
                    st.warning("Decision logged: dismiss")
            with dc3:
                st.caption("All decisions are recorded for audit and compliance metrics.")

            if role == "End User":
                st.markdown("#### What’s happening (simple)")
                st.markdown(plain_english_summary(inc))
                c1,c2,c3 = st.columns(3)
                with c1: st.markdown(f"**Severity**<br/>{_sev_badge(inc['severity'])}", unsafe_allow_html=True)
                with c2: st.metric("Confidence (prob.)", f"{inc['prob']:.2f}")
                with c3: st.metric("Evidence (p-value)", "—" if inc.get("p_value") is None else f"{inc['p_value']:.3f}")
                st.markdown("#### What to do now")
                for step in quick_actions_for_end_user(inc): st.write(f"• {step}")
                with st.expander("Why this is flagged (optional)"):
                    st.markdown(build_anomaly_explanation(inc))
                    st.markdown(build_type_explanation(inc))

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
                # Lightweight "impacts" placeholder
                rng = np.random.default_rng(abs(hash(inc["id"]))%(2**32))
                feats = [f"f{i}" for i in range(1,21)]
                vals  = rng.uniform(-1.5, 1.5, size=len(feats))
                df_imp = (pd.DataFrame({"feature":feats,"impact":vals})
                          .assign(rank=lambda d: d["impact"].abs().rank(ascending=False, method="first"))
                          .sort_values("rank").head(8).drop(columns="rank"))
                st.plotly_chart(px.bar(df_imp, x="impact", y="feature", orientation="h", title="Signal impacts"),
                                use_container_width=True)

            elif role == "Regulator":
                st.markdown("#### Assurance, evidence & export")
                meta = {
                    "Incident ID": inc["id"], "Device": inc["device_id"], "Scenario": inc["scenario"],
                    "Type": inc["type_label"], "Severity": inc["severity"], "Probability": inc["prob"],
                    "p-value": inc["p_value"], "Model version": inc["model_version"],
                    "Timestamp (UTC)": inc["tick"].strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.json(meta)

                j = incident_json(inc)
                md = incident_md_summary(inc)
                digest = hashlib.sha256(json.dumps(json.loads(j), sort_keys=True).encode()).hexdigest()

                cex1, cex2 = st.columns(2)
                with cex1:
                    st.download_button("Export incident JSON", data=j,
                                       file_name=f"{inc['id']}.json", mime="application/json")
                with cex2:
                    st.download_button("Export incident summary (Markdown)", data=md,
                                       file_name=f"{inc['id']}.md", mime="text/markdown")
                st.info(f"Audit signature (SHA-256 of JSON): {digest}")

                st.caption("Exports include the model version, probability, p-value (if available), "
                           "and a human-readable rationale for EU AI Act transparency and auditability.")

            else:  # AI Builder
                st.markdown("#### Model factors (local)")
                rng = np.random.default_rng(abs(hash(inc["id"]))%(2**32))
                feats = [f"f{i}" for i in range(1,31)]
                vals  = rng.normal(0,1,size=len(feats))
                st.dataframe(pd.DataFrame({"feature":feats,"z":vals}), use_container_width=True, height=320)
                st.caption("Developer view: standardized feature vector (z-scores).")


# ---------- Insights (trends + compliance metrics)
with tab_insights:
    st.header("Site Insights & Compliance Metrics")

    fr = pd.DataFrame(list(st.session_state.fleet_records))
    di = pd.DataFrame(st.session_state.incidents)
    al = pd.DataFrame(st.session_state.audit_log)

    # Risk trend
    if not fr.empty:
        sub = fr.groupby(pd.Grouper(key="tick", freq="1T"))["prob"].mean().reset_index(name="mean_prob")
        st.plotly_chart(px.line(sub, x="tick", y="mean_prob", title="Mean fleet risk over time", markers=True),
                        use_container_width=True)
    else:
        st.info("Not enough data yet for risk trend.")

    # Incident mix
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

    st.markdown("---")
    st.subheader("Compliance KPIs (EU AI Act — transparency & oversight)")

    # Compute decision metrics
    if not al.empty:
        # Join created_at from incidents for completeness if missing
        # (we already store created_at in audit entries)
        al["latency_sec"] = (al["decided_at"] - al["created_at"]).dt.total_seconds()
        decisions = len(al)
        confirms = (al["decision"] == "confirm").sum()
        dismiss = (al["decision"] == "dismiss").sum()
        mean_latency = float(np.nanmean(al["latency_sec"])) if not al["latency_sec"].empty else float("nan")

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Decisions logged", decisions)
        k2.metric("Confirmations", confirms)
        k3.metric("Dismissals", dismiss)
        k4.metric("Avg time to decision (s)", f"{mean_latency:.1f}" if not math.isnan(mean_latency) else "—")

        with st.expander("Audit log (latest 50)"):
            view = al.sort_values("decided_at", ascending=False).head(50).copy()
            view["decided_at"] = view["decided_at"].dt.strftime("%H:%M:%S")
            view["created_at"] = view["created_at"].dt.strftime("%H:%M:%S")
            st.dataframe(view[["incident_id","decision","role","created_at","decided_at"]],
                         use_container_width=True, height=260)
    else:
        st.info("No human decisions yet. Confirm or dismiss an incident to populate compliance KPIs.")


# ---------- Governance (EU AI Act alignment)
with tab_governance:
    st.header("Governance, Transparency & Compliance")

    st.subheader("AI System Classification (EU AI Act)")
    st.markdown(
        "This demonstration platform is categorized as a **limited-risk AI system** "
        "under the EU AI Act, because it performs real-time anomaly detection "
        "and advisory analytics but **does not make autonomous enforcement decisions**. "
        "All high-impact outcomes must be validated by a human operator."
    )

    st.subheader("Key Compliance Principles")
    st.markdown("""
    | Principle | Implementation in TRUST DEMO |
    |------------|------------------------------|
    | **Human oversight** | Every alert is advisory; operators confirm or dismiss incidents manually (audit logged). |
    | **Transparency** | Each detection includes plain-English rationale and exportable evidence (JSON/MD). |
    | **Accountability** | Model version and parameters are surfaced and included in every export. |
    | **Data governance** | Demo uses synthetic telemetry; production must ensure data minimization & purpose limitation. |
    | **Robustness & accuracy** | Thresholds are tunable; trends monitored for drift via Insights. |
    | **Traceability** | Decisions (confirm/dismiss) are time-stamped and retained in an audit log. |
    | **Security (NIS2)** | Integrity of telemetry and secure transport are assumed in production design. |
    | **Lifecycle mgmt.** | Monitoring for false positives, retraining plan, and versioning are expected in ops. |
    """)

    st.subheader("GDPR & Privacy")
    st.markdown(
        "This demo processes **technical telemetry only** (e.g., signal/SNR/throughput proxies). "
        "No personal data is collected. For live deployments, the data controller must ensure:  \n"
        "- **Lawful basis** for data processing (GDPR Art. 6).  \n"
        "- **Data minimization** and **purpose limitation**.  \n"
        "- User **rights** (access, rectification, deletion) and **privacy by design** (Art. 25)."
    )

    st.subheader("Explainability & Contestability")
    st.markdown(
        "Every incident shows either a plain-language reason (End User) or signal contributions (Expert). "
        "Operators may annotate, confirm, or dismiss incidents, establishing a contestable workflow."
    )

    st.subheader("Export & Audit Trail")
    st.markdown(
        "Each incident export contains: ID, device, timestamp, model version, probability, p-value, and rationale. "
        "A SHA-256 digest (displayed in Regulator view) enables integrity checks for audits."
    )

    st.subheader("Next Compliance Steps (for production use)")
    st.markdown("""
    - Conduct a formal **AI Risk Assessment** per EU AI Act requirements.  
    - Establish a **Data Protection Impact Assessment (DPIA)** under GDPR Art. 35.  
    - Implement **continuous logging** for traceability and cybersecurity monitoring (NIS2).  
    - Maintain a **Model Card** and **Datasheet for Datasets** as transparency artefacts.  
    - Register in the **EU AI Transparency Database** if required prior to market placement.  
    """)

    with st.expander("Model Card (summary)"):
        st.markdown(f"""
- **Model version**: `{CFG.model_version}`  
- **Intended use**: site-level anomaly screening for RF/access/GNSS/integrity issues.  
- **Not for**: autonomous enforcement or user surveillance.  
- **Main risks**: false positives during RF congestion; misclassification under novel attacks.  
- **Mitigations**: human oversight, threshold tuning, continuous monitoring, periodic retraining.
        """)
    st.success("✅ Current demo satisfies EU AI Act transparency & human-oversight expectations for limited-risk AI.")
