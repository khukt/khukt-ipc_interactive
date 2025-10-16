# streamlit_app.py
# TRUST DEMO — Map + Attack Injection + Persona-Aware Incidents + Governance (Charts)
# Requires: streamlit, pandas, numpy, plotly, pydeck

from __future__ import annotations
import math, random, hashlib, json
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pydeck as pdk

# ---------------- Config ----------------
@dataclass
class CFG:
    seed: int = 42
    n_devices: int = 36
    max_history: int = 2000
    map_center_lat: float = 59.3326
    map_center_lon: float = 18.0649
    site_radius_m: int = 400
    threshold: float = 0.7
    jam_radius_m: int = 220
    breach_radius_m: int = 180
    spoof_radius_m: int = 220
    model_version: str = "demo-0.3.3"

random.seed(CFG.seed)
np.random.seed(CFG.seed)

# ---------------- Utils ----------------
def meters_to_latlon_offset(d_north_m: float, d_east_m: float, lat0: float):
    dlat = d_north_m / 111_111.0
    dlon = d_east_m / (111_111.0 * math.cos(math.radians(lat0)))
    return dlat, dlon

def random_point_in_disc(lat0: float, lon0: float, r_m: int):
    rho = r_m * math.sqrt(random.random()); theta = random.random() * 2*math.pi
    dN, dE = rho*math.cos(theta), rho*math.sin(theta)
    dlat, dlon = meters_to_latlon_offset(dN, dE, lat0)
    return lat0 + dlat, lon0 + dlon

def safe_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def _download_bytes(text: str, filename: str, label: str, mime="text/plain"):
    st.download_button(label, data=text.encode("utf-8"), file_name=filename, mime=mime)

# ---------------- Synthetic fleet ----------------
TYPE_COLORS = {
    "AMR": [0,128,255,180], "Truck": [255,165,0,180],
    "Sensor": [34,197,94,180], "Gateway": [147,51,234,200],
}

def seed_devices(n=CFG.n_devices):
    devs = []
    for i in range(n):
        t = random.choices(["AMR","Truck","Sensor","Gateway"], weights=[5,3,4,2])[0]
        lat, lon = random_point_in_disc(CFG.map_center_lat, CFG.map_center_lon, CFG.site_radius_m)
        devs.append(dict(device_id=f"dev-{i:03d}", type=t, lat=lat, lon=lon))
    return pd.DataFrame(devs)

def inject_attack_risk(row, scenario, jam_mode, breach_mode, spoof_scope, radii):
    base = {"AMR": np.random.uniform(0.10,0.25), "Truck": np.random.uniform(0.10,0.25),
            "Sensor": np.random.uniform(0.05,0.20), "Gateway": np.random.uniform(0.05,0.15)}[row["type"]]
    dist_m = math.hypot(
        (row.lat - CFG.map_center_lat)*111_111.0,
        (row.lon - CFG.map_center_lon)*111_111.0*math.cos(math.radians(CFG.map_center_lat)),
    )
    prob, pval = base, None
    if scenario.startswith("Jamming"):
        R = radii["jam"]; core_boost = 0.55 if jam_mode=="Broadband noise" else (0.6 if jam_mode=="Reactive" else 0.5)
        if dist_m <= R: prob = base + core_boost*(1 - dist_m/R) + np.random.uniform(0,0.1); pval = np.clip(0.02+0.03*(dist_m/R)+np.random.uniform(-0.01,0.02),0.0005,0.2)
    elif scenario.startswith("Access Breach"):
        R = radii["breach"]; lure = 0.50 if breach_mode=="Evil Twin" else (0.40 if breach_mode=="Rogue Open AP" else 0.35)
        expo = 1.0 if row["type"] in ["AMR","Truck"] else 0.7
        if dist_m <= R: prob = base + expo*lure*(1 - 0.6*(dist_m/R)) + np.random.uniform(0,0.08); pval = np.clip(0.03 + 0.05*(dist_m/R), 0.001, 0.3)
    elif scenario.startswith("GPS Spoofing"):
        R = radii["spoof"]; scope_mul = 1.0 if spoof_scope=="Site-wide" else (0.85 if spoof_scope=="Localized area" else 0.6)
        mob_only = st.session_state.get("spoof_mobile_only", True); mobile = row["type"] in ["AMR","Truck"]
        if (not mob_only) or mobile:
            if dist_m <= R: prob = base + scope_mul*0.55*(1 - 0.5*(dist_m/R)) + np.random.uniform(0,0.08); pval = np.clip(0.02 + 0.05*(dist_m/R), 0.0005, 0.25)
    elif scenario.startswith("Data Integrity"):
        mul = 0.55 if row["type"] in ["Gateway","Sensor"] else 0.4
        prob = base + mul*np.random.uniform(0.4,0.9); pval = np.clip(0.05 + np.random.uniform(-0.03,0.1), 0.005, 0.4)
    return float(np.clip(prob,0,1)), pval

# ---------------- Classifier head ----------------
def classify_type(scenario, prob):
    if prob < 0.35: return "Unknown", None
    if scenario.startswith("Jamming"): return "Jamming", np.interp(prob,[0.35,1.0],[0.55,0.98])
    if scenario.startswith("Access Breach"): return "Access Breach", np.interp(prob,[0.35,1.0],[0.55,0.95])
    if scenario.startswith("GPS Spoofing"): return "GPS Spoof", np.interp(prob,[0.35,1.0],[0.55,0.96])
    if scenario.startswith("Data Integrity"): return "Data Tamper", np.interp(prob,[0.35,1.0],[0.55,0.94])
    return "Unknown", None

def severity(prob, pval):
    if prob >= 0.85 or (pval is not None and pval <= 0.05): return "High","red"
    if prob >= 0.70 or (pval is not None and pval <= 0.20): return "Medium","orange"
    return "Low","green"

# ---------------- Persona helpers ----------------
def _sev_color(sev: str) -> str:
    return {"High":"#e03131","Medium":"#f08c00","Low":"#2f9e44"}.get(sev,"#868e96")

def _sev_badge(sev: str) -> str:
    c=_sev_color(sev); return f"<span style='background:{c};color:white;padding:2px 10px;border-radius:999px;font-weight:700'>{sev}</span>"

def plain_english_summary(inc: dict) -> str:
    sev=inc.get("severity","—"); scen=inc.get("scenario","—").split(" (")[0]; t=inc.get("type_label","Unknown"); pv=inc.get("p_value")
    ptxt=" with strong evidence" if (pv is not None and pv<=0.05) else ""
    return f"**{inc['device_id']}** likely has a **{t if t!='Unknown' else 'network issue'}** during **{scen}**. Risk is **{sev}**{ptxt}."

def quick_actions_for_end_user(inc: dict) -> list[str]:
    s=inc.get("scenario","")
    if "Jamming" in s: return ["Move 50–100 m away and retry.","If possible, switch channel/network."]
    if "Access Breach" in s: return ["Avoid unknown SSIDs/PLMNs.","Use only approved networks."]
    if "GPS Spoof" in s or "GNSS" in s: return ["Use IMU/UWB fallback if available.","Reduce speed until GPS stabilizes."]
    if "Tamper" in s: return ["Resend once.","If repeats, notify ops with incident ID."]
    return ["Retry once.","If repeats, notify ops."]

def build_anomaly_explanation(_): return ("We compare each device’s recent signals to its learned baseline. Multiple metrics deviate together, which is unlikely by chance.")
def build_type_explanation(inc):
    label=inc.get("type_label","Unknown")
    return {
        "Jamming":"Abrupt SNR drop, noise floor rise, throughput collapse.",
        "Access Breach":"Auth anomalies and suspicious association patterns.",
        "GPS Spoof":"Inconsistent satellite geometry and sudden heading jumps.",
        "Data Tamper":"Checksum/sequence mismatches and replay-like timing.",
        "Unknown":"Doesn’t match a known threat family with high confidence.",
    }[label]

# ---------------- Audit helpers ----------------
def incident_json(inc: dict) -> str:
    payload = {
        "incident_id": inc["id"], "device_id": inc["device_id"],
        "created_utc": inc["tick"].strftime("%Y-%m-%d %H:%M:%S"),
        "scenario": inc["scenario"], "type_label": inc["type_label"],
        "severity": inc["severity"], "probability": float(inc["prob"]),
        "p_value": None if inc.get("p_value") is None else float(inc["p_value"]),
        "model_version": inc.get("model_version", CFG.model_version),
        "explanation": {"plain": plain_english_summary(inc), "type_reason": build_type_explanation(inc)},
    }
    return json.dumps(payload, indent=2)

def incident_md_summary(inc: dict) -> str:
    j=json.loads(incident_json(inc)); blob=json.dumps(j,sort_keys=True).encode(); digest=hashlib.sha256(blob).hexdigest()
    return (
        f"# Incident {j['incident_id']}\n\n"
        f"- **Device**: {j['device_id']}\n- **Created (UTC)**: {j['created_utc']}\n- **Scenario**: {j['scenario']}\n"
        f"- **Type**: {j['type_label']}\n- **Severity**: {j['severity']}\n- **Probability**: {j['probability']:.2f}\n"
        f"- **p-value**: {j['p_value']}\n- **Model version**: {j['model_version']}\n\n"
        f"## Explanation\n- {j['explanation']['plain']}\n- Detail: {j['explanation']['type_reason']}\n\n"
        f"---\nAudit signature (SHA-256 of JSON): `{digest}`"
    )

# ---------------- App state ----------------
def init_state():
    ss = st.session_state
    if "devices" not in ss: ss.devices = seed_devices()
    if "tick" not in ss: ss.tick = 0
    if "fleet_records" not in ss: ss.fleet_records = deque(maxlen=CFG.max_history)
    if "incidents" not in ss: ss.incidents = []
    if "latest_probs" not in ss: ss.latest_probs = {}
    if "audit_log" not in ss: ss.audit_log = []

def _init_gov_state():
    ss = st.session_state
    ss.setdefault("gov_controls", {
        "human_oversight": True, "transparency_cards": True, "traceability_audit": True,
        "data_minimization": True, "privacy_by_design": True, "security_transport": True,
        "robustness_eval": True, "lifecycle_mgmt": True,
    })
    ss.setdefault("gov_data_inventory", pd.DataFrame([
        {"Dataset": "Telemetry (synthetic)", "Purpose": "Anomaly screening", "PII": "No", "Retention_days": 3},
        {"Dataset": "Incident metadata", "Purpose": "Audit & oversight", "PII": "No", "Retention_days": 365},
    ]))
    ss.setdefault("gov_risk_register", pd.DataFrame([
        {"Risk": "RF congestion -> false positives", "Severity": "Medium", "Likelihood": "Medium", "Mitigation": "Human review, threshold tuning", "Owner": "Ops Lead", "Status": "Planned"},
        {"Risk": "Novel attack not in taxonomy", "Severity": "High", "Likelihood": "Low", "Mitigation": "Drift watch, retraining plan", "Owner": "ML Lead", "Status": "Ongoing"},
    ]))
    ss.setdefault("gov_retention_days_incidents", 365)
    ss.setdefault("gov_log_level", "Info")
    ss.setdefault("gov_hash_algo", "SHA-256")
    ss.setdefault("gov_pseudonymization", True)
    ss.setdefault("gov_role_matrix", pd.DataFrame([
        {"Role": "End User", "Read Incidents": True, "Export": False, "Decide": False, "Admin": False},
        {"Role": "Executive", "Read Incidents": True, "Export": True, "Decide": True, "Admin": False},
        {"Role": "Domain Expert", "Read Incidents": True, "Export": True, "Decide": True, "Admin": False},
        {"Role": "Regulator", "Read Incidents": True, "Export": True, "Decide": False, "Admin": False},
        {"Role": "AI Builder", "Read Incidents": True, "Export": False, "Decide": False, "Admin": False},
        {"Role": "Admin", "Read Incidents": True, "Export": True, "Decide": True, "Admin": True},
    ]))

# ---------------- Sidebar ----------------
st.set_page_config(page_title="TRUST DEMO", layout="wide")
init_state()

st.sidebar.title("TRUST DEMO — Live Site")
scenario = st.sidebar.selectbox(
    "Scenario",
    ["Benign / Normal","Jamming — RF interference","Access Breach Attempt","GPS Spoofing (GNSS)","Data Integrity Tamper"],
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
    st.info("Use the left panel to inject attacks. The **map** shows live risk. Incidents are persona-aware; Insights add trends & compliance metrics.")

# ---------------- Generate one tick ----------------
radii = {"jam": CFG.jam_radius_m, "breach": CFG.breach_radius_m, "spoof": CFG.spoof_radius_m}
records = []
for _, row in st.session_state.devices.iterrows():
    prob, pval = inject_attack_risk(row, scenario, jam_mode, breach_mode, spoof_scope, radii)
    tlabel, tconf = classify_type(scenario, prob)
    sev, _ = severity(prob, pval)
    st.session_state.latest_probs[row.device_id] = prob
    records.append({
        "tick": datetime.utcnow(), "device_id": row.device_id, "type": row.type,
        "lat": row.lat, "lon": row.lon, "scenario": scenario, "prob": prob, "p_value": pval,
        "type_label": tlabel, "type_conf": tconf, "severity": sev, "risk": prob, "model_version": CFG.model_version
    })
now_records = pd.DataFrame(records)
st.session_state.fleet_records.extend(now_records.to_dict(orient="records"))
new_incidents = now_records[now_records["prob"] >= CFG.threshold]
for _, r in new_incidents.iterrows():
    st.session_state.incidents.append(dict(
        id=f"INC-{len(st.session_state.incidents)+1:05d}", tick=r["tick"], device_id=r["device_id"],
        scenario=r["scenario"], type_label=r["type_label"], prob=float(r["prob"]),
        p_value=(None if pd.isna(r["p_value"]) else float(r["p_value"])),
        severity=r["severity"], model_version=r.get("model_version", CFG.model_version)
    ))
st.session_state.tick += 1

# ---------------- KPIs ----------------
k1,k2,k3,k4 = st.columns(4)
with k1: st.metric("Active scenario", scenario.split(" — ")[0])
with k2: st.metric("Devices", len(st.session_state.devices))
with k3: st.metric("Fleet risk (mean prob)", f"{(np.mean(list(st.session_state.latest_probs.values())) if st.session_state.latest_probs else 0):.2f}")
with k4: st.metric("Incidents (session)", len(st.session_state.incidents))
st.caption("🛈 Demo data is synthetic. Live deployments must ensure GDPR lawful basis, minimization, and NIS2-grade security.")

# ---------------- TABS (DEFINE FIRST!) ----------------
tab_overview, tab_incidents, tab_insights, tab_governance = st.tabs(
    ["Overview", "Incidents", "Insights", "Governance"]
)

# ---------------- Overview ----------------
with tab_overview:
    st.header("Live Site Overview")
    df_map = now_records.copy()
    df_map["fill_color"] = df_map["type"].map(TYPE_COLORS)
    df_map["radius"] = 6 + (df_map["risk"] * 16)
    df_map["label"] = df_map.apply(lambda r: f"{r.device_id} ({r.type})", axis=1)
    disc_layers=[]
    if scenario.startswith("Jamming"):
        disc_layers.append(pdk.Layer("ScatterplotLayer", data=[{"lat":CFG.map_center_lat,"lon":CFG.map_center_lon}],
                                     get_position='[lon, lat]', get_radius=CFG.jam_radius_m,
                                     get_fill_color=[255,0,0,28], get_line_color=[255,0,0,160], get_line_width=2))
    if scenario.startswith("Access Breach"):
        disc_layers.append(pdk.Layer("ScatterplotLayer", data=[{"lat":CFG.map_center_lat,"lon":CFG.map_center_lon}],
                                     get_position='[lon, lat]', get_radius=CFG.breach_radius_m,
                                     get_fill_color=[255,140,0,26], get_line_color=[255,140,0,160], get_line_width=2))
    if scenario.startswith("GPS"):
        disc_layers.append(pdk.Layer("ScatterplotLayer", data=[{"lat":CFG.map_center_lat,"lon":CFG.map_center_lon}],
                                     get_position='[lon, lat]', get_radius=CFG.spoof_radius_m,
                                     get_fill_color=[0,128,255,22], get_line_color=[0,128,255,160], get_line_width=2))
    layers = [
        pdk.Layer("ScatterplotLayer", data=df_map, get_position='[lon, lat]', get_fill_color='fill_color',
                  get_radius='radius', get_line_color=[0,0,0,100], get_line_width=1, pickable=True),
        pdk.Layer("TextLayer", data=df_map, get_position='[lon, lat]', get_text='label',
                  get_color=[20,20,20,255], get_size=12, get_alignment_baseline="top", get_pixel_offset=[0,10]),
        *disc_layers
    ]
    view_state = pdk.ViewState(latitude=CFG.map_center_lat, longitude=CFG.map_center_lon, zoom=14.5, pitch=35)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None,
                             tooltip={"text":"{label}\nprob={risk}"}), use_container_width=True)

    st.markdown("### Fleet snapshot")
    cols = ["device_id","type","prob","p_value","type_label","severity","model_version"]
    st.dataframe(now_records[safe_columns(now_records, cols)]
                 .sort_values(by=[c for c in ["prob"] if c in now_records.columns], ascending=False)
                 .reset_index(drop=True), use_container_width=True, height=420)

# ---------------- Incidents ----------------
def render_executive_overview(incidents):
    if not incidents: st.info("No incidents yet."); return
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
    k1.metric("Incidents (session)", len(df)); k2.metric("Unique devices", df["device_id"].nunique())
    k3.metric("Active scenario", str(df.tail(1)["scenario"].values[0]) if not df.empty else "—")

with tab_incidents:
    st.header("Incident Center")
    role = st.selectbox("Viewer role", ["End User","Executive","Domain Expert","Regulator","AI Builder"], index=0)
    if role == "Executive": render_executive_overview(st.session_state.incidents); st.markdown("---")

    df_all = pd.DataFrame(st.session_state.incidents)
    if df_all.empty:
        st.info("No incidents recorded yet.")
    else:
        c1,c2,c3 = st.columns(3)
        with c1: sev_sel = st.multiselect("Severity", ["High","Medium","Low"], default=["High","Medium","Low"])
        with c2: types = sorted(df_all["type_label"].unique().tolist()); type_sel = st.multiselect("Type", types, default=types)
        with c3: devs = sorted(df_all["device_id"].unique().tolist()); dev_sel = st.multiselect("Device", devs, default=devs)
        mask = df_all["severity"].isin(sev_sel) & df_all["type_label"].isin(type_sel) & df_all["device_id"].isin(dev_sel)
        df_f = df_all.loc[mask].sort_values("tick")
        base_cols = ["id","tick","device_id","scenario","type_label","severity","prob","p_value","model_version"]
        show_cols = safe_columns(df_f, base_cols)
        rename_map = {"id":"Incident","tick":"Time (UTC)","device_id":"Device","type_label":"Type","prob":"Prob."}
        df_show = df_f[show_cols].rename(columns=rename_map)
        if "Time (UTC)" in df_show.columns: df_show["Time (UTC)"] = pd.to_datetime(df_show["Time (UTC)"]).dt.strftime("%H:%M:%S")
        st.dataframe(df_show, use_container_width=True, height=360)

        if not df_f.empty:
            sel = st.selectbox("Open incident", df_f["id"].tolist(), index=len(df_f)-1)
            inc = df_f[df_f["id"]==sel].iloc[0].to_dict()
            st.markdown("---"); st.markdown(f"### {inc['id']} — {inc['device_id']}"); st.caption(inc["tick"].strftime("%Y-%m-%d %H:%M:%S UTC"))
            dc1, dc2, dc3 = st.columns([1,1,2])
            with dc1:
                if st.button("✅ Confirm", key=f"confirm_{inc['id']}"):
                    st.session_state.audit_log.append({"incident_id": inc["id"], "decision": "confirm", "role": role, "decided_at": datetime.utcnow(), "created_at": inc["tick"]})
                    st.success("Decision logged: confirm")
            with dc2:
                if st.button("✖️ Dismiss", key=f"dismiss_{inc['id']}"):
                    st.session_state.audit_log.append({"incident_id": inc["id"], "decision": "dismiss", "role": role, "decided_at": datetime.utcnow(), "created_at": inc["tick"]})
                    st.warning("Decision logged: dismiss")
            with dc3: st.caption("All decisions are recorded for audit and compliance metrics.")

            if role == "End User":
                st.markdown("#### What’s happening (simple)"); st.markdown(plain_english_summary(inc))
                c1,c2,c3 = st.columns(3)
                with c1: st.markdown(f"**Severity**<br/>{_sev_badge(inc['severity'])}", unsafe_allow_html=True)
                with c2: st.metric("Confidence (prob.)", f"{inc['prob']:.2f}")
                with c3: st.metric("Evidence (p-value)", "—" if inc.get("p_value") is None else f"{inc['p_value']:.3f}")
                st.markdown("#### What to do now"); [st.write(f"• {s}") for s in quick_actions_for_end_user(inc)]
                with st.expander("Why this is flagged (optional)"): st.markdown(build_anomaly_explanation(inc)); st.markdown(build_type_explanation(inc))

            elif role == "Executive":
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Severity", inc["severity"]); c2.metric("Type", inc["type_label"])
                c3.metric("Prob.", f"{inc['prob']:.2f}"); c4.metric("p-value","—" if inc.get("p_value") is None else f"{inc['p_value']:.3f}")
                with st.expander("Summary"): st.write(plain_english_summary(inc)); st.write(build_type_explanation(inc))

            elif role == "Domain Expert":
                st.markdown("#### Signals & Playbook")
                st.write("- Verify channel conditions; retry or switch band."); st.write("- Correlate with neighbor devices in ~100 m radius."); st.write("- If persistent, escalate to spectrum capture.")
                st.markdown("#### Local model factors")
                rng = np.random.default_rng(abs(hash(inc["id"]))%(2**32))
                feats=[f"f{i}" for i in range(1,21)]; vals=rng.uniform(-1.5,1.5,size=len(feats))
                df_imp=(pd.DataFrame({"feature":feats,"impact":vals}).assign(rank=lambda d: d["impact"].abs().rank(ascending=False,method="first")).sort_values("rank").head(8).drop(columns="rank"))
                st.plotly_chart(px.bar(df_imp, x="impact", y="feature", orientation="h", title="Signal impacts"), use_container_width=True)

            elif role == "Regulator":
                st.markdown("#### Assurance, evidence & export")
                meta = {"Incident ID":inc["id"],"Device":inc["device_id"],"Scenario":inc["scenario"],"Type":inc["type_label"],"Severity":inc["severity"],"Probability":inc["prob"],"p-value":inc["p_value"],"Model version":inc.get("model_version",CFG.model_version),"Timestamp (UTC)":inc["tick"].strftime("%Y-%m-%d %H:%M:%S")}
                st.json(meta)
                j=incident_json(inc); md=incident_md_summary(inc); digest=hashlib.sha256(json.dumps(json.loads(j),sort_keys=True).encode()).hexdigest()
                cex1, cex2 = st.columns(2)
                with cex1: st.download_button("Export incident JSON", data=j, file_name=f"{inc['id']}.json", mime="application/json")
                with cex2: st.download_button("Export incident summary (Markdown)", data=md, file_name=f"{inc['id']}.md", mime="text/markdown")
                st.info(f"Audit signature (SHA-256 of JSON): {digest}")

            else:  # AI Builder
                st.markdown("#### Model factors (local)")
                rng = np.random.default_rng(abs(hash(inc["id"]))%(2**32))
                feats=[f"f{i}" for i in range(1,31)]; vals=rng.normal(0,1,size=len(feats))
                st.dataframe(pd.DataFrame({"feature":feats,"z":vals}), use_container_width=True, height=320)
                st.caption("Developer view: standardized feature vector (z-scores).")

# ---------------- Insights ----------------
with tab_insights:
    st.header("Site Insights & Compliance Metrics")
    fr = pd.DataFrame(list(st.session_state.fleet_records)); di = pd.DataFrame(st.session_state.incidents); al = pd.DataFrame(st.session_state.audit_log)
    if not fr.empty:
        sub = fr.groupby(pd.Grouper(key="tick", freq="1T"))["prob"].mean().reset_index(name="mean_prob")
        st.plotly_chart(px.line(sub, x="tick", y="mean_prob", title="Mean fleet risk over time", markers=True), use_container_width=True)
    else: st.info("Not enough data yet for risk trend.")
    if not di.empty:
        c1,c2 = st.columns(2)
        with c1: bt=di["type_label"].value_counts().reset_index(); bt.columns=["Type","Incidents"]; st.plotly_chart(px.bar(bt, x="Type", y="Incidents", title="Incident mix by type"), use_container_width=True)
        with c2: bs=di["severity"].value_counts().reset_index(); bs.columns=["Severity","Incidents"]; st.plotly_chart(px.bar(bs, x="Severity", y="Incidents", title="Incident mix by severity"), use_container_width=True)
    else: st.info("No incidents yet for mix charts.")
    st.markdown("---"); st.subheader("Compliance KPIs (EU AI Act — transparency & oversight)")
    if not al.empty:
        al["decided_at"]=pd.to_datetime(al["decided_at"]); al["created_at"]=pd.to_datetime(al["created_at"]); al["latency_sec"]=(al["decided_at"]-al["created_at"]).dt.total_seconds()
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Decisions logged", len(al)); k2.metric("Confirmations", (al["decision"]=="confirm").sum()); k3.metric("Dismissals", (al["decision"]=="dismiss").sum())
        mean_latency=float(np.nanmean(al["latency_sec"])) if not al["latency_sec"].empty else float("nan"); k4.metric("Avg time to decision (s)", f"{mean_latency:.1f}" if not math.isnan(mean_latency) else "—")
        with st.expander("Audit log (latest 50)"):
            view=al.sort_values("decided_at", ascending=False).head(50).copy()
            view["decided_at"]=view["decided_at"].dt.strftime("%H:%M:%S"); view["created_at"]=view["created_at"].dt.strftime("%H:%M:%S")
            st.dataframe(view[["incident_id","decision","role","created_at","decided_at"]], use_container_width=True, height=260)
    else: st.info("No human decisions yet. Confirm or dismiss an incident to populate compliance KPIs.")

# ---------------- Governance (Charts++) ----------------
with tab_governance:
    _init_gov_state(); ss = st.session_state
    st.header("Governance, Transparency & Compliance")

    def compute_scores():
        controls = ss.gov_controls; di = ss.gov_data_inventory; al = pd.DataFrame(ss.audit_log)
        data_transparency = 0
        data_transparency += 25 if controls.get("data_minimization") else 0
        data_transparency += 25 if controls.get("privacy_by_design") else 0
        data_transparency += 25 if controls.get("traceability_audit") else 0
        pii_ratio = 0.0
        if not di.empty and "PII" in di.columns: pii_ratio = (di["PII"].astype(str).str.lower()=="yes").mean()
        data_transparency += 25 * (1 - pii_ratio)
        model_transparency = (35 if controls.get("transparency_cards") else 0) + (35 if controls.get("robustness_eval") else 0) + (30 if controls.get("lifecycle_mgmt") else 0)
        oversight = (50 if controls.get("human_oversight") else 0) + (50 if not al.empty else 0)
        robustness = (50 if controls.get("robustness_eval") else 0) + (50 if controls.get("lifecycle_mgmt") else 0)
        traceability = (50 if controls.get("traceability_audit") else 0) + (50 if not al.empty else 0)
        privacy = (50 if controls.get("privacy_by_design") else 0) + (50 if ss.get("gov_pseudonymization", False) else 0)
        return {"Data Transparency":int(data_transparency),"Model Transparency":int(model_transparency),
                "Human Oversight":int(oversight),"Robustness":int(robustness),"Traceability":int(traceability),"Privacy":int(privacy),"PII Ratio":float(pii_ratio)}
    scores = compute_scores()

    subtabs = st.tabs(["Overview","Controls & Roles","Data & Privacy","Model & Datasets","Risk Register","Audit & Exports","Ops Settings"])

    # Overview
    with subtabs[0]:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Model version", CFG.model_version)
        di = pd.DataFrame(ss.incidents) if ss.incidents else pd.DataFrame()
        c2.metric("Incidents (session)", 0 if di.empty else len(di))
        al_df = pd.DataFrame(ss.audit_log) if ss.audit_log else pd.DataFrame()
        c3.metric("Decisions logged", 0 if al_df.empty else len(al_df))
        c4.metric("Controls enabled", sum(bool(v) for v in ss.gov_controls.values()))
        colA, colB = st.columns([2,1], gap="large")
        with colA:
            cats = ["Data Transparency","Model Transparency","Human Oversight","Robustness","Traceability","Privacy"]
            vals = [scores[k] for k in cats]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself", name="Score"))
            fig_radar.update_layout(title="Compliance Dimensions (0–100)", polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=False)
            st.plotly_chart(fig_radar, use_container_width=True)
        with colB:
            enabled = sum(1 for v in ss.gov_controls.values() if v); disabled = len(ss.gov_controls) - enabled
            st.plotly_chart(px.bar(pd.DataFrame({"Status":["Enabled","Disabled"],"Count":[enabled,disabled]}),
                                   x="Status", y="Count", title="Controls coverage"), use_container_width=True)
        st.markdown("#### Human decisions over time")
        if not al_df.empty:
            tl = al_df.copy(); tl["minute"] = pd.to_datetime(tl["decided_at"]).dt.floor("T")
            line = tl.groupby("minute").size().reset_index(name="decisions")
            st.plotly_chart(px.line(line, x="minute", y="decisions", markers=True, title="Decision timeline"), use_container_width=True)
        else: st.info("No decisions yet. Confirm or dismiss an incident to populate this chart.")

    # Controls & Roles
    with subtabs[1]:
        st.subheader("Controls checklist")
        cols = st.columns(4); keys=list(ss.gov_controls.keys())
        for i,k in enumerate(keys):
            with cols[i%4]: ss.gov_controls[k] = st.checkbox(k.replace("_"," ").title(), value=ss.gov_controls[k])
        st.markdown("#### Role & Permission Matrix (heatmap)")
        rm = ss.gov_role_matrix.copy(); bool_cols=[c for c in rm.columns if c!="Role"]; mat=rm[bool_cols].astype(int)
        st.plotly_chart(px.imshow(mat, text_auto=True, aspect="auto", x=bool_cols, y=rm["Role"], title="Permissions (1=yes)"), use_container_width=True)

    # Data & Privacy
    with subtabs[2]:
        st.subheader("Data Inventory & Transparency")
        st.dataframe(ss.gov_data_inventory, use_container_width=True, height=240)
        c1,c2,c3 = st.columns(3)
        with c1: ss.gov_retention_days_incidents = st.number_input("Incident retention (days)", 30, 1825, ss.gov_retention_days_incidents, 1)
        with c2: ss.gov_pseudonymization = st.checkbox("Pseudonymization enabled", ss.gov_pseudonymization)
        with c3: lawful_basis = st.selectbox("GDPR lawful basis (deployment)", ["— Demo —","Legitimate interests (Art.6.1.f)","Public interest (Art.6.1.e)","Contract (Art.6.1.b)"], index=0)
        dii = ss.gov_data_inventory.copy()
        if not dii.empty and "PII" in dii.columns:
            pie_df = dii["PII"].astype(str).str.title().value_counts().reset_index(); pie_df.columns=["PII","Datasets"]
            st.plotly_chart(px.pie(pie_df, names="PII", values="Datasets", hole=0.5, title="PII vs Non-PII in Data Inventory"), use_container_width=True)
        rec_df = pd.DataFrame([
            {"Artifact":"Incidents","Days":ss.gov_retention_days_incidents,"Recommended":365},
            {"Artifact":"Telemetry (raw)","Days":int(dii["Retention_days"].max() if "Retention_days" in dii else 3),"Recommended":7},
        ])
        fig_ret = go.Figure(data=[go.Bar(name="Configured", x=rec_df["Artifact"], y=rec_df["Days"]),
                                  go.Bar(name="Recommended", x=rec_df["Artifact"], y=rec_df["Recommended"])])
        fig_ret.update_layout(barmode="group", title="Retention: configured vs recommended")
        st.plotly_chart(fig_ret, use_container_width=True)
        dpiamd = f"""# DPIA (Draft Template)
**Scope**: Site anomaly screening (RF, access, GNSS, integrity).
**Controller**: Your Org • **Model version**: {CFG.model_version}
**Lawful Basis**: {lawful_basis}
**Data categories**: Technical telemetry (no PII by design).
**Risks**: False positives; misclassification under novel attacks.
**Mitigations**: Human oversight, threshold tuning, monitoring, retraining, secure transport, retention {ss.gov_retention_days_incidents} days.
**Rights & safeguards**: Access, rectification, deletion; privacy by design; pseudonymization: {"ON" if ss.gov_pseudonymization else "OFF"}.
"""
        _download_bytes(dpiamd, "DPIA_draft.md", "⬇️ Download DPIA (draft)")

    # Model & Datasets
    with subtabs[3]:
        st.subheader("Model Transparency & Dataset Cards")
        mt = scores["Model Transparency"]; dt = scores["Data Transparency"]
        cA,cB = st.columns(2)
        with cA: st.plotly_chart(px.bar(pd.DataFrame({"Metric":["Model Transparency"],"Score":[mt]}), x="Metric", y="Score", range_y=[0,100], title="Model Transparency (0–100)"), use_container_width=True)
        with cB: st.plotly_chart(px.bar(pd.DataFrame({"Metric":["Data Transparency"],"Score":[dt]}), x="Metric", y="Score", range_y=[0,100], title="Data Transparency (0–100)"), use_container_width=True)
        st.markdown("#### Data Lineage (demo)")
        nodes=["Devices","Telemetry (synthetic)","Anomaly Engine","Incidents","Exports (JSON/MD)","Audit Log"]; nmap={n:i for i,n in enumerate(nodes)}
        links=dict(source=[nmap["Devices"],nmap["Telemetry (synthetic)"],nmap["Anomaly Engine"],nmap["Anomaly Engine"],nmap["Incidents"]],
                   target=[nmap["Telemetry (synthetic)"],nmap["Anomaly Engine"],nmap["Incidents"],nmap["Exports (JSON/MD)"],nmap["Audit Log"]],
                   value=[100,100,60,40,60])
        fig_sk = go.Figure(data=[go.Sankey(node=dict(label=nodes,pad=15,thickness=16),
                                           link=dict(source=links["source"], target=links["target"], value=links["value"]))])
        fig_sk.update_layout(title_text="Data Flow: Sources → Processing → Outputs", font_size=12)
        st.plotly_chart(fig_sk, use_container_width=True)
        st.markdown("#### Model Card"); mc = f"""# Model Card — TRUST DEMO
**Version**: {CFG.model_version}
**Intended use**: Site-level anomaly screening; advisory only.
**Limits**: Not for autonomous enforcement or user surveillance.
**Known failure modes**: RF congestion may cause false positives; unknown attack classes.
**Evaluation**: Threshold-based; qualitative expert review.
**Ethical considerations**: Human-in-the-loop, transparency, contestability.
"""
        st.code(mc, language="markdown"); _download_bytes(mc, f"model_card_{CFG.model_version}.md", "⬇️ Model Card (MD)")
        st.markdown("#### Datasheet for Datasets"); ds = """# Datasheet — Telemetry (synthetic)
**Composition**: Synthetic RF/access/GNSS/integrity proxies.
**Collection**: Procedural generation; no user data.
**Preprocessing**: Normalization & signal proxies.
**Uses**: Demo/training of anomaly screens.
**Distribution**: Internal demo only.
"""
        st.code(ds, language="markdown"); _download_bytes(ds, "datasheet_telemetry_synthetic.md", "⬇️ Datasheet (MD)")

    # Risk Register
    with subtabs[4]:
        st.subheader("Risk Register"); st.dataframe(ss.gov_risk_register, use_container_width=True, height=280)
        with st.expander("Add new risk"):
            r1,r2=st.columns([3,1]); new_risk=r1.text_input("Risk description"); new_sev=r2.selectbox("Severity",["Low","Medium","High"],index=1)
            r3,r4,r5=st.columns([1,2,1]); new_lik=r3.selectbox("Likelihood",["Low","Medium","High"],index=1); new_mit=r4.text_input("Mitigation"); new_owner=r5.text_input("Owner",value="—")
            if st.button("Add risk") and new_risk:
                ss.gov_risk_register = pd.concat([ss.gov_risk_register, pd.DataFrame([{"Risk":new_risk,"Severity":new_sev,"Likelihood":new_lik,"Mitigation":new_mit or "TBD","Owner":new_owner or "TBD","Status":"New"}])], ignore_index=True)
                st.success("Risk added.")
        _download_bytes(ss.gov_risk_register.to_csv(index=False), "risk_register.csv", "⬇️ Export CSV", mime="text/csv")

    # Audit & Exports
    with subtabs[5]:
        st.subheader("Audit Trail & Evidence")
        di_all=pd.DataFrame(ss.incidents) if ss.incidents else pd.DataFrame()
        al_all=pd.DataFrame(ss.audit_log) if ss.audit_log else pd.DataFrame()
        c1,c2=st.columns(2)
        with c1: st.markdown("**Incidents (session)**"); st.dataframe(di_all, use_container_width=True, height=250)
        with c2:
            st.markdown("**Decisions**")
            if not al_all.empty:
                av=al_all.copy(); av["created_at"]=pd.to_datetime(av["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S"); av["decided_at"]=pd.to_datetime(av["decided_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                st.dataframe(av, use_container_width=True, height=250)
            else: st.info("No human decisions logged yet.")
        st.markdown("#### Bulk Exports")
        _download_bytes(di_all.to_csv(index=False) if not di_all.empty else "", "incidents.csv", "⬇️ Incidents (CSV)", mime="text/csv")
        _download_bytes(di_all.to_json(orient="records", indent=2) if not di_all.empty else "[]", "incidents.json", "⬇️ Incidents (JSON)", mime="application/json")
        _download_bytes(al_all.to_csv(index=False) if not al_all.empty else "", "audit_log.csv", "⬇️ Audit Log (CSV)", mime="text/csv")
        _download_bytes(al_all.to_json(orient="records", indent=2) if not al_all.empty else "[]", "audit_log.json", "⬇️ Audit Log (JSON)", mime="application/json")

    # Ops Settings
    with subtabs[6]:
        st.subheader("Operational Settings (affect governance artifacts)")
        c1,c2,c3=st.columns(3)
        with c1: ss.gov_log_level = st.selectbox("Log level", ["Debug","Info","Warning","Error"], index=["Debug","Info","Warning","Error"].index(ss.gov_log_level))
        with c2: ss.gov_hash_algo = st.selectbox("Evidence hash", ["SHA-256","SHA-512"], index=0 if ss.gov_hash_algo=="SHA-256" else 1)
        with c3: _ = st.slider("Default incident threshold (reference)", 0.30, 0.95, float(CFG.threshold), 0.01)
        st.write("Changes reflect in DPIA, Model Card, and evidence context.")
