# app.py
import math
import time
from collections import deque
from dataclasses import dataclass
import json
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
import pydeck as pdk
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Silence noisy SHAP warning
warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier .* TreeExplainer shap values output has changed to a list of ndarray",
    category=UserWarning
)

SEED = 42
np.random.seed(SEED)

# =========================
# Config & constants
# =========================
@dataclass
class Config:
    n_devices: int = 30
    rolling_len: int = 8
    max_plot_points: int = 1200
    threshold: float = 0.75
    coverage: float = 0.90
    n_estimators: int = 60
    max_depth: int = 3
    learning_rate: float = 0.08

    # Sundsvall (move inland to avoid water)
    site_center: tuple = (62.4030, 17.3400)
    site_radius_m: float = 500

    # Scenario radii
    jam_radius_m: float = 200
    breach_radius_m: float = 120
    spoof_radius_m: float = 250

    retrain_on_start: bool = True

CFG = Config()

# Softer type-confidence threshold + low-conf fallback
TYPE_TAU = 0.50  # type classifier abstains if below this prob

RAW_FEATURES = [
    # RF/QoS (common)
    "rssi","snr","packet_loss","latency_ms","jitter_ms","throughput_mbps","channel_util",
    "noise_floor_dbm","phy_error_rate","cca_busy_frac","beacon_miss_rate",

    # Access/Auth (Wi-Fi & generic access)
    "deauth_rate","assoc_churn","eapol_retry_rate","dhcp_fail_rate","rogue_rssi_gap",

    # GNSS realism
    "pos_error_m","gnss_sats","gnss_hdop","gnss_doppler_var","gnss_clk_drift_ppm",
    "cno_mean_dbhz","cno_std_dbhz",

    # Cellular realism (Road profile)
    "rsrp_dbm","rsrq_db","sinr_db","bler","harq_nack_ratio",
    "rrc_reestablish","rlf_count","ho_fail_rate","attach_reject_rate","ta_anomaly","pci_anomaly",

    # Integrity / data-path
    "payload_entropy","ts_skew_s","seq_gap","dup_ratio","schema_violation_rate","hmac_fail_rate","crc_err"
]

FEATURE_GLOSSARY = {
    "snr": "Signal-to-Noise Ratio (dB): higher is cleaner link; <10 dB is often unreliable.",
    "packet_loss": "Packet loss (%): higher means more delivery failures.",
    "latency_ms": "Network latency (ms): time for a packet round-trip; lower is better.",
    "jitter_ms": "Latency variation (ms): unstable delay; lower is smoother.",
    "pos_error_m": "GNSS position error (m): higher suggests spoofing or weak satellites.",
    "noise_floor_dbm": "RF noise floor (dBm): higher noise reduces usable SNR.",
    "cca_busy_frac": "Channel busy fraction: how often the medium is sensed busy.",
    "phy_error_rate": "PHY-layer error rate / PER/BLER-ish.",
    "beacon_miss_rate": "Missed AP beacons (Wi-Fi/private-5G).",
    "deauth_rate": "Deauthentication bursts (Wi-Fi) / attach pressure (cellular).",
    "assoc_churn": "Association/roaming churn (or cellular reattach churn).",
    "eapol_retry_rate": "802.1X retries (or cellular attach/auth retries).",
    "dhcp_fail_rate": "DHCP failure rate (or session/IP setup failures).",
    "rogue_rssi_gap": "Rogue node signal − Legit signal: >0 means rogue is more attractive.",
    "payload_entropy": "Payload entropy: randomness; too low/too high can be suspicious.",
    "ts_skew_s": "Timestamp skew (s) vs wall clock: replay/stale hints.",
    "seq_gap": "Sequence gap ratio: missing counters/frames.",
    "dup_ratio": "Duplicate payloads: replay/reinjection symptoms.",
    "schema_violation_rate": "Payload schema violations: field types/units off.",
    "hmac_fail_rate": "Signature (HMAC) failures: integrity/auth mismatch.",
    "throughput_mbps": "Throughput (Mb/s): effective data rate.",
    "channel_util": "Channel utilization (%): how busy the medium is.",
    "rssi": "Received signal strength (dBm): higher is closer/clearer.",
    "auth_fail_rate": "Authentication failures.",
    "crc_err": "CRC errors (count): integrity errors at frame level.",
    # GNSS
    "gnss_sats": "Satellites used in fix (count): typical 8–14; sudden drops are suspicious.",
    "gnss_hdop": "Horizontal Dilution of Precision: geometry; <1 good, >2 poor. Odd combos can indicate spoofing.",
    "gnss_doppler_var": "Variance of Doppler residuals: inconsistency across satellites suggests spoofing.",
    "gnss_clk_drift_ppm": "Receiver clock drift (ppm): abnormal jumps suggest time spoof/replay.",
    "cno_mean_dbhz": "GNSS C/N₀ mean (dB-Hz): 30–45 typical; patterns shift under spoof/jam.",
    "cno_std_dbhz": "GNSS C/N₀ variability (dB-Hz): anomalous spread indicates manipulation.",
    # Cellular
    "rsrp_dbm": "Cellular RSRP (dBm): stronger (closer to 0) is better.",
    "rsrq_db": "Reference Signal Received Quality (dB): combines RSSI+RSRP; higher is better.",
    "sinr_db": "Signal-to-Interference-plus-Noise Ratio (dB): higher is better.",
    "bler": "Block Error Rate: fraction of failed code blocks; lower is better.",
    "harq_nack_ratio": "HARQ NACK ratio: retransmission demand; higher under interference.",
    "rrc_reestablish": "RRC re-establishments (rate): link recoveries.",
    "rlf_count": "Radio Link Failures (rate).",
    "ho_fail_rate": "Handover failure rate.",
    "attach_reject_rate": "Attach/registration reject rate.",
    "ta_anomaly": "Timing advance anomaly (proxy for uplink RF oddities).",
    "pci_anomaly": "Unexpected Physical Cell ID changes (rogue gNB / misconfig).",
}

DOMAIN_MAP = {
    # RF / QoS
    "rssi":"RF", "snr":"RF", "noise_floor_dbm":"RF", "phy_error_rate":"RF", "cca_busy_frac":"RF", "beacon_miss_rate":"RF",
    "packet_loss":"RF", "latency_ms":"RF", "jitter_ms":"RF", "throughput_mbps":"RF", "channel_util":"RF",
    "rsrp_dbm":"RF","rsrq_db":"RF","sinr_db":"RF","bler":"RF","harq_nack_ratio":"RF",
    # GNSS
    "pos_error_m":"GNSS","gnss_sats":"GNSS","gnss_hdop":"GNSS","gnss_doppler_var":"GNSS","gnss_clk_drift_ppm":"GNSS",
    "cno_mean_dbhz":"GNSS","cno_std_dbhz":"GNSS",
    # Access/Auth
    "deauth_rate":"Access","assoc_churn":"Access","eapol_retry_rate":"Access","dhcp_fail_rate":"Access","rogue_rssi_gap":"Access",
    "rrc_reestablish":"Access","rlf_count":"Access","ho_fail_rate":"Access","attach_reject_rate":"Access","ta_anomaly":"Access","pci_anomaly":"Access",
    # Integrity
    "payload_entropy":"Integrity","ts_skew_s":"Integrity","seq_gap":"Integrity","dup_ratio":"Integrity","schema_violation_rate":"Integrity","hmac_fail_rate":"Integrity","crc_err":"Integrity"
}

DEVICE_TYPES = ["AMR", "Truck", "Sensor", "Gateway"]
MOBILE_TYPES = {"AMR", "Truck"}

# =========================
# Streamlit page
# =========================
st.set_page_config(page_title="TRUST AI — Wireless Threats (Sundsvall)", layout="wide")
st.title("TRUST AI — Realistic Wireless Threat Detection (Sundsvall, Mid Sweden University)")
st.caption("AMR & logistics fleet • RF/network realism • LightGBM + SHAP + Conformal • Type classifier • Persona XAI • Live Training Progress + ETA")

# Sidebar
with st.sidebar:
    st.header("Demo Controls")
    profile = st.selectbox(
        "Comms profile",
        ["Yard (Wi-Fi/private-5G dominant)", "Road (Cellular 5G/LTE dominant)"],
        index=1
    )
    st.session_state["cellular_mode"] = profile.startswith("Road")

    scenario = st.selectbox(
        "Scenario",
        ["Normal", "Jamming (localized)", "Access Breach (AP/gNB)", "GPS Spoofing (subset)", "Data Tamper (gateway)"],
        index=0
    )

    jam_mode = None
    breach_mode = None
    if scenario.startswith("Jamming"):
        jam_mode = st.radio("Jamming type", ["Broadband noise", "Reactive", "Mgmt (deauth)"], index=0, key="jam_mode")
        CFG.jam_radius_m = st.slider("Jam coverage (m)", 50, 500, CFG.jam_radius_m, 10, key="jam_radius")
    if scenario.startswith("Access Breach"):
        breach_mode = st.radio("Breach mode", ["Evil Twin", "Rogue Open AP", "Credential hammer"], index=0, key="breach_mode")
        CFG.breach_radius_m = st.slider("Rogue node lure radius (m)", 50, 300, CFG.breach_radius_m, 10, key="breach_radius")
    if scenario.startswith("GPS Spoofing"):
        st.radio("Spoofing scope", ["Single device", "Localized area", "Site-wide"], index=1, key="spoof_mode")
        st.checkbox("Affect mobile (AMR/Truck) only", True, key="spoof_mobile_only")
        CFG.spoof_radius_m = st.slider("Spoof coverage (m)", 50, 500, CFG.spoof_radius_m, 10, key="spoof_radius")

    speed = st.slider("Playback speed (ticks/refresh)", 1, 10, 3)
    auto = st.checkbox("Auto stream", True)
    reset = st.button("Reset")

    st.divider()
    st.subheader("Model")
    use_conformal = st.checkbox("Conformal risk (calibrated p-value)", True)
    th_val = st.slider("Incident threshold (model prob.)", 0.30, 0.95, CFG.threshold, 0.01, key="th_slider")
    CFG.threshold = th_val
    if st.session_state.get("suggested_threshold") is not None:
        if st.button(f"Apply suggested threshold ({st.session_state.suggested_threshold:.2f})"):
            st.session_state.th_slider = float(st.session_state.suggested_threshold)
    st.caption("Alerts fire when probability ≥ threshold; p-value refines severity if enabled.")

    st.divider()
    st.subheader("Display & Access")
    show_map = st.checkbox("Show geospatial map", True)
    show_heatmap = st.checkbox("Show fleet heatmap (metric z-scores)", True)
    type_filter = st.multiselect("Show device types", DEVICE_TYPES, default=DEVICE_TYPES)
    role = st.selectbox("Viewer role", ["End User", "Domain Expert", "Regulator", "AI Builder", "Executive"], index=3)

    st.divider()
    retrain = st.button("Retrain model now")

    st.divider()
    help_mode = st.checkbox("Help mode (inline hints)", True)
    show_eu_status = st.checkbox("Show EU AI Act status banner", True)

# Simple onboarding + EU status
if help_mode:
    st.info(
        "How to use: ① Pick a **Scenario** → ② Watch **map, KPIs, charts** update → "
        "③ Open **Incidents** (role-aware explanations) → ④ See **Insights** (global behavior) → "
        "⑤ Check **Governance** for EU AI Act transparency & evidence."
    )
if show_eu_status:
    st.success(
        "EU AI Act status: **Limited/Minimal risk demo** (synthetic telemetry; no safety control loop). "
        "If integrated as a **safety component** or for **critical infrastructure control**, it may become **High-risk** with additional obligations."
    )

# =========================
# Helpers
# =========================
def to_df(X, cols): return pd.DataFrame(X, columns=cols)

def shap_pos(explainer, X_df):
    vals = explainer.shap_values(X_df)
    if isinstance(vals, list):
        return vals[1] if len(vals) > 1 else vals[0]
    return vals

def severity(prob, pval):
    high = (prob >= 0.85) or (pval is not None and pval <= 0.05)
    med  = (prob >= 0.70) or (pval is not None and pval <= 0.20)
    return ("High","red") if high else (("Medium","orange") if med else ("Low","green"))

def meters_to_latlon_offset(d_north_m, d_east_m, lat0):
    dlat = d_north_m / 111_111.0
    dlon = d_east_m / (111_111.0 * math.cos(math.radians(lat0)))
    return dlat, dlon

def rand_point_near(lat0, lon0, radius_m):
    r = radius_m * np.sqrt(np.random.rand())
    theta = 2*np.pi*np.random.rand()
    dn, de = r*np.cos(theta), r*np.sin(theta)
    dlat, dlon = meters_to_latlon_offset(dn, de, lat0)
    return lat0 + dlat, lon0 + dlon

def haversine_m(lat1, lon1, lat2, lon2):
    R=6371000
    p1,p2 = math.radians(lat1),math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def time_of_day_load(tick):
    return 0.5 + 0.5*math.sin((tick%600)/600*2*math.pi)

def fmt_eta(seconds):
    if seconds is None or not np.isfinite(seconds): return "—"
    seconds = max(0, int(seconds)); m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"

def glossary_line(feature_key):
    base = feature_key.split("_")[0]
    return FEATURE_GLOSSARY.get(base, base)

def why_this_alert_plain(reasons):
    lines=[]
    for r in reasons[:4]:
        fk = r["feature"]; base = fk.split("_")[0]
        desc = FEATURE_GLOSSARY.get(base, base).split(":")[0]
        impact = r["impact"]
        trend = "unusually high" if impact>0 else "unusually low"
        lines.append(f"- {desc} looked **{trend}** (impact {impact:+.2f}).")
    return "\n".join(lines) if lines else "Model saw unusual patterns across multiple signals."

def group_reasons_by_domain(reasons, topk=6):
    if not reasons: return pd.DataFrame(columns=["domain","impact"])
    agg={}
    for r in reasons[:topk]:
        base = r["feature"].split("_")[0]
        dom = DOMAIN_MAP.get(base, "Other")
        agg[dom] = agg.get(dom, 0.0) + abs(float(r["impact"]))
    data = [{"domain":k, "impact":v} for k,v in agg.items()]
    return pd.DataFrame(sorted(data, key=lambda x: x["impact"], reverse=True))

def get_device_history(device_id, fields):
    buf = st.session_state.dev_buf.get(device_id, [])
    if not buf: return pd.DataFrame()
    df = pd.DataFrame(list(buf))
    return df[fields] if all(f in df.columns for f in fields) else df

def incident_id(inc):
    return f"{inc['device_id']}_{inc['ts']}_{inc['tick']}"

def model_card_data():
    mc = {
        "model": "LightGBM (binary) on rolling-window features",
        "version": "v3.0-sundsvall",
        "features_per_device": len(feature_cols()),
        "window_length": CFG.rolling_len,
        "hyperparameters": {
            "n_estimators": CFG.n_estimators,
            "max_depth": CFG.max_depth,
            "learning_rate": CFG.learning_rate,
            "min_child_samples": 12,
            "subsample": 0.9,
            "colsample_bytree": 0.9
        },
        "detection_rule": f"probability >= {CFG.threshold:.2f}",
        "calibration": "Conformal p-value (coverage ~90%)" if st.session_state.get("conformal_scores") is not None else "Disabled",
        "training_metrics": st.session_state.get("metrics", {}),
        "data": {
            "source": "Synthetic demo telemetry (no personal data)",
            "raw_signals": RAW_FEATURES,
            "engineered_features_count": len(feature_cols())
        },
        "intended_use": "Demo of trustworthy anomaly detection for wireless/logistics",
        "limitations": [
            "Synthetic, physics-inspired signals",
            "Single-site example; not production-ready"
        ],
        "type_classifier": {
            "model": "DecisionTree (max_depth=4, class_weight=balanced)",
            "accuracy_est": st.session_state.get("type_metrics", {}).get("accuracy", None),
            "classes": st.session_state.get("type_metrics", {}).get("classes", []),
            "abstain_threshold": TYPE_TAU
        }
    }
    return mc

def data_schema_json():
    return {
        "raw_features": RAW_FEATURES,
        "engineered_features": feature_cols(),
        "pii": False,
        "notes": "Synthetic signals only; no personal data. Export for transparency."
    }

# ---- JSON sanitizer (handles numpy/pandas types) ----
def _to_builtin(o):
    import numpy as _np
    import pandas as _pd

    if isinstance(o, dict):
        return {str(k): _to_builtin(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_to_builtin(v) for v in o]
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    if isinstance(o, (_np.ndarray,)):
        return o.tolist()
    if isinstance(o, (_pd.Timestamp,)):
        return o.isoformat()
    return o

def audit_log_json():
    raw = {"incidents": st.session_state.get("incidents", [])}
    return _to_builtin(raw)

# =========================
# Session init
# =========================
def init_state():
    lat0, lon0 = CFG.site_center
    ap_lat, ap_lon = rand_point_near(lat0, lon0, 50)
    jam_lat, jam_lon = rand_point_near(lat0, lon0, 100)
    rog_lat, rog_lon = rand_point_near(lat0, lon0, 80)
    spf_lat, spf_lon = rand_point_near(lat0, lon0, 120)

    devices = []
    for i in range(CFG.n_devices):
        d_type = np.random.choice(DEVICE_TYPES, p=[0.45, 0.25, 0.20, 0.10])
        lat, lon = rand_point_near(lat0, lon0, CFG.site_radius_m)
        speed_mps = np.random.uniform(0.5, 2.5) if d_type in MOBILE_TYPES else 0.0
        devices.append({
            "device_id": f"D{i:03d}",
            "type": d_type,
            "lat": lat, "lon": lon,
            "speed_mps": speed_mps,
            "heading": np.random.uniform(0, 2*np.pi),
            "active": True
        })
    st.session_state.devices = pd.DataFrame(devices)
    st.session_state.ap = {"lat": ap_lat, "lon": ap_lon}
    st.session_state.jammer = {"lat": jam_lat, "lon": jam_lon}
    st.session_state.rogue = {"lat": rog_lat, "lon": rog_lon}
    st.session_state.spoofer = {"lat": spf_lat, "lon": spf_lon}
    st.session_state.tick = 0

    st.session_state.dev_buf = {row.device_id: deque(maxlen=CFG.rolling_len) for _, row in st.session_state.devices.iterrows()}
    st.session_state.last_features = {}
    st.session_state.latest_probs = {}
    st.session_state.fleet_records = deque(maxlen=CFG.max_plot_points)
    st.session_state.incidents = []
    st.session_state.group_incidents = []
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.explainer = None
    st.session_state.conformal_scores = None
    st.session_state.metrics = {}
    st.session_state.eval = {}
    st.session_state.last_train_secs = None
    st.session_state.suggested_threshold = None
    st.session_state.seq_counter = {row.device_id: 0 for _, row in st.session_state.devices.iterrows()}
    st.session_state.spoof_target_id = None
    st.session_state.incident_labels = {}  # incident_id -> {"ack": bool, "false_positive": bool}
    # Unique UI nonce to avoid duplicate keys across tabs
    st.session_state.ui_nonce = st.session_state.get("ui_nonce") or str(int(time.time()))

if "devices" not in st.session_state or reset:
    init_state()

# =========================
# Realistic metric synthesis
# =========================
def update_positions(df):
    latc, lonc = CFG.site_center
    for i in df.index:
        if df.at[i,"type"] in MOBILE_TYPES:
            h = df.at[i,"heading"] + np.random.normal(0, 0.2)
            df.at[i,"heading"] = h
            step = df.at[i,"speed_mps"] * np.random.uniform(0.6, 1.4)
            dn, de = step*math.cos(h), step*math.sin(h)
            dlat, dlon = meters_to_latlon_offset(dn, de, df.at[i,"lat"])
            lat_new = df.at[i,"lat"] + dlat
            lon_new = df.at[i,"lon"] + dlon

            # Bounce back inside if leaving the site circle
            dist = haversine_m(lat_new, lon_new, latc, lonc)
            if dist > CFG.site_radius_m:
                df.at[i,"heading"] = (h + math.pi/2) % (2*math.pi)
                vec_n = lat_new - latc
                vec_e = lon_new - lonc
                scale = 0.98 * CFG.site_radius_m / max(dist, 1.0)
                dlat_back = vec_n * scale
                dlon_back = vec_e * scale
                df.at[i,"lat"] = latc + dlat_back
                df.at[i,"lon"] = lonc + dlon_back
            else:
                df.at[i,"lat"] = lat_new
                df.at[i,"lon"] = lon_new
    return df

def rf_and_network_model(row, tick, scen=None, tamper_mode=None, crypto_enabled=True, training=False,
                         jam_mode=None, breach_mode=None):
    if scen is None: scen = scenario
    if jam_mode is None: jam_mode = st.session_state.get("jam_mode")
    if breach_mode is None: breach_mode = st.session_state.get("breach_mode")

    # (Removed training-time flip of cellular_mode)
    cellular_mode = st.session_state.get("cellular_mode", False)

    ap   = st.session_state.ap
    jam  = st.session_state.jammer
    rog  = st.session_state.rogue
    spf  = st.session_state.spoofer

    d_ap  = max(1.0, haversine_m(row.lat, row.lon, ap["lat"], ap["lon"]))
    d_jam = haversine_m(row.lat, row.lon, jam["lat"], jam["lon"])
    d_rog = haversine_m(row.lat, row.lon, rog["lat"], rog["lon"])
    d_spf = haversine_m(row.lat, row.lon, spf["lat"], spf["lon"])
    load  = time_of_day_load(tick)

    # ---------- Baseline RF/QoS ----------
    rssi = -40 - 18 * math.log10(d_ap) + np.random.normal(0, 2)
    rssi = float(np.clip(rssi, -90, -40))
    base_snr = 35 - 0.008 * d_ap + np.random.normal(0, 1.5)
    jam_penalty = 0.0

    noise_floor_dbm = -95 + 5*load + np.random.normal(0, 1.5)
    cca_busy_frac   = float(np.clip(0.20*load + np.random.normal(0,0.03), 0.0, 0.95))
    phy_error_rate  = float(np.clip(np.random.beta(1, 60), 0.0, 0.3))
    beacon_miss_rate= float(np.clip(np.random.beta(1,100), 0.0, 0.2))
    deauth_rate     = float(np.clip(np.random.beta(1,200), 0.0, 0.2))
    assoc_churn     = float(np.clip(np.random.beta(1,100), 0.0, 0.3))
    eapol_retry_rate= float(np.clip(np.random.beta(1,120), 0.0, 0.5))
    dhcp_fail_rate  = float(np.clip(np.random.beta(1,200), 0.0, 1.0))
    rogue_rssi_gap  = -30.0

    # Cellular KPIs (always present, but near-zero in Wi-Fi mode)
    if cellular_mode:
        rsrp_dbm = float(np.clip(-65 - 20*math.log10(d_ap/50.0) + np.random.normal(0,2.5), -120, -60))
        rsrq_db  = float(np.clip(-3.0 - 0.003*d_ap + np.random.normal(0,1.0), -19, -3))
        sinr_db  = float(np.clip(22 - 0.006*d_ap + np.random.normal(0,1.2), -5, 30))
        bler     = float(np.clip(np.random.beta(1,120), 0.0, 0.20))
        harq_nack_ratio = float(np.clip(np.random.beta(1,100), 0.0, 0.30))
        rrc_reestablish = float(np.clip(np.random.beta(1,200), 0.0, 0.10))
        rlf_count       = float(np.clip(np.random.beta(1,250), 0.0, 0.05))
        ho_fail_rate    = float(np.clip(np.random.beta(1,180), 0.0, 0.15))
        attach_reject_rate = float(np.clip(np.random.beta(1,250), 0.0, 0.10))
        ta_anomaly = float(np.random.rand()<0.01)
        pci_anomaly= float(np.random.rand()<0.01)
    else:
        rsrp_dbm = -115.0; rsrq_db = -18.0; sinr_db = 0.0
        bler=harq_nack_ratio=rrc_reestablish=rlf_count=ho_fail_rate=attach_reject_rate=0.0
        ta_anomaly=pci_anomaly=0.0

    # ---------- Jamming realism (cellular-aware) ----------
    if str(scen).startswith("Jamming") and d_jam <= CFG.jam_radius_m:
        reach = max(0.15, 1.0 - d_jam/CFG.jam_radius_m)
        if jam_mode == "Broadband noise" or jam_mode is None:
            bump = np.random.uniform(10, 22) * reach
            jam_penalty     += bump
            noise_floor_dbm += bump * np.random.uniform(0.8, 1.1)
            if cellular_mode:
                sinr_db -= 0.6*bump
                bler = float(np.clip(bler + 0.25*reach + 0.15*np.random.rand(), 0, 0.95))
                harq_nack_ratio = float(np.clip(harq_nack_ratio + 0.25*reach + 0.10*np.random.rand(), 0, 1.0))
                rrc_reestablish = float(np.clip(rrc_reestablish + 0.15*reach, 0, 1.0))
                ho_fail_rate    = float(np.clip(ho_fail_rate + 0.20*reach, 0, 1.0))
            else:
                cca_busy_frac    = float(np.clip(cca_busy_frac + 0.25*reach + 0.15*np.random.rand(), 0, 0.98))
                phy_error_rate   = float(np.clip(phy_error_rate + 0.20*reach + 0.10*np.random.rand(), 0, 0.95))
                beacon_miss_rate = float(np.clip(beacon_miss_rate + 0.10*reach + 0.10*np.random.rand(), 0, 1.0))
        elif jam_mode == "Reactive":
            util = (0.4 + 0.5*load)
            bump = np.random.uniform(7, 16) * reach * util
            jam_penalty     += bump
            noise_floor_dbm += bump * np.random.uniform(0.7, 1.0)
            if cellular_mode:
                sinr_db -= 0.5*bump
                bler = float(np.clip(bler + 0.30*reach*util + 0.10*np.random.rand(), 0, 0.98))
                harq_nack_ratio = float(np.clip(harq_nack_ratio + 0.25*reach*util + 0.10*np.random.rand(), 0, 1.0))
                ho_fail_rate    = float(np.clip(ho_fail_rate + 0.25*reach*util, 0, 1.0))
            else:
                cca_busy_frac    = float(np.clip(cca_busy_frac + 0.18*reach*util + 0.07*np.random.rand(), 0, 0.98))
                phy_error_rate   = float(np.clip(phy_error_rate + 0.28*reach*util, 0, 0.98))
        else:  # Mgmt (deauth)
            if cellular_mode:
                rrc_reestablish = float(np.clip(rrc_reestablish + 0.40*reach + 0.15*np.random.rand(), 0, 1.0))
                ho_fail_rate    = float(np.clip(ho_fail_rate + 0.15*reach, 0, 1.0))
            else:
                deauth_rate      = float(np.clip(deauth_rate + 0.45*reach + 0.20*np.random.rand(), 0, 1.0))
                assoc_churn      = float(np.clip(assoc_churn + 0.35*reach + 0.15*np.random.rand(), 0, 1.0))
                beacon_miss_rate = float(np.clip(beacon_miss_rate + 0.12*reach, 0, 1.0))

    snr = max(0.0, base_snr - jam_penalty)
    loss = 1/(1+np.exp(0.35*(snr-18))) + 0.18*load + np.random.normal(0, 0.02)
    loss = float(np.clip(loss*100, 0, 95))
    latency = float(max(5, 18 + 2.2*loss + 28*load + np.random.normal(0, 7)))
    jitter  = float(max(0.3, 1.2 + 0.06*loss + 9*load + np.random.normal(0, 1.3)))
    thr     = float(max(0.5, 95 - 0.9*loss - 45*load + np.random.normal(0, 6)))
    channel_util = float(np.clip(100*(0.4+0.5*load) + 100*(cca_busy_frac-0.15) + np.random.normal(0, 5), 0, 100))

    # ---------- Access Breach realism ----------
    if str(scen).startswith("Access Breach"):
        affect_wifi    = (row.type in {"AMR","Sensor","Gateway"})
        affect_cellular= (row.type in {"Truck","Gateway"})
        should_affect = (cellular_mode and affect_cellular) or ((not cellular_mode) and affect_wifi)
        if should_affect and d_rog <= CFG.breach_radius_m:
            if row.type in MOBILE_TYPES or np.random.rand()<0.3:
                rog_rssi = -40 - 18 * math.log10(max(1.0, d_rog)) + np.random.normal(0, 2)
                rogue_rssi_gap = float(rog_rssi - rssi)
                mode = st.session_state.get("breach_mode","Evil Twin")
                if mode == "Evil Twin":
                    if cellular_mode:
                        attach_reject_rate = float(np.clip(attach_reject_rate + np.random.uniform(0.25,0.60), 0, 1.0))
                        ho_fail_rate = float(np.clip(ho_fail_rate + np.random.uniform(0.20,0.40), 0, 1.0))
                        pci_anomaly = 1.0
                    else:
                        deauth_rate      = float(np.clip(deauth_rate + np.random.uniform(0.25, 0.60), 0, 1.0))
                        assoc_churn      = float(np.clip(assoc_churn + np.random.uniform(0.25, 0.50), 0, 1.0))
                        eapol_retry_rate = float(np.clip(eapol_retry_rate + np.random.uniform(0.10, 0.30), 0, 1.0))
                elif mode == "Rogue Open AP":
                    if cellular_mode:
                        rrc_reestablish = float(np.clip(rrc_reestablish + np.random.uniform(0.20,0.40),0,1.0))
                    else:
                        assoc_churn      = float(np.clip(assoc_churn + np.random.uniform(0.15, 0.35), 0, 1.0))
                        dhcp_fail_rate   = float(np.clip(dhcp_fail_rate + np.random.uniform(0.25, 0.60), 0, 1.0))
                else:  # Credential hammer
                    if cellular_mode:
                        attach_reject_rate = float(np.clip(attach_reject_rate + np.random.uniform(0.35,0.70), 0, 1.0))
                    else:
                        eapol_retry_rate = float(np.clip(eapol_retry_rate + np.random.uniform(0.35, 0.70), 0, 1.0))

    # ---------- GNSS baselines ----------
    pos_error = np.random.normal(2.0, 0.7)
    gnss_sats = int(np.clip(np.random.normal(11, 2.0), 5, 18))
    gnss_hdop = float(np.clip(np.random.normal(1.0, 0.25), 0.6, 2.5))
    gnss_doppler_var = float(np.clip(np.random.normal(3.0, 0.8), 0.5, 8.0))
    gnss_clk_drift_ppm = float(np.clip(np.random.normal(0.10, 0.05), 0.0, 0.5))
    cno_mean_dbhz = float(np.clip(np.random.normal(38, 3.0), 25, 48))
    cno_std_dbhz  = float(np.clip(np.random.normal(2.0, 0.8), 0.3, 6.0))

    # ---------- Data Tamper realism ----------
    if "seq_counter" not in st.session_state:
        st.session_state.seq_counter = {}
    if row.device_id not in st.session_state.seq_counter:
        st.session_state.seq_counter[row.device_id] = 0
    st.session_state.seq_counter[row.device_id] += 1

    device_ts = tick + np.random.normal(0, 0.05)
    payload_entropy = float(np.random.normal(5.8, 0.4))
    ts_skew_s = float(device_ts - tick)
    seq_gap = 0.0
    dup_ratio = 0.0
    schema_violation_rate = 0.0
    hmac_fail_rate = 0.0 if crypto_enabled else 0.0
    crc_err   = int(np.random.poisson(0.2))

    if str(scen).startswith("Data Tamper"):
        tm = tamper_mode
        if training and tm is None:
            tm = np.random.choice(["Replay","Constant injection","Bias/Drift","Bitflip/Noise","Scale/Unit mismatch"])
        if tm == "Replay":
            ts_skew_s += float(np.random.uniform(30, 120))
            dup_ratio += float(np.random.uniform(0.15, 0.50))
            seq_gap   += float(np.random.uniform(0.10, 0.40))
            if crypto_enabled: hmac_fail_rate += float(np.random.uniform(0.6, 1.0))
        elif tm == "Constant injection":
            payload_entropy = float(np.random.uniform(1.0, 2.5))
            dup_ratio += float(np.random.uniform(0.20, 0.60))
            schema_violation_rate += float(np.random.uniform(0.05, 0.20))
            if crypto_enabled: hmac_fail_rate += float(np.random.uniform(0.3, 0.8))
        elif tm == "Bias/Drift":
            drift_amt = float(np.random.uniform(5, 15))
            thr = max(1.0, thr - drift_amt)  # reduce throughput under drift
            latency += float(np.random.uniform(10, 30))
            payload_entropy = float(np.random.uniform(4.0, 5.0))
            schema_violation_rate += float(np.random.uniform(0.02, 0.08))
        elif tm == "Bitflip/Noise":
            payload_entropy = float(np.random.uniform(7.0, 8.0))
            crc_err += int(np.random.poisson(2))
            if crypto_enabled: hmac_fail_rate += float(np.random.uniform(0.5, 1.0))
        elif tm == "Scale/Unit mismatch":
            latency *= float(np.random.uniform(1.8, 3.0))
            schema_violation_rate += float(np.random.uniform(0.30, 0.70))
            payload_entropy = float(np.random.uniform(4.5, 6.0))

    # ---------- Training-time GPS spoofing ----------
    if str(scen).startswith("GPS Spoofing") and training:
        minutes = tick / 60.0
        bias = 30.0
        drift = 8.0 * minutes
        pos_error += np.random.uniform(0.6*bias, 1.2*bias) + 0.6*drift
        if np.random.rand() < 0.5: gnss_hdop = float(np.random.uniform(0.25, 0.5))
        else:                      gnss_hdop = float(np.random.uniform(2.5, 4.0))
        gnss_sats = int(np.random.randint(4, 8))
        gnss_doppler_var = float(np.random.uniform(4.5, 9.5))
        gnss_clk_drift_ppm = float(np.random.uniform(0.5, 1.5))
        # C/N0 patterns can look "too good" or unstable under spoof
        if np.random.rand()<0.5:
            cno_mean_dbhz = float(np.random.uniform(42, 47)); cno_std_dbhz = float(np.random.uniform(0.2, 0.8))
        else:
            cno_mean_dbhz = float(np.random.uniform(28, 34)); cno_std_dbhz = float(np.random.uniform(3.0, 6.0))

    return dict(
        # RF/QoS
        rssi=float(rssi), snr=float(snr), packet_loss=float(loss), latency_ms=float(latency),
        jitter_ms=float(jitter), throughput_mbps=float(thr), channel_util=float(channel_util),
        noise_floor_dbm=float(noise_floor_dbm), phy_error_rate=float(phy_error_rate),
        cca_busy_frac=float(cca_busy_frac), beacon_miss_rate=float(beacon_miss_rate),
        # Access/Auth
        deauth_rate=float(deauth_rate), assoc_churn=float(assoc_churn),
        eapol_retry_rate=float(eapol_retry_rate), dhcp_fail_rate=float(dhcp_fail_rate),
        rogue_rssi_gap=float(rogue_rssi_gap),
        # GNSS
        pos_error_m=float(max(0.3, pos_error)),
        gnss_sats=int(gnss_sats), gnss_hdop=float(gnss_hdop),
        gnss_doppler_var=float(gnss_doppler_var), gnss_clk_drift_ppm=float(gnss_clk_drift_ppm),
        cno_mean_dbhz=float(cno_mean_dbhz), cno_std_dbhz=float(cno_std_dbhz),
        # Cellular
        rsrp_dbm=float(rsrp_dbm), rsrq_db=float(rsrq_db), sinr_db=float(sinr_db),
        bler=float(bler), harq_nack_ratio=float(harq_nack_ratio),
        rrc_reestablish=float(rrc_reestablish), rlf_count=float(rlf_count),
        ho_fail_rate=float(ho_fail_rate), attach_reject_rate=float(attach_reject_rate),
        ta_anomaly=float(ta_anomaly), pci_anomaly=float(pci_anomaly),
        # Integrity / data-path
        payload_entropy=float(np.clip(payload_entropy, 0.0, 8.0)),
        ts_skew_s=float(ts_skew_s), seq_gap=float(np.clip(seq_gap, 0.0, 1.0)),
        dup_ratio=float(np.clip(dup_ratio, 0.0, 1.0)),
        schema_violation_rate=float(np.clip(schema_violation_rate, 0.0, 1.0)),
        hmac_fail_rate=float(np.clip(hmac_fail_rate, 0.0, 1.0)),
        crc_err=int(max(0, crc_err))
    )

# =========================
# Feature engineering
# =========================
def build_window_features(buffer_rows):
    df = pd.DataFrame(buffer_rows)
    if df.empty: return {}
    feats = {}
    for feat in RAW_FEATURES:
        s = df[feat]
        feats[f"{feat}_mean"] = s.mean()
        feats[f"{feat}_std"]  = s.std(ddof=0) if len(s)>1 else 0.0
        feats[f"{feat}_min"]  = s.min()
        feats[f"{feat}_max"]  = s.max()
        feats[f"{feat}_last"] = s.iloc[-1]
        if len(s)>=3:
            x=np.arange(len(s)); feats[f"{feat}_slope"]=float(np.polyfit(x, s.values, 1)[0])
        else:
            feats[f"{feat}_slope"]=0.0
        mu,sd= s.mean(), s.std(ddof=0) if len(s)>1 else 1.0
        feats[f"{feat}_z"] = 0.0 if sd==0 else (s.iloc[-1]-mu)/sd
        feats[f"{feat}_jump"]=float(s.iloc[-1]-s.iloc[-2]) if len(s)>=2 else 0.0
    return feats

def feature_cols():
    cols=[]
    aggs=["mean","std","min","max","last","slope","z","jump"]
    for f in RAW_FEATURES:
        for a in aggs: cols.append(f"{f}_{a}")
    return cols

# =========================
# Training (with progress, ETA, and eval stashing)
# =========================
def make_training_data(n_ticks=400, progress_cb=None, pct_start=0, pct_end=70):
    lat0, lon0 = CFG.site_center
    ap_lat, ap_lon = rand_point_near(lat0, lon0, 50)
    jam_lat, jam_lon = rand_point_near(lat0, lon0, 100)
    rog_lat, rog_lon = rand_point_near(lat0, lon0, 80)
    spf_lat, spf_lon = rand_point_near(lat0, lon0, 120)

    devs=[]
    for i in range(18):
        d_type=np.random.choice(DEVICE_TYPES, p=[0.5,0.2,0.2,0.1])
        lat,lon = rand_point_near(lat0, lon0, CFG.site_radius_m)
        devs.append(dict(device_id=f"T{i:02d}", type=d_type, lat=lat, lon=lon,
                         speed_mps=(np.random.uniform(0.5,2.5) if d_type in MOBILE_TYPES else 0.0),
                         heading=np.random.uniform(0,2*np.pi)))
    D=pd.DataFrame(devs)

    if "seq_counter" not in st.session_state:
        st.session_state.seq_counter = {}
    for dev_id in D["device_id"]:
        st.session_state.seq_counter.setdefault(dev_id, 0)

    buf = {d: deque(maxlen=CFG.rolling_len) for d in D.device_id}
    X_rows=[]; y=[]; labels=[]  # labels: Normal/Jamming/Breach/Spoof/Tamper
    t0 = time.time()
    for t in range(n_ticks):
        scen_code = np.random.choice(["Normal","J","Breach","GPS","Tamper"],
                                     p=[0.55,0.15,0.12,0.10,0.08])
        if   scen_code=="J":     sc="Jamming (localized)";  label_type="Jamming";  jm=np.random.choice(["Broadband noise","Reactive","Mgmt (deauth)"]); bm=None
        elif scen_code=="Breach":sc="Access Breach (AP/gNB)"; label_type="Breach"; bm=np.random.choice(["Evil Twin","Rogue Open AP","Credential hammer"]); jm=None
        elif scen_code=="GPS":   sc="GPS Spoofing (subset)"; label_type="Spoof";   jm=bm=None
        elif scen_code=="Tamper":sc="Data Tamper (gateway)"; label_type="Tamper";  jm=bm=None
        else:                    sc="Normal";               label_type="Normal";   jm=bm=None

        # Move devices
        for i in D.index:
            if D.at[i,"type"] in MOBILE_TYPES:
                D.at[i,"heading"] += np.random.normal(0,0.3)
                step = D.at[i,"speed_mps"]*np.random.uniform(0.6,1.4)
                dn,de = step*math.cos(D.at[i,"heading"]), step*math.sin(D.at[i,"heading"])
                dlat,dlon = meters_to_latlon_offset(dn,de,D.at[i,"lat"])
                D.at[i,"lat"]+=dlat; D.at[i,"lon"]+=dlon

        st.session_state.ap={"lat":ap_lat,"lon":ap_lon}
        st.session_state.jammer={"lat":jam_lat,"lon":jam_lon}
        st.session_state.rogue={"lat":rog_lat,"lon":rog_lon}
        st.session_state.spoofer={"lat":spf_lat,"lon":spf_lon}

        for _, row in D.iterrows():
            fake_row = type("R",(object,),row.to_dict())()
            m = rf_and_network_model(
                fake_row, t, sc,
                tamper_mode=None,
                crypto_enabled=bool(np.random.rand()<0.7),
                training=True,
                jam_mode=(jm if "Jamming" in sc else None),
                breach_mode=(bm if "Access Breach" in sc else None)
            )
            buf[row.device_id].append(m)
            feats = build_window_features(buf[row.device_id])
            if feats:
                X_rows.append(feats)
                y.append(0 if sc=="Normal" else 1)
                labels.append(label_type)

        if progress_cb:
            frac = (t+1)/n_ticks
            elapsed = time.time()-t0
            eta = (elapsed/frac)*(1-frac) if frac>0 else None
            pct = pct_start + (pct_end-pct_start)*frac
            progress_cb(int(pct), f"Synthesizing windows {int(frac*100)}%", eta)

    X=pd.DataFrame(X_rows).fillna(0.0); y=np.array(y); labels=np.array(labels)
    X_train,X_tmp,y_train,y_tmp,lab_train,lab_tmp = train_test_split(X,y,labels,test_size=0.40,random_state=SEED,shuffle=True,stratify=y)
    X_cal,X_test,y_cal,y_test,lab_cal,lab_test = train_test_split(X_tmp,y_tmp,lab_tmp,test_size=0.50,random_state=SEED,shuffle=True,stratify=y_tmp)
    return (X_train,y_train,lab_train),(X_cal,y_cal,lab_cal),(X_test,y_test,lab_test),X,labels

# Union of bases (Wi-Fi + Cellular) so type-head is profile-robust
def _select_type_bases():
    wifi_jam   = ["snr","noise_floor_dbm","phy_error_rate","cca_busy_frac","packet_loss","latency_ms","jitter_ms"]
    cell_jam   = ["sinr_db","rsrp_dbm","rsrq_db","bler","harq_nack_ratio","assoc_churn","packet_loss","latency_ms"]
    breach     = ["deauth_rate","assoc_churn","eapol_retry_rate","dhcp_fail_rate","rogue_rssi_gap","beacon_miss_rate","attach_reject_rate","pci_anomaly"]
    spoof      = ["pos_error_m","gnss_hdop","gnss_sats","gnss_doppler_var","gnss_clk_drift_ppm","cno_mean_dbhz","cno_std_dbhz"]
    tamper     = ["hmac_fail_rate","dup_ratio","seq_gap","ts_skew_s","schema_violation_rate","payload_entropy","crc_err"]
    bases = set(wifi_jam + cell_jam + breach + spoof + tamper)
    return sorted(list(bases))

def _cols_from_bases(all_cols, bases):
    return [c for c in all_cols if any(c.startswith(f"{b}_") for b in bases)]

def train_model_with_progress(n_ticks=350):
    bar = st.progress(0, text="Preparing training…")
    note = st.empty()
    t_start = time.time()

    def update(pct, msg, eta=None):
        label = f"{msg} • ETA {fmt_eta(eta)}" if eta is not None else msg
        bar.progress(min(100, int(pct)), text=label)

    (X_train,y_train,lab_train),(X_cal,y_cal,lab_cal),(X_test,y_test,lab_test),X_all,labels_all = make_training_data(
        n_ticks=n_ticks, progress_cb=update, pct_start=0, pct_end=70
    )

    cols=list(X_train.columns)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train); Xca=scaler.transform(X_cal); Xte=scaler.transform(X_test); Xall=scaler.transform(X_all)
    Xtr_df=to_df(Xtr,cols); Xca_df=to_df(Xca,cols); Xte_df=to_df(Xte,cols); Xall_df=to_df(Xall,cols)

    model = LGBMClassifier(
        n_estimators=CFG.n_estimators,
        max_depth=CFG.max_depth,
        learning_rate=CFG.learning_rate,
        subsample=0.9, colsample_bytree=0.9,
        min_child_samples=12, force_col_wise=True, random_state=SEED
    )

    iter0 = time.time()
    total_iters = CFG.n_estimators
    def lgbm_progress_callback(env):
        it = getattr(env, "iteration", 0)
        it = max(0, min(total_iters, it))
        frac = it/total_iters if total_iters>0 else 1.0
        elapsed = time.time() - iter0
        eta = (elapsed/frac)*(1-frac) if frac>0 else None
        pct = 70 + 20*frac
        update(pct, f"Training trees {it}/{total_iters}", eta)

    update(70, "Starting model training…")
    model.fit(Xtr_df, y_train, callbacks=[lgb.log_evaluation(period=0), lgbm_progress_callback])

    update(90, "Calibrating confidence…")
    cal_p = model.predict_proba(Xca_df)[:,1]
    cal_nc = 1 - np.where(y_cal==1, cal_p, 1-cal_p)

    update(93, "Evaluating detector…")
    te_p = model.predict_proba(Xte_df)[:,1]
    preds=(te_p>=CFG.threshold).astype(int)
    prec,rec,f1,_ = precision_recall_fscore_support(y_test,preds,average="binary",zero_division=0)
    auc = roc_auc_score(y_test, te_p)
    brier = brier_score_loss(y_test, te_p)
    ths = np.linspace(0.30, 0.90, 61)
    best_f1, best_th = 0.0, CFG.threshold
    for th in ths:
        pred = (te_p >= th).astype(int)
        _, _, f1c, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        if f1c > best_f1:
            best_f1, best_th = f1c, th
    st.session_state.suggested_threshold = float(best_th)

    # -------- Stage-2: Attack type classifier (on positives only) --------
    update(95, "Training attack type classifier…")
    pos_mask = labels_all != "Normal"
    if np.any(pos_mask):
        bases = _select_type_bases()  # union of bases
        type_cols = _cols_from_bases(cols, bases)
        X_pos = to_df(Xall[:, [cols.index(c) for c in type_cols]], type_cols)[pos_mask]
        y_pos = labels_all[pos_mask]
        type_clf = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=SEED)
        X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(X_pos, y_pos, test_size=0.25, random_state=SEED, stratify=y_pos)
        type_clf.fit(X_tr_t, y_tr_t)
        acc = float((type_clf.predict(X_te_t)==y_te_t).mean())
        type_expl = shap.TreeExplainer(type_clf)
        st.session_state.type_clf = type_clf
        st.session_state.type_cols = type_cols
        st.session_state.type_labels = sorted(list(np.unique(y_pos)))
        st.session_state.type_explainer = type_expl
        st.session_state.type_metrics = {"accuracy": acc, "classes": st.session_state.type_labels, "tau": TYPE_TAU}
    else:
        st.session_state.type_clf = None
        st.session_state.type_cols = []
        st.session_state.type_labels = []
        st.session_state.type_explainer = None
        st.session_state.type_metrics = {}

    # -------- Stash everything --------
    st.session_state.model=model
    st.session_state.scaler=scaler
    st.session_state.explainer=shap.TreeExplainer(model)
    st.session_state.conformal_scores=cal_nc
    st.session_state.metrics={"precision":prec,"recall":rec,"f1":f1,"auc":auc,"brier":brier}
    st.session_state.baseline = Xall_df
    st.session_state.eval = {"y_test": y_test, "te_p": te_p, "brier": brier}

    total_secs = time.time()-t_start
    st.session_state.last_train_secs = total_secs
    bar.progress(100, text=f"Training complete • {int(total_secs)}s")
    note.success(
        f"Model trained: AUC={auc:.2f}, Precision={prec:.2f}, Recall={rec:.2f} • "
        f"Brier={brier:.3f} • Duration {int(total_secs)}s • Suggested threshold={best_th:.2f} • "
        f"Type clf acc={st.session_state.type_metrics.get('accuracy','—')}"
    )

if (CFG.retrain_on_start and st.session_state.get("model") is None) or retrain:
    train_model_with_progress(n_ticks=350)

def conformal_pvalue(prob):
    cal=st.session_state.conformal_scores
    if cal is None: return None
    nc=1-prob
    return float((np.sum(cal>=nc)+1)/(len(cal)+1))

# =========================
# Streaming tick (batch-optimized)
# =========================
def tick_once():
    st.session_state.devices = update_positions(st.session_state.devices.copy())
    tick = st.session_state.tick

    if st.session_state.spoof_target_id is None:
        st.session_state.spoof_target_id = np.random.choice(st.session_state.devices["device_id"])

    fleet_rows=[]
    for _, row in st.session_state.devices.iterrows():
        m = rf_and_network_model(
            row, tick, scenario,
            tamper_mode=st.session_state.get("tamper_mode"),
            crypto_enabled=st.session_state.get("crypto_enabled", True),
            training=False,
            jam_mode=st.session_state.get("jam_mode"),
            breach_mode=st.session_state.get("breach_mode")
        )

        # Live GPS spoofing effect (aligned with training)
        if scenario.startswith("GPS Spoofing"):
            spf = st.session_state.spoofer
            d_spf = haversine_m(row.lat, row.lon, spf["lat"], spf["lon"])
            scope = st.session_state.get("spoof_mode", "Localized area")
            mobile_only = st.session_state.get("spoof_mobile_only", True)
            hit = (scope=="Site-wide") or \
                  (scope=="Single device" and row.device_id==st.session_state.spoof_target_id) or \
                  (scope=="Localized area" and d_spf <= CFG.spoof_radius_m)
            if hit and (not mobile_only or row.type in MOBILE_TYPES):
                minutes = tick / 60.0
                bias = 30.0
                drift = 8.0 * minutes
                scale = 1.0 if scope!="Localized area" else max(0.20, 1.0 - d_spf/CFG.spoof_radius_m)
                m["pos_error_m"] += scale * (np.random.uniform(0.6*bias, 1.2*bias) + 0.6*drift)
                if np.random.rand() < 0.5:
                    m["gnss_hdop"] = float(np.random.uniform(0.25, 0.5))
                else:
                    m["gnss_hdop"] = float(np.random.uniform(2.5, 4.0))
                m["gnss_sats"] = int(np.random.randint(4, 8))
                m["gnss_doppler_var"] = float(np.random.uniform(4.5, 9.5))
                m["gnss_clk_drift_ppm"] = float(np.random.uniform(0.5, 1.5))
                # C/N0 manipulations
                if np.random.rand()<0.5:
                    m["cno_mean_dbhz"] = float(np.random.uniform(42, 47)); m["cno_std_dbhz"] = float(np.random.uniform(0.2, 0.8))
                else:
                    m["cno_mean_dbhz"] = float(np.random.uniform(28, 34)); m["cno_std_dbhz"] = float(np.random.uniform(3.0, 6.0))

        st.session_state.dev_buf[row.device_id].append(m)
        rec = {"tick":int(tick), "device_id":row.device_id, "type":row.type, "lat":float(row.lat), "lon":float(row.lon), **m}
        fleet_rows.append(rec)

    device_ids=[]; feats_list=[]
    for _, row in st.session_state.devices.iterrows():
        feats = build_window_features(st.session_state.dev_buf[row.device_id])
        if feats:
            st.session_state.last_features[row.device_id] = feats
            device_ids.append(row.device_id)
            feats_list.append(feats)

    incidents_this_tick=[]
    st.session_state.latest_probs.clear()
    if feats_list:
        X = pd.DataFrame(feats_list).fillna(0.0)
        Xs = st.session_state.scaler.transform(X)
        probs = st.session_state.model.predict_proba(Xs)[:,1]
        for did, p in zip(device_ids, probs):
            st.session_state.latest_probs[did] = float(p)

        idx_alert = np.where(probs >= CFG.threshold)[0]
        if len(idx_alert)>0:
            expl_inputs = to_df(Xs, X.columns).iloc[idx_alert]
            shap_vals = shap_pos(st.session_state.explainer, expl_inputs)
            shap_arr = np.array(shap_vals)
            if shap_arr.ndim == 1:
                shap_arr = shap_arr[:, None]

            for j, idx in enumerate(idx_alert):
                did = device_ids[idx]
                row = st.session_state.devices[st.session_state.devices["device_id"]==did].iloc[0]
                prob = float(probs[idx])
                pval = conformal_pvalue(prob) if (use_conformal and st.session_state.get("conformal_scores") is not None) else None
                sev,_ = severity(prob, pval)
                shap_vec = shap_arr[j]
                pairs = sorted(list(zip(X.columns, shap_vec)), key=lambda kv: abs(kv[1]), reverse=True)[:6]

                # Stage-2 type classification (robust + fallback)
                type_label = "Unknown"; type_conf = None; type_pairs=[]
                if st.session_state.get("type_clf") is not None:
                    tcols_all = st.session_state.type_cols
                    tcols = [c for c in tcols_all if c in X.columns]
                    xrow_full = to_df(Xs, X.columns).iloc[[idx]]
                    xrow = xrow_full[tcols] if tcols else xrow_full

                    probs_type = st.session_state.type_clf.predict_proba(xrow)[0]
                    best_idx = int(np.argmax(probs_type))
                    best_p = float(probs_type[best_idx])
                    raw_label = st.session_state.type_clf.classes_[best_idx]

                    if best_p >= TYPE_TAU:
                        type_label, type_conf = raw_label, best_p
                    elif best_p >= 0.40:
                        type_label, type_conf = f"{raw_label} (low conf)", best_p
                    else:
                        type_label, type_conf = "Unknown", best_p

                    # explain type decision
                    try:
                        tvals = st.session_state.type_explainer.shap_values(xrow)
                        tv = tvals[best_idx][0] if isinstance(tvals, list) else tvals[0]
                        type_pairs = sorted(list(zip(xrow.columns, tv)), key=lambda kv: abs(kv[1]), reverse=True)[:6]
                    except Exception:
                        type_pairs=[]

                inc = dict(
                    ts=int(time.time()), tick=int(tick),
                    device_id=did, type=row.type, lat=float(row.lat), lon=float(row.lon),
                    scenario=scenario, prob=float(prob), p_value=(None if pval is None else float(pval)), severity=sev,
                    features=st.session_state.last_features[did],
                    reasons=[{"feature":k,"impact":float(v)} for k,v in pairs],
                    type_label=type_label, type_conf=type_conf,
                    type_reasons=[{"feature":k,"impact":float(v)} for k,v in type_pairs]
                )
                st.session_state.incidents.append(inc)
                incidents_this_tick.append(inc)

    if incidents_this_tick:
        affected = {i["device_id"] for i in incidents_this_tick}
        ratio = len(affected)/len(st.session_state.devices)
        if ratio>=0.25:
            st.session_state.group_incidents.append({
                "ts": int(time.time()), "tick": int(tick), "scenario": scenario,
                "affected": int(len(affected)), "fleet": int(len(st.session_state.devices)), "ratio": float(ratio),
            })

    st.session_state.fleet_records.extend(fleet_rows)
    st.session_state.tick += 1

if st.session_state.get("model") is not None:
    if auto:
        for _ in range(speed): tick_once()
    else:
        if st.button("Step once"): tick_once()

# =========================
# Transparency banner (engine)
# =========================
with st.container():
    m = st.session_state.metrics or {}
    cols = st.columns([1.2,1,1,1,1,1.2])
    cols[0].markdown("### Detection Engine")
    cols[1].metric("Model", "LightGBM")
    cols[2].metric("Window", f"{CFG.rolling_len} ticks")
    cols[3].metric("Features", f"{len(feature_cols())}")
    cols[4].metric("Threshold", f"{CFG.threshold:.2f}")
    cols[5].metric("Conformal", "On" if st.session_state.get("conformal_scores") is not None and use_conformal else "Off")

    with st.expander("How detection works"):
        st.markdown(
            """
**Pipeline**: Rolling-window features → **LightGBM** → probability `p`  
**Rule**: Raise incident when `p ≥ threshold`.  
**Confidence**: Conformal calibration gives a p-value.  
**XAI**: **SHAP** local (per alert) + global (Insights).  
**Type**: Small decision tree labels {Jamming, Breach, Spoof, Tamper} with abstain if low confidence.
            """
        )
        st.markdown("**Hyperparameters**")
        st.json({
            "n_estimators": CFG.n_estimators,
            "max_depth": CFG.max_depth,
            "learning_rate": CFG.learning_rate,
            "min_child_samples": 12,
            "subsample": 0.9,
            "colsample_bytree": 0.9
        })
        if m:
            st.markdown("**Training metrics**")
            st.json(m)

# =========================
# KPIs + metric glossary (inline)
# =========================
k1,k2,k3,k4,k5 = st.columns(5)
with k1: st.metric("Devices", len(st.session_state.devices))
with k2: st.metric("Incidents (session)", len(st.session_state.incidents))
with k3:
    auc = (st.session_state.metrics or {}).get("auc", 0)
    st.metric("Model AUC", f"{auc:.2f}")
    if help_mode:
        st.caption("**Model AUC**: discrimination; 0.5 = random, 1.0 = perfect. Higher is better.")
with k4:
    probs = list(st.session_state.latest_probs.values())
    st.metric("Fleet risk (mean prob)", f"{(np.mean(probs) if probs else 0):.2f}")
    if help_mode:
        st.caption("**Fleet risk (mean prob)**: average anomaly probability across devices now; higher implies site-wide stress.")
with k5:
    train_secs = st.session_state.get("last_train_secs")
    st.metric("Last train duration", f"{int(train_secs)}s" if train_secs else "—")

# =========================
# SHAP renderer for incidents (snapshot) — SCOPED KEYS
# =========================
def render_device_inspector_from_incident(inc, topk=8, scope="main"):
    feats = inc.get("features") or st.session_state.last_features.get(inc["device_id"])
    if not feats:
        st.info("Not enough samples for device window yet.")
        return

    base_key = f"{incident_id(inc)}_{scope}"

    X = pd.DataFrame([feats]).fillna(0.0)
    Xs = st.session_state.scaler.transform(X)
    Xs_df = pd.DataFrame(Xs, columns=X.columns)
    prob = float(st.session_state.model.predict_proba(Xs_df)[:, 1][0])

    # Risk gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        number={'suffix':"%"},
        gauge={
            'axis': {'range': [0,100]},
            'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.8, 'value': CFG.threshold*100}
        }
    ))
    st.plotly_chart(gauge, use_container_width=True, key=f"gauge_{base_key}")

    # Top SHAP contributions
    shap_vec = shap_pos(st.session_state.explainer, Xs_df)[0]
    pairs = sorted(zip(X.columns, shap_vec), key=lambda kv: abs(kv[1]), reverse=True)[:topk]
    df_shap = pd.DataFrame(pairs, columns=["feature", "impact"])
    fig = px.bar(df_shap.sort_values("impact"), x="impact", y="feature", orientation="h",
                 title=f"Top local contributions — {inc['device_id']}",
                 labels={"impact": "contribution → anomaly"})
    st.plotly_chart(fig, use_container_width=True, key=f"local_shap_{base_key}")

    # Mini history (last window)
    hist = get_device_history(inc["device_id"], ["snr","packet_loss","latency_ms","pos_error_m"])
    if len(hist)>0:
        hist = hist.reset_index(drop=True)
        c1,c2 = st.columns(2)
        c1.plotly_chart(px.line(hist, y="snr", title="SNR (last window)"), use_container_width=True, key=f"hist_snr_{base_key}")
        c1.plotly_chart(px.line(hist, y="packet_loss", title="Packet loss % (last window)"), use_container_width=True, key=f"hist_loss_{base_key}")
        c2.plotly_chart(px.line(hist, y="latency_ms", title="Latency ms (last window)"), use_container_width=True, key=f"hist_latency_{base_key}")
        c2.plotly_chart(px.line(hist, y="pos_error_m", title="GNSS error m (last window)"), use_container_width=True, key=f"hist_gnss_{base_key}")

    with st.expander("Raw window features (standardized input)"):
        st.dataframe(Xs_df.T.rename(columns={0: "z-value"}), use_container_width=True)

# =========================
# Role-aware incident rendering & categorization
# =========================
def incident_category(inc):
    s = inc.get("scenario", "")
    for cat in ["Jamming", "Access Breach", "GPS Spoofing", "Data Tamper"]:
        if s.startswith(cat): 
            return cat
    return "Other"

def render_incident_body_for_role(inc, role, scope="main"):
    base_key = f"{incident_id(inc)}_{scope}"
    pv = inc.get("p_value"); pv_str = f"{pv:.3f}" if pv is not None else "—"
    sev = inc.get("severity", "—")
    cols = st.columns(3)
    cols[0].metric("Severity", sev)
    cols[1].metric("Prob.", f"{inc['prob']:.2f}")
    cols[2].metric("p-value", pv_str)

    # Classification badge
    tlabel = inc.get("type_label","Unknown")
    tconf  = inc.get("type_conf", None)
    if tconf is not None:
        st.metric("Attack type", f"{tlabel}", f"{tconf*100:.0f}%")
    else:
        st.metric("Attack type", f"{tlabel}")

    if role == "End User":
        st.markdown("**What it means**")
        if "Jamming" in inc["scenario"]:
            st.write("Temporary slow or unstable connection nearby.")
        elif "Access Breach" in inc["scenario"]:
            st.write("Unknown/unsafe access node nearby disrupted connectivity.")
        elif "GPS Spoofing" in inc["scenario"]:
            st.write("Location may be inaccurate.")
        elif "Data Tamper" in inc["scenario"]:
            st.write("Some readings looked reused or out of range.")

        st.markdown("**What to do**")
        if "Jamming" in inc["scenario"]:
            st.write("Move 50–100 m away or retry; the network will change channel.")
        elif "Access Breach" in inc["scenario"]:
            st.write("Avoid joining unknown SSIDs/PLMNs; protections are active.")
        elif "GPS Spoofing" in inc["scenario"]:
            st.write("Use UWB/IMU fallback; limit speed in GNSS-denied mode.")
        else:
            st.write("Retry; if it persists, notify ops.")

    elif role == "Domain Expert":
        st.markdown("**Operational signals**")
        if "Jamming" in inc["scenario"]:
            st.write("↑ noise_floor_dbm, ↓ SNR / SINR, ↑ phy_error_rate/BLER, ↑ churn")
        elif "Access Breach" in inc["scenario"]:
            st.write("↑ deauth_rate, ↑ assoc_churn, ↑ eapol_retry_rate / dhcp_fail_rate, rogue_rssi_gap>0")
        elif "Data Tamper" in inc["scenario"]:
            st.write("↑ dup_ratio, ↑ ts_skew_s, ↑ schema_violation_rate, ↑ hmac_fail_rate (if enabled)")
        elif "GPS Spoofing" in inc["scenario"]:
            st.write("↑ pos_error_m with odd GNSS: hdop, sats, Doppler var, clock drift, C/N0 patterns")

        st.markdown("**Playbook**")
        if "Jamming" in inc["scenario"]:
            st.write("Channel hop ▸ Spectrum scan ▸ Directional/backup link.")
        elif "Access Breach" in inc["scenario"]:
            st.write("Quarantine SSID/PLMN ▸ Rotate credentials ▸ Rogue node sweep.")
        elif "Data Tamper" in inc["scenario"]:
            st.write("Reject stale/invalid payloads ▸ Verify signatures ▸ Audit gateway.")
        else:
            st.write("GNSS-denied nav ▸ Verify time source ▸ Geofence sanity.")

        with st.expander("Device Inspector (local SHAP)"):
            render_device_inspector_from_incident(inc, topk=8, scope=scope)

        # Why classified as <type>?
        treasons = inc.get("type_reasons", [])
        if treasons:
            st.markdown(f"**Why classified as _{inc.get('type_label','?')}_?**")
            df_tr = pd.DataFrame(treasons, columns=["feature","impact"])
            st.plotly_chart(
                px.bar(df_tr.sort_values("impact"), x="impact", y="feature", orientation="h",
                       title="Type head contributions"),
                use_container_width=True, key=f"type_shap_{incident_id(inc)}_{scope}"
            )
            dom_df = group_reasons_by_domain(treasons, topk=6)
            if len(dom_df)>0:
                st.plotly_chart(
                    px.bar(dom_df, x="impact", y="domain", orientation="h",
                           title="Domains driving classification (abs impact)"),
                    use_container_width=True, key=f"type_domains_{incident_id(inc)}_{scope}"
                )
        else:
            st.caption("Type classifier abstained or lacks enough signal.")

    elif role == "Regulator":
        st.markdown("**Assurance & governance**")
        st.write("- Calibrated confidence via conformal p-value (lower is stronger evidence).")
        st.write("- Audit trail available; no personal data—technical telemetry only.")
        evidence = {
            "ts": inc["ts"], "tick": inc["tick"], "device_id": inc["device_id"], "type": inc["type"],
            "lat": inc["lat"], "lon": inc["lon"], "scenario": inc["scenario"], "severity": inc["severity"],
            "prob": inc["prob"], "p_value": inc["p_value"], "model_version": "LightGBM v3.0-demo",
            "explanations": inc["reasons"], "type_label": inc.get("type_label"), "type_conf": inc.get("type_conf")
        }
        st.download_button(
            "Download incident evidence (JSON)",
            data=json.dumps(_to_builtin(evidence), indent=2).encode("utf-8"),
            file_name=f"incident_{inc['device_id']}_{inc['ts']}_{inc['tick']}.json",
            mime="application/json",
            key=f"dl_evidence_{base_key}"
        )

    elif role == "AI Builder":
        st.markdown("**Model view**")
        render_device_inspector_from_incident(inc, topk=8, scope=scope)
        with st.expander("Standardized feature vector (z-values)"):
            feats = inc.get("features") or st.session_state.last_features.get(inc["device_id"])
            if feats:
                X = pd.DataFrame([feats]).fillna(0.0)
                Xs = st.session_state.scaler.transform(X)
                Xs_df = pd.DataFrame(Xs, columns=X.columns)
                st.dataframe(Xs_df.T.rename(columns={0:"z"}), use_container_width=True)

        # Why classified as <type>?
        treasons = inc.get("type_reasons", [])
        if treasons:
            df_tr = pd.DataFrame(treasons, columns=["feature","impact"])
            st.plotly_chart(
                px.bar(df_tr.sort_values("impact"), x="impact", y="feature", orientation="h",
                       title="Type head contributions"),
                use_container_width=True, key=f"type_shap_{incident_id(inc)}_{scope}_builder"
            )

    else:  # Executive
        st.markdown("**Executive summary**")
        st.write(f"- Device **{inc['device_id']}** at risk: **{inc['severity']}**; scenario: **{inc['scenario']}**; type: **{inc.get('type_label','Unknown')}**.")
        st.write("- Impact: transient performance/security risk; mitigations active.")
        st.markdown("**KPIs**")
        st.write("- Incidents, MTTD, Packet loss, Latency, SNR/SINR, Deauth/BLER")

def render_incident_card(inc, role, scope="main"):
    base_key = f"{incident_id(inc)}_{scope}"
    pv = inc.get("p_value"); pv_str = f"{pv:.3f}" if pv is not None else "—"
    sev = inc.get("severity", "—")
    color = {"High":"red","Medium":"orange","Low":"green"}.get(sev, "gray")
    badge = f"<span style='background:{color};color:white;padding:2px 8px;border-radius:8px;'>{sev}</span>"

    cols = st.columns([1.8, 1])
    with cols[0]:
        st.markdown(
            f"**{inc['device_id']}** ({inc['type']}) · {inc['scenario']} · prob={inc['prob']:.2f} · p={pv_str} · {badge}",
            unsafe_allow_html=True
        )
        concise = [{"feature": r["feature"], "impact": r["impact"]} for r in inc["reasons"]][:3]
        st.markdown(why_this_alert_plain(concise))

        lcol, rcol = st.columns(2)
        with lcol:
            if st.button("Acknowledge", key=f"ack_{base_key}"):
                st.session_state.incident_labels[base_key] = {"ack": True, "false_positive": False}
        with rcol:
            if st.button("Mark false positive", key=f"fp_{base_key}"):
                st.session_state.incident_labels[base_key] = {"ack": True, "false_positive": True}

        label = st.session_state.incident_labels.get(base_key)
        if label:
            tag = "FALSE POSITIVE" if label.get("false_positive") else "ACKNOWLEDGED"
            st.caption(f"Label: **{tag}** (human oversight)")

        with st.expander("Details"):
            render_incident_body_for_role(inc, role, scope=scope)

    with cols[1]:
        st.json({k: inc[k] for k in ["tick","device_id","type","prob","p_value"]})

# =========================
# Layout tabs
# =========================
tab_overview, tab_fleet, tab_incidents, tab_insights, tab_governance = st.tabs(
    ["Overview", "Fleet View", "Incidents", "Insights", "Governance"]
)

# ---------- Overview
with tab_overview:
    left, right = st.columns([2,1])

    with left:
        if show_map and st.session_state.get("model") is not None:
            df_map = st.session_state.devices.copy()
            if type_filter and len(type_filter) < len(DEVICE_TYPES):
                df_map = df_map[df_map["type"].isin(type_filter)].copy()

            df_map["risk"] = df_map["device_id"].map(st.session_state.latest_probs).fillna(0.0)
            snrs, losses = [], []
            for _, r in df_map.iterrows():
                buf = st.session_state.dev_buf.get(r.device_id, [])
                if buf and len(buf) > 0:
                    snrs.append(float(buf[-1].get("snr", np.nan)))
                    losses.append(float(buf[-1].get("packet_loss", np.nan)))
                else:
                    snrs.append(np.nan); losses.append(np.nan)
            df_map["snr"] = snrs; df_map["packet_loss"] = losses

            type_colors = {
                "AMR":     [0, 128, 255, 220],
                "Truck":   [255, 165, 0, 220],
                "Sensor":  [34, 197, 94, 220],
                "Gateway": [147, 51, 234, 220],
            }
            df_map["fill_color"] = df_map["type"].map(type_colors)
            df_map["label"] = df_map.apply(lambda r: f"{r.device_id} ({r.type})", axis=1)
            df_map["radius"] = 6 + (df_map["risk"] * 16)

            layers = [
                pdk.Layer("ScatterplotLayer", data=df_map, get_position='[lon, lat]',
                          get_fill_color='fill_color', get_radius='radius',
                          get_line_color=[0,0,0,140], get_line_width=1, pickable=True),
                pdk.Layer("TextLayer", data=df_map, get_position='[lon, lat]', get_text='label',
                          get_color=[20,20,20,255], get_size=12, get_alignment_baseline="'top'", get_pixel_offset=[0,10]),
            ]

            # AP / gNB markers based on profile
            cellular_mode = st.session_state.get("cellular_mode", False)
            infra_label = "gNB" if cellular_mode else "AP"
            rogue_label = "Rogue gNB" if cellular_mode else "Rogue AP"

            ap_df = pd.DataFrame([{"lat": st.session_state.ap["lat"], "lon": st.session_state.ap["lon"], "label": infra_label}])
            layers += [
                pdk.Layer("ScatterplotLayer", data=ap_df, get_position='[lon, lat]',
                          get_fill_color='[30,144,255,240]', get_radius=10),
                pdk.Layer("TextLayer", data=ap_df, get_position='[lon, lat]', get_text='label',
                          get_color=[30,144,255,255], get_size=14, get_alignment_baseline="'bottom'", get_pixel_offset=[0,-10]),
            ]

            # Risk hazard overlay (⚠ + red ring)
            warn_df = df_map[df_map["risk"] >= CFG.threshold].copy()
            if len(warn_df) > 0:
                warn_df["warn"] = "⚠"
                layers += [
                    pdk.Layer("TextLayer", data=warn_df, get_position='[lon, lat]', get_text='warn',
                              get_color=[255,0,0,255], get_size=18, get_alignment_baseline="'bottom'", get_pixel_offset=[0,-18]),
                    pdk.Layer("ScatterplotLayer", data=warn_df, get_position='[lon, lat]',
                              get_fill_color='[0,0,0,0]', get_radius=26,
                              stroked=True, get_line_color=[255,0,0,200], get_line_width=2)
                ]

            # Coverage circles helper
            def circle_layer(center, radius_m, color):
                angles = np.linspace(0, 2*np.pi, 60)
                lat_mean = float(st.session_state.devices.lat.mean())
                path = [[
                    center["lon"] + meters_to_latlon_offset(radius_m * math.sin(a), radius_m * math.cos(a), lat_mean)[1],
                    center["lat"] + meters_to_latlon_offset(radius_m * math.sin(a), radius_m * math.cos(a), lat_mean)[0]
                ] for a in angles]
                return pdk.Layer("PathLayer", [{"path": path}], get_path="path", get_color=color, width_scale=4, width_min_pixels=1, opacity=0.25)

            # Scenario overlays
            if scenario.startswith("Jamming"):
                jam = st.session_state.jammer
                jam_df = pd.DataFrame([{"lat": jam["lat"], "lon": jam["lon"], "label": "Jammer"}])
                layers += [
                    pdk.Layer("ScatterplotLayer", data=jam_df, get_position='[lon, lat]',
                              get_fill_color='[255,0,0,240]', get_radius=10),
                    pdk.Layer("TextLayer", data=jam_df, get_position='[lon, lat]', get_text='label',
                              get_color=[255,0,0,255], get_size=14, get_alignment_baseline="'bottom'", get_pixel_offset=[0,-10]),
                    circle_layer(jam, CFG.jam_radius_m, [255,0,0])
                ]
            if scenario.startswith("Access Breach"):
                rog = st.session_state.rogue
                rog_df = pd.DataFrame([{"lat": rog["lat"], "lon": rog["lon"], "label": rogue_label}])
                layers += [
                    pdk.Layer("ScatterplotLayer", data=rog_df, get_position='[lon, lat]',
                              get_fill_color='[0,255,255,240]', get_radius=10),
                    pdk.Layer("TextLayer", data=rog_df, get_position='[lon, lat]', get_text='label',
                              get_color=[0,200,200,255], get_size=14, get_alignment_baseline="'bottom'", get_pixel_offset=[0,-10]),
                    circle_layer(rog, CFG.breach_radius_m, [0,200,200])
                ]
            if scenario.startswith("GPS Spoofing"):
                spf = st.session_state.spoofer
                spf_df = pd.DataFrame([{"lat": spf["lat"], "lon": spf["lon"], "label": "Spoofer"}])
                layers += [
                    pdk.Layer("ScatterplotLayer", data=spf_df, get_position='[lon, lat]',
                              get_fill_color='[255,215,0,240]', get_radius=10),
                    pdk.Layer("TextLayer", data=spf_df, get_position='[lon, lat]', get_text='label',
                              get_color=[255,215,0,255], get_size=14, get_alignment_baseline="'bottom'", get_pixel_offset=[0,-10]),
                    circle_layer(spf, CFG.spoof_radius_m, [255,215,0])
                ]

            view_state = pdk.ViewState(
                latitude=float(st.session_state.devices.lat.mean()),
                longitude=float(st.session_state.devices.lon.mean()),
                zoom=14, pitch=0
            )
            tooltip = {"html": "<b>{device_id}</b> • {type}<br/>Risk: {risk:.2f}<br/>SNR: {snr} dB<br/>Loss: {packet_loss}%",
                       "style": {"backgroundColor": "rgba(255,255,255,0.95)", "color": "#111"}}
            st.pydeck_chart(
                pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None, tooltip=tooltip),
                use_container_width=True
            )

            st.markdown(
                """
                <div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;">
                  <span style="display:inline-flex;align-items:center;gap:6px;">
                    <span style="width:12px;height:12px;background:#0080FF;border-radius:2px;display:inline-block;"></span> AMR
                  </span>
                  <span style="display:inline-flex;align-items:center;gap:6px;">
                    <span style="width:12px;height:12px;background:#FFA500;border-radius:2px;display:inline-block;"></span> Truck
                  </span>
                  <span style="display:inline-flex;align-items:center;gap:6px;">
                    <span style="width:12px;height:12px;background:#22C55E;border-radius:2px;display:inline-block;"></span> Sensor
                  </span>
                  <span style="display:inline-flex;align-items:center;gap:6px;">
                    <span style="width:12px;height:12px;background:#9333EA;border-radius:2px;display:inline-block;"></span> Gateway
                  </span>
                  <span style="margin-left:auto;opacity:.8;">Dot size = risk • ⚠ = above threshold</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Fleet averages over time (including SNR)
        fr = pd.DataFrame(list(st.session_state.fleet_records))
        if len(fr)>0:
            for y in ["snr","packet_loss","latency_ms","pos_error_m"]:
                sub = fr.groupby("tick")[y].mean().reset_index()
                fig=px.line(sub, x="tick", y=y, title=f"Fleet avg {y}")
                st.plotly_chart(fig, use_container_width=True, key=f"overview_{y}")
                if y == "snr" and help_mode:
                    st.caption("**Fleet avg SNR**: mean link quality; <10 dB shaky; >20 dB healthy.")

    with right:
        leaderboard=[]
        if st.session_state.get("model") is not None:
            for _, r in st.session_state.devices.iterrows():
                prob = st.session_state.latest_probs.get(r.device_id, 0.0)
                pval=conformal_pvalue(prob) if use_conformal else None
                sev,_=severity(prob,pval)
                leaderboard.append({"device_id":r.device_id,"type":r.type,"prob":prob,"p_value":pval,"severity":sev})
        if leaderboard:
            df_lead=pd.DataFrame(leaderboard).sort_values("prob",ascending=False).head(10)
            st.markdown("### Top risk devices")
            st.dataframe(df_lead, use_container_width=True)

# ---------- Fleet View
with tab_fleet:
    fr = pd.DataFrame(list(st.session_state.fleet_records))
    if len(fr)>0 and show_heatmap:
        recent = fr[fr["tick"]>=st.session_state.tick-40]
        cols = ["snr","packet_loss","latency_ms","jitter_ms","pos_error_m","auth_fail_rate","crc_err","throughput_mbps","channel_util",
                "noise_floor_dbm","cca_busy_frac","phy_error_rate","deauth_rate","assoc_churn","eapol_retry_rate","dhcp_fail_rate"]
        cols = [c for c in cols if c in recent.columns]
        mat = recent.groupby("device_id")[cols].mean()
        z = (mat-mat.mean())/mat.std(ddof=0).replace(0,1)
        fig = px.imshow(z.T, color_continuous_scale="RdBu_r", aspect="auto",
                        labels=dict(color="z-score"), title="Fleet heatmap (recent mean z-scores)")
        st.plotly_chart(fig, use_container_width=True, key="fleet_heatmap")
    st.dataframe(st.session_state.devices, use_container_width=True)

# ---------- Incidents
with tab_incidents:
    st.subheader("Incidents")

    all_inc = st.session_state.incidents
    if not all_inc:
        st.success("No incidents yet.")
    else:
        # Build categories (by scenario)
        cat_map = {}
        for inc in reversed(all_inc):  # newest first
            cat_map.setdefault(incident_category(inc), []).append(inc)

        # Stable order with counts
        order = ["Jamming", "Access Breach", "GPS Spoofing", "Data Tamper", "Other"]
        cats_present = [c for c in order if c in cat_map] + [c for c in cat_map if c not in order]
        tab_labels = [f"{c} ({len(cat_map[c])})" for c in cats_present]
        tabs = st.tabs(tab_labels + [f"All ({len(all_inc)})"])

        # Per-category tabs
        for t_idx, c in enumerate(cats_present):
            with tabs[t_idx]:
                sev_sel = st.multiselect("Show severities", ["High","Medium","Low"],
                                         default=["High","Medium","Low"], key=f"sev_{c}")
                for inc in cat_map[c]:
                    if inc["severity"] in sev_sel:
                        render_incident_card(inc, role, scope=f"cat_{c}")

        # 'All' tab
        with tabs[-1]:
            sev_sel_all = st.multiselect("Show severities", ["High","Medium","Low"],
                                         default=["High","Medium","Low"], key="sev_all_tabs")
            for inc in all_inc:
                if inc["severity"] in sev_sel_all:
                    render_incident_card(inc, role, scope="all")

    if st.session_state.incidents:
        df_inc = pd.DataFrame(st.session_state.incidents)
        df_inc["top_features"] = df_inc["reasons"].apply(lambda r: "; ".join([f"{x['feature']}:{x['impact']:+.3f}" for x in r]))
        csv = df_inc.drop(columns=["reasons","type_reasons"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download incidents CSV", csv, "incidents.csv", "text/csv", key="dl_incidents_csv")

# ---------- Insights (unique keys via ui_nonce)
with tab_insights:
    nonce = st
