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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Silence a noisy SHAP warning we already handle
warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier .* TreeExplainer shap values output has changed to a list of ndarray",
    category=UserWarning
)

# =========================
# Config & constants
# =========================
SEED = 42
np.random.seed(SEED)

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

    # Sundsvall — Mid Sweden University area
    site_center: tuple = (62.3925, 17.3066)
    site_radius_m: float = 400

    # Scenario radii
    jam_radius_m: float = 200
    breach_radius_m: float = 120
    spoof_radius_m: float = 250

    retrain_on_start: bool = True

CFG = Config()

# Telemetry features (base + integrity/freshness + Wi-Fi/Jam)
RAW_FEATURES = [
    "rssi","snr","packet_loss","latency_ms","jitter_ms",
    "pos_error_m","auth_fail_rate","crc_err","throughput_mbps","channel_util",
    # Integrity / freshness / semantics
    "payload_entropy","ts_skew_s","seq_gap","dup_ratio","schema_violation_rate","hmac_fail_rate",
    # Wi-Fi breach & jamming signals
    "noise_floor_dbm","cca_busy_frac","phy_error_rate","beacon_miss_rate","deauth_rate",
    "assoc_churn","eapol_retry_rate","dhcp_fail_rate","rogue_rssi_gap"
]

DEVICE_TYPES = ["AMR", "Truck", "Sensor", "Gateway"]
MOBILE_TYPES = {"AMR", "Truck"}

# =========================
# Streamlit page
# =========================
st.set_page_config(page_title="TRUST AI — Wireless Threats (Sundsvall)", layout="wide")
st.title("TRUST AI — Realistic Wireless Threat Detection (Sundsvall, Mid Sweden University)")
st.caption("AMR & logistics fleet • RF/network realism • LightGBM + SHAP + Conformal • Persona XAI • Live Training Progress + ETA")

# Sidebar
with st.sidebar:
    st.header("Demo Controls")
    scenario = st.selectbox(
        "Scenario",
        ["Normal", "Jamming (localized)", "Wi-Fi Breach (AP)", "GPS Spoofing (subset)", "Data Tamper (gateway)"],
        index=0
    )

    # Scenario-specific controls
    jam_mode = None
    breach_mode = None
    if scenario.startswith("Jamming"):
        jam_mode = st.radio("Jamming type", ["Broadband noise", "Reactive", "Mgmt (deauth)"], index=0, key="jam_mode")
        CFG.jam_radius_m = st.slider("Jam coverage (m)", 50, 500, CFG.jam_radius_m, 10, key="jam_radius")
    if scenario.startswith("Wi-Fi Breach"):
        breach_mode = st.radio("Breach mode", ["Evil Twin", "Rogue Open AP", "Credential hammer"], index=0, key="breach_mode")
        CFG.breach_radius_m = st.slider("Rogue AP lure radius (m)", 50, 300, CFG.breach_radius_m, 10, key="breach_radius")
    if scenario.startswith("GPS Spoofing"):
        spoof_mode = st.radio("Spoofing scope", ["Single device", "Localized area", "Site-wide"], index=1)
        spoof_mobile_only = st.checkbox("Affect mobile (AMR/Truck) only", True)
        CFG.spoof_radius_m = st.slider("Spoof coverage (m)", 50, 500, CFG.spoof_radius_m, 10)

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
    st.caption("Alerts fire when model probability exceeds this threshold.")

    st.divider()
    st.subheader("Display & Access")
    show_map = st.checkbox("Show geospatial map", True)
    show_heatmap = st.checkbox("Show fleet heatmap (metric z-scores)", True)
    type_filter = st.multiselect("Show device types", DEVICE_TYPES, default=DEVICE_TYPES)
    role = st.selectbox("Viewer role", ["End User", "Domain Expert", "Regulator", "AI Builder", "Executive"], index=3)

    st.divider()
    retrain = st.button("Retrain model now")

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
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"

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
    st.session_state.fleet_records = deque(maxlen=CFG.max_plot_points)
    st.session_state.incidents = []
    st.session_state.group_incidents = []
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.explainer = None
    st.session_state.conformal_scores = None
    st.session_state.metrics = {}
    st.session_state.last_train_secs = None
    st.session_state.suggested_threshold = None
    st.session_state.seq_counter = {row.device_id: 0 for _, row in st.session_state.devices.iterrows()}

if "devices" not in st.session_state or reset:
    init_state()

# =========================
# Realistic metric synthesis
# =========================
def update_positions(df):
    for i in df.index:
        if df.at[i,"type"] in MOBILE_TYPES:
            h = df.at[i,"heading"] + np.random.normal(0, 0.2)
            df.at[i,"heading"] = h
            step = df.at[i,"speed_mps"] * np.random.uniform(0.6, 1.4)
            dn, de = step*math.cos(h), step*math.sin(h)
            dlat, dlon = meters_to_latlon_offset(dn, de, df.at[i,"lat"])
            df.at[i,"lat"] += dlat
            df.at[i,"lon"] += dlon
    return df

def rf_and_network_model(row, tick, scen=None, tamper_mode=None, crypto_enabled=True, training=False,
                         jam_mode=None, breach_mode=None):
    """
    Generate correlated metrics based on distances and load for the selected scenario.
    """
    if scen is None: scen = scenario
    if jam_mode is None: jam_mode = st.session_state.get("jam_mode")
    if breach_mode is None: breach_mode = st.session_state.get("breach_mode")

    ap   = st.session_state.ap
    jam  = st.session_state.jammer
    rog  = st.session_state.rogue
    spf  = st.session_state.spoofer

    d_ap  = max(1.0, haversine_m(row.lat, row.lon, ap["lat"], ap["lon"]))
    d_jam = haversine_m(row.lat, row.lon, jam["lat"], jam["lon"])
    d_rog = haversine_m(row.lat, row.lon, rog["lat"], rog["lon"])
    d_spf = haversine_m(row.lat, row.lon, spf["lat"], spf["lon"])
    load  = time_of_day_load(tick)

    # ---------- Base RF / network ----------
    rssi = -40 - 18 * math.log10(d_ap) + np.random.normal(0, 2)
    rssi = float(np.clip(rssi, -90, -40))
    base_snr = 35 - 0.008 * d_ap + np.random.normal(0, 1.5)
    jam_penalty = 0.0

    # ---------- Baseline Wi-Fi/PHY signals ----------
    noise_floor_dbm = -95 + 5*load + np.random.normal(0, 1.5)
    cca_busy_frac   = float(np.clip(0.20*load + np.random.normal(0,0.03), 0.0, 0.95))
    phy_error_rate  = float(np.clip(np.random.beta(1, 60), 0.0, 0.3))
    beacon_miss_rate= float(np.clip(np.random.beta(1,100), 0.0, 0.2))
    deauth_rate     = float(np.clip(np.random.beta(1,200), 0.0, 0.2))
    assoc_churn     = float(np.clip(np.random.beta(1,100), 0.0, 0.3))
    eapol_retry_rate= float(np.clip(np.random.beta(1,120), 0.0, 0.5))
    dhcp_fail_rate  = float(np.clip(np.random.beta(1,200), 0.0, 1.0))
    rogue_rssi_gap  = -30.0  # rogue AP weaker by default

    # ---------- Jamming realism ----------
    if str(scen).startswith("Jamming") and d_jam <= CFG.jam_radius_m:
        factor = max(0.15, 1.0 - d_jam/CFG.jam_radius_m)
        if jam_mode == "Broadband noise" or jam_mode is None:
            jam_penalty += np.random.uniform(8, 15) * factor
            noise_floor_dbm += np.random.uniform(8, 15) * factor
            cca_busy_frac   = float(np.clip(cca_busy_frac + 0.25*factor + 0.15*np.random.rand(), 0, 0.98))
            phy_error_rate  = float(np.clip(phy_error_rate + 0.15*factor + 0.10*np.random.rand(), 0, 0.9))
            beacon_miss_rate= float(np.clip(beacon_miss_rate + 0.10*factor + 0.10*np.random.rand(), 0, 1.0))
        elif jam_mode == "Reactive":
            util = (0.4+0.5*load)
            jam_penalty += np.random.uniform(4, 9) * factor * util
            noise_floor_dbm += np.random.uniform(4, 9) * factor * util
            cca_busy_frac   = float(np.clip(cca_busy_frac + 0.15*factor*util + 0.05*np.random.rand(), 0, 0.98))
            phy_error_rate  = float(np.clip(phy_error_rate + 0.20*factor*util, 0, 0.9))
            beacon_miss_rate= float(np.clip(beacon_miss_rate + 0.06*factor*util, 0, 1.0))
        else:  # Mgmt (deauth)
            deauth_rate     = float(np.clip(deauth_rate + 0.40*factor + 0.20*np.random.rand(), 0, 1.0))
            assoc_churn     = float(np.clip(assoc_churn + 0.35*factor + 0.15*np.random.rand(), 0, 1.0))
            beacon_miss_rate= float(np.clip(beacon_miss_rate + 0.12*factor, 0, 1.0))
            jam_penalty     += np.random.uniform(0, 3) * factor

    snr = max(0.0, base_snr - jam_penalty)

    # Packet loss as logistic of SNR + congestion
    cong = 0.15*load
    loss = 1/(1+np.exp(0.4*(snr-15))) + cong + np.random.normal(0, 0.02)
    loss = float(np.clip(loss*100, 0, 90))

    # Latency & jitter
    latency = 20 + 2.0*loss + 30*load + np.random.normal(0, 6)
    jitter  = 1.0 + 0.05*loss + 8*load + np.random.normal(0, 1.0)
    latency = float(max(3, latency))
    jitter  = float(max(0.2, jitter))

    # Throughput
    thr = 90 - 0.8*loss - 40*load + np.random.normal(0, 5)
    thr = float(max(1, thr))

    # Channel util
    channel_util = float(np.clip(100*(0.4+0.5*load) + 100*(cca_busy_frac-0.15) + np.random.normal(0, 5), 0, 100))

    # Auth baseline & CRC baseline
    auth_fail = np.random.exponential(0.05)
    crc_err   = int(np.random.poisson(0.2))

    # ---------- Wi-Fi breach realism ----------
    if str(scen).startswith("Wi-Fi Breach") and d_rog <= CFG.breach_radius_m:
        if row.type in MOBILE_TYPES or np.random.rand()<0.3:
            rog_rssi = -40 - 18 * math.log10(max(1.0, d_rog)) + np.random.normal(0, 2)
            rogue_rssi_gap = float(rog_rssi - rssi)
            if breach_mode == "Evil Twin" or breach_mode is None:
                deauth_rate      = float(np.clip(deauth_rate + np.random.uniform(0.25, 0.60), 0, 1.0))
                assoc_churn      = float(np.clip(assoc_churn + np.random.uniform(0.25, 0.50), 0, 1.0))
                eapol_retry_rate = float(np.clip(eapol_retry_rate + np.random.uniform(0.10, 0.30), 0, 1.0))
                auth_fail        += np.random.uniform(0.2, 0.6)
            elif breach_mode == "Rogue Open AP":
                assoc_churn      = float(np.clip(assoc_churn + np.random.uniform(0.15, 0.35), 0, 1.0))
                dhcp_fail_rate   = float(np.clip(dhcp_fail_rate + np.random.uniform(0.25, 0.60), 0, 1.0))
            else:  # Credential hammer
                eapol_retry_rate = float(np.clip(eapol_retry_rate + np.random.uniform(0.35, 0.70), 0, 1.0))
                auth_fail        += np.random.uniform(0.4, 0.9)

    # ---------- GPS Spoofing realism ----------
    pos_error = np.random.normal(2.0, 0.7)
    if str(scen).startswith("GPS Spoofing"):
        # Decide targeting
        hit = False
        if 'spoof_mode' in globals():
            pass  # running in function scope
        spoof_scope = st.session_state.get("spoof_scope")
        # sidebar uses local variables; but we can read from outer if present
        spoof_scope = st.session_state.get("spoof_scope_override", None)
        # fallback: infer from widgets by storing once
        # simpler: re-evaluate using current sidebar inputs if exist
        # We'll directly access from widgets saved earlier:
        try:
            spoof_scope = st.session_state.get("_radio", None)  # not reliable
        except Exception:
            spoof_scope = None
        # Instead, reconstruct from top-level context:
        spoof_scope = st.session_state.get("spoof_scope", None)

        # We can't rely on the above; instead, query Streamlit session directly via known keys we set:
        spoof_mode_local = st.session_state.get("Spoofing scope", None)  # may not exist
        # Given widget keys vary across sessions, use safer logic:
        # We'll recompute based on distance & one-target retention:

        # Single device mode replicated using session cache
        if "spoof_mode_choice" in st.session_state:
            mode_choice = st.session_state["spoof_mode_choice"]
        else:
            mode_choice = None

        # Infer from presence of sidebar radio: we stored earlier while building sidebar
        mode_choice = st.session_state.get("jam_mode", None)  # wrong key; ignore
        # Instead, we pass spoofing behavior via globals in tick_once(); here we just use distance & cached target.
        # Determine effect below using globals set in tick_once()

    # (We will actually handle spoofing targeting in tick_once() and pass booleans there.)

    # ---------- Data Tamper realism ----------
    # Integrity/freshness baseline
    if "seq_counter" not in st.session_state:
        st.session_state.seq_counter = {}
    if row.device_id not in st.session_state.seq_counter:
        st.session_state.seq_counter[row.device_id] = 0
    st.session_state.seq_counter[row.device_id] += 1
    seq_no = st.session_state.seq_counter[row.device_id]
    device_ts = tick + np.random.normal(0, 0.05)
    payload_entropy = float(np.random.normal(5.8, 0.4))
    ts_skew_s = float(device_ts - tick)
    seq_gap = 0.0
    dup_ratio = 0.0
    schema_violation_rate = 0.0
    hmac_fail_rate = 0.0 if crypto_enabled else 0.0

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
            drift = float(np.random.uniform(5, 15))
            thr = max(1.0, thr - drift)
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

    # Compose output
    return dict(
        rssi=float(rssi), snr=float(snr),
        packet_loss=float(loss), latency_ms=float(latency), jitter_ms=float(jitter),
        pos_error_m=float(max(0.3, pos_error)),
        auth_fail_rate=float(max(0, auth_fail)),
        crc_err=int(max(0, crc_err)),
        throughput_mbps=float(thr), channel_util=float(channel_util),
        noise_floor_dbm=float(noise_floor_dbm),
        cca_busy_frac=float(cca_busy_frac),
        phy_error_rate=float(phy_error_rate),
        beacon_miss_rate=float(beacon_miss_rate),
        deauth_rate=float(deauth_rate),
        assoc_churn=float(assoc_churn),
        eapol_retry_rate=float(eapol_retry_rate),
        dhcp_fail_rate=float(dhcp_fail_rate),
        rogue_rssi_gap=float(rogue_rssi_gap),
        payload_entropy=float(np.clip(payload_entropy, 0.0, 8.0)),
        ts_skew_s=float(ts_skew_s),
        seq_gap=float(np.clip(seq_gap, 0.0, 1.0)),
        dup_ratio=float(np.clip(dup_ratio, 0.0, 1.0)),
        schema_violation_rate=float(np.clip(schema_violation_rate, 0.0, 1.0)),
        hmac_fail_rate=float(np.clip(hmac_fail_rate, 0.0, 1.0))
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
# Training (with progress & ETA)
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
    # Ensure seq counters exist for training-time device IDs
    if "seq_counter" not in st.session_state:
        st.session_state.seq_counter = {}
    for dev_id in D["device_id"]:
        st.session_state.seq_counter.setdefault(dev_id, 0)

    buf = {d: deque(maxlen=CFG.rolling_len) for d in D.device_id}
    X_rows=[]; y=[]
    t0 = time.time()
    for t in range(n_ticks):
        scen_code = np.random.choice(["Normal","J","Breach","GPS","Tamper"],
                                     p=[0.55,0.15,0.12,0.10,0.08])
        if   scen_code=="J":     sc="Jamming (localized)";  jm=np.random.choice(["Broadband noise","Reactive","Mgmt (deauth)"]); bm=None
        elif scen_code=="Breach":sc="Wi-Fi Breach (AP)";    bm=np.random.choice(["Evil Twin","Rogue Open AP","Credential hammer"]); jm=None
        elif scen_code=="GPS":   sc="GPS Spoofing (subset)"; jm=bm=None
        elif scen_code=="Tamper":sc="Data Tamper (gateway)"; jm=bm=None
        else:                    sc="Normal"; jm=bm=None

        # move devices
        for i in D.index:
            if D.at[i,"type"] in MOBILE_TYPES:
                D.at[i,"heading"] += np.random.normal(0,0.3)
                step = D.at[i,"speed_mps"]*np.random.uniform(0.6,1.4)
                dn,de = step*math.cos(D.at[i,"heading"]), step*math.sin(D.at[i,"heading"])
                dlat,dlon = meters_to_latlon_offset(dn,de,D.at[i,"lat"])
                D.at[i,"lat"]+=dlat; D.at[i,"lon"]+=dlon

        # anchors for training pass
        st.session_state.ap={"lat":ap_lat,"lon":ap_lon}
        st.session_state.jammer={"lat":jam_lat,"lon":jam_lon}
        st.session_state.rogue={"lat":rog_lat,"lon":rog_lon}
        st.session_state.spoofer={"lat":spf_lat,"lon":spf_lon}

        for _, row in D.iterrows():
            fake_row = type("R",(object,),row.to_dict())()
            metrics = rf_and_network_model(
                fake_row, t, sc,
                tamper_mode=None,
                crypto_enabled=bool(np.random.rand()<0.7),
                training=True,
                jam_mode=(jm if "Jamming" in sc else None),
                breach_mode=(bm if "Wi-Fi Breach" in sc else None)
            )
            buf[row.device_id].append(metrics)
            feats = build_window_features(buf[row.device_id])
            if feats:
                X_rows.append(feats)
                y.append(0 if sc=="Normal" else 1)

        if progress_cb:
            frac = (t+1)/n_ticks
            elapsed = time.time()-t0
            eta = (elapsed/frac)*(1-frac) if frac>0 else None
            pct = pct_start + (pct_end-pct_start)*frac
            progress_cb(int(pct), f"Synthesizing windows {int(frac*100)}%", eta)

    X=pd.DataFrame(X_rows).fillna(0.0); y=np.array(y)
    X_train,X_tmp,y_train,y_tmp = train_test_split(X,y,test_size=0.40,random_state=SEED,shuffle=True,stratify=y)
    X_cal,X_test,y_cal,y_test = train_test_split(X_tmp,y_tmp,test_size=0.50,random_state=SEED,shuffle=True,stratify=y_tmp)
    return (X_train,y_train),(X_cal,y_cal),(X_test,y_test),X

def train_model_with_progress(n_ticks=350):
    bar = st.progress(0, text="Preparing training…")
    note = st.empty()
    t_start = time.time()

    def update(pct, msg, eta=None):
        label = f"{msg} • ETA {fmt_eta(eta)}" if eta is not None else msg
        bar.progress(min(100, int(pct)), text=label)

    # ---- Stage 1: Data synthesis (0..70%)
    (X_train,y_train),(X_cal,y_cal),(X_test,y_test),X_all = make_training_data(
        n_ticks=n_ticks, progress_cb=update, pct_start=0, pct_end=70
    )

    # ---- Stage 2: Scale + Fit (70..95%)
    cols=list(X_train.columns)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train); Xca=scaler.transform(X_cal); Xte=scaler.transform(X_test); Xall=scaler.transform(X_all)
    Xtr_df=to_df(Xtr,cols); Xca_df=to_df(Xca,cols); Xte_df=to_df(Xte,cols); Xall_df=to_df(Xall,cols)

    model = LGBMClassifier(
        n_estimators=CFG.n_estimators,
        max_depth=CFG.max_depth,
        learning_rate=CFG.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=12,   # keep only this (avoid clash with min_data_in_leaf)
        force_col_wise=True,
        random_state=SEED
    )

    iter0 = time.time()
    total_iters = CFG.n_estimators
    def lgbm_progress_callback(env):
        it = getattr(env, "iteration", 0)
        it = max(0, min(total_iters, it))
        frac = it/total_iters if total_iters>0 else 1.0
        elapsed = time.time() - iter0
        eta = (elapsed/frac)*(1-frac) if frac>0 else None
        pct = 70 + 25*frac
        update(pct, f"Training trees {it}/{total_iters}", eta)

    update(70, "Starting model training…")
    model.fit(Xtr_df, y_train, callbacks=[lgb.log_evaluation(period=0), lgbm_progress_callback])

    # ---- Stage 3: Calibration + Metrics (95..100%)
    update(95, "Calibrating confidence…")
    cal_p = model.predict_proba(Xca_df)[:,1]
    cal_nc = 1 - np.where(y_cal==1, cal_p, 1-cal_p)

    update(97, "Evaluating metrics…")
    te_p = model.predict_proba(Xte_df)[:,1]
    preds=(te_p>=CFG.threshold).astype(int)
    prec,rec,f1,_ = precision_recall_fscore_support(y_test,preds,average="binary",zero_division=0)
    auc = roc_auc_score(y_test, te_p)

    ths = np.linspace(0.30, 0.90, 61)
    best_f1, best_th = 0.0, CFG.threshold
    for th in ths:
        pred = (te_p >= th).astype(int)
        _, _, f1c, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        if f1c > best_f1:
            best_f1, best_th = f1c, th
    st.session_state.suggested_threshold = float(best_th)

    st.session_state.model=model
    st.session_state.scaler=scaler
    st.session_state.explainer=shap.TreeExplainer(model)
    st.session_state.conformal_scores=cal_nc
    st.session_state.metrics={"precision":prec,"recall":rec,"f1":f1,"auc":auc}
    st.session_state.baseline = Xall_df

    total_secs = time.time()-t_start
    st.session_state.last_train_secs = total_secs
    bar.progress(100, text=f"Training complete • {int(total_secs)}s")
    note.success(
        f"Model trained: AUC={auc:.2f}, Precision={prec:.2f}, Recall={rec:.2f} • "
        f"Duration {int(total_secs)}s • Suggested threshold={best_th:.2f}"
    )

if (CFG.retrain_on_start and st.session_state.get("model") is None) or retrain:
    train_model_with_progress(n_ticks=350)

def conformal_pvalue(prob):
    cal=st.session_state.conformal_scores
    if cal is None: return None
    nc=1-prob
    return float((np.sum(cal>=nc)+1)/(len(cal)+1))

# =========================
# Streaming tick
# =========================
def tick_once():
    st.session_state.devices = update_positions(st.session_state.devices.copy())
    tick = st.session_state.tick

    # Keep spoof targeting cached
    if "spoof_target_id" not in st.session_state:
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

        # Apply GPS spoofing scope here (single/local/site-wide)
        if scenario.startswith("GPS Spoofing"):
            hit = False
            spf = st.session_state.spoofer
            d_spf = haversine_m(row.lat, row.lon, spf["lat"], spf["lon"])
            try:
                current_scope = st.session_state["Spoofing scope"]  # may not exist
            except KeyError:
                current_scope = None
            # Read from radio value if present in widget state
            # Fallback to heuristics: assume Localized by default
            selected = current_scope if isinstance(current_scope, str) else "Localized area"
            if selected == "Site-wide":
                hit = True
            elif selected == "Single device":
                hit = (row.device_id == st.session_state.spoof_target_id)
            else:  # Localized area
                hit = (d_spf <= CFG.spoof_radius_m)

            if hit and (row.type in MOBILE_TYPES or st.session_state.get("spoof_mobile_only", True) is False):
                minutes = tick / 60.0
                bias = 20.0
                drift = 6.0 * minutes
                scale = 1.0
                if selected == "Localized area":
                    scale = max(0.15, 1.0 - d_spf/CFG.spoof_radius_m)
                m["pos_error_m"] += scale * (np.random.uniform(0.6*bias, 1.2*bias) + 0.5*drift)

        st.session_state.dev_buf[row.device_id].append(m)
        rec = {"tick":tick, "device_id":row.device_id, "type":row.type, "lat":row.lat, "lon":row.lon, **m}
        fleet_rows.append(rec)

    # Inference + incidents
    incidents_this_tick=[]
    for _, row in st.session_state.devices.iterrows():
        feats = build_window_features(st.session_state.dev_buf[row.device_id])
        if not feats: continue
        st.session_state.last_features[row.device_id] = feats
        X = pd.DataFrame([feats]).fillna(0.0)
        Xs = st.session_state.scaler.transform(X)
        Xs_df = to_df(Xs, X.columns)
        prob = float(st.session_state.model.predict_proba(Xs_df)[:,1][0])
        pval = conformal_pvalue(prob) if use_conformal else None
        sev,_ = severity(prob,pval)
        if prob>=CFG.threshold:
            shap_mat = shap_pos(st.session_state.explainer, Xs_df)
            shap_vec = shap_mat[0]
            pairs = sorted(list(zip(X.columns, shap_vec)), key=lambda x: abs(x[1]), reverse=True)[:6]
            inc = dict(
                ts=int(time.time()), tick=tick, device_id=row.device_id, type=row.type,
                lat=row.lat, lon=row.lon, scenario=scenario,
                prob=prob, p_value=pval, severity=sev, reasons=[{"feature":k,"impact":float(v)} for k,v in pairs]
            )
            st.session_state.incidents.append(inc)
            incidents_this_tick.append(inc)

    if incidents_this_tick:
        affected = {i["device_id"] for i in incidents_this_tick}
        ratio = len(affected)/len(st.session_state.devices)
        if ratio>=0.25:
            st.session_state.group_incidents.append({
                "ts": int(time.time()),
                "tick": tick,
                "scenario": scenario,
                "affected": len(affected),
                "fleet": len(st.session_state.devices),
                "ratio": ratio,
            })

    st.session_state.fleet_records.extend(fleet_rows)
    st.session_state.tick += 1

# Drive ticks
if st.session_state.get("model") is not None:
    if auto:
        for _ in range(speed): tick_once()
    else:
        if st.button("Step once"): tick_once()

# =========================
# KPIs
# =========================
k1,k2,k3,k4,k5 = st.columns(5)
with k1: st.metric("Devices", len(st.session_state.devices))
with k2: st.metric("Incidents (session)", len(st.session_state.incidents))
with k3:
    auc = (st.session_state.metrics or {}).get("auc", 0)
    st.metric("Model AUC", f"{auc:.2f}")
with k4:
    probs=[]
    for _, feats in st.session_state.last_features.items():
        X=pd.DataFrame([feats]); Xs=st.session_state.scaler.transform(X); Xs_df=to_df(Xs,X.columns)
        probs.append(float(st.session_state.model.predict_proba(Xs_df)[:,1][0]))
    st.metric("Fleet risk (mean prob)", f"{(np.mean(probs) if probs else 0):.2f}")
with k5:
    train_secs = st.session_state.get("last_train_secs")
    st.metric("Last train duration", f"{int(train_secs)}s" if train_secs else "—")

# =========================
# Layout tabs
# =========================
tab_overview, tab_fleet, tab_incidents, tab_insights = st.tabs(["Overview", "Fleet View", "Incidents", "Insights"])

# ---------- Overview
with tab_overview:
    left, right = st.columns([2,1])

    with left:
        if show_map and st.session_state.get("model") is not None:
            df_map = st.session_state.devices.copy()
            if type_filter and len(type_filter) < len(DEVICE_TYPES):
                df_map = df_map[df_map["type"].isin(type_filter)].copy()

            risks, snrs, losses = [], [], []
            for _, r in df_map.iterrows():
                feats = st.session_state.last_features.get(r.device_id, {})
                if feats:
                    X = pd.DataFrame([feats]); Xs = st.session_state.scaler.transform(X); Xs_df = to_df(Xs, X.columns)
                    risks.append(float(st.session_state.model.predict_proba(Xs_df)[:, 1][0]))
                else:
                    risks.append(0.0)
                buf = st.session_state.dev_buf.get(r.device_id, [])
                if buf and len(buf) > 0:
                    snrs.append(float(buf[-1].get("snr", np.nan)))
                    losses.append(float(buf[-1].get("packet_loss", np.nan)))
                else:
                    snrs.append(np.nan); losses.append(np.nan)

            df_map["risk"] = risks
            df_map["snr"] = snrs
            df_map["packet_loss"] = losses

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

            # AP marker
            ap_df = pd.DataFrame([{"lat": st.session_state.ap["lat"], "lon": st.session_state.ap["lon"], "label": "AP"}])
            layers += [
                pdk.Layer("ScatterplotLayer", data=ap_df, get_position='[lon, lat]',
                          get_fill_color='[30,144,255,240]', get_radius=10),
                pdk.Layer("TextLayer", data=ap_df, get_position='[lon, lat]', get_text='label',
                          get_color=[30,144,255,255], get_size=14, get_alignment_baseline="'bottom'", get_pixel_offset=[0,-10]),
            ]

            # Jammer overlay
            if scenario.startswith("Jamming"):
                jam = st.session_state.jammer
                jam_df = pd.DataFrame([{"lat": jam["lat"], "lon": jam["lon"], "label": "Jammer"}])
                layers += [
                    pdk.Layer("ScatterplotLayer", data=jam_df, get_position='[lon, lat]',
                              get_fill_color='[255,0,0,240]', get_radius=10),
                    pdk.Layer("TextLayer", data=jam_df, get_position='[lon, lat]', get_text='label',
                              get_color=[255,0,0,255], get_size=14, get_alignment_baseline="'bottom'", get_pixel_offset=[0,-10]),
                ]
                angles = np.linspace(0, 2*np.pi, 60)
                circle = [{"path": [[
                    jam["lon"] + meters_to_latlon_offset(CFG.jam_radius_m * math.sin(a),
                                                         CFG.jam_radius_m * math.cos(a),
                                                         st.session_state.devices.lat.mean())[1],
                    jam["lat"] + meters_to_latlon_offset(CFG.jam_radius_m * math.sin(a),
                                                         CFG.jam_radius_m * math.cos(a),
                                                         st.session_state.devices.lat.mean())[0]
                ] for a in angles]}]
                layers.append(pdk.Layer("PathLayer", circle, get_path="path", get_color=[255,0,0], width_scale=4, width_min_pixels=1, opacity=0.25))

            # Rogue AP overlay
            if scenario.startswith("Wi-Fi Breach"):
                rog = st.session_state.rogue
                rog_df = pd.DataFrame([{"lat": rog["lat"], "lon": rog["lon"], "label": "Rogue AP"}])
                layers += [
                    pdk.Layer("ScatterplotLayer", data=rog_df, get_position='[lon, lat]',
                              get_fill_color='[0,255,255,240]', get_radius=10),
                    pdk.Layer("TextLayer", data=rog_df, get_position='[lon, lat]', get_text='label',
                              get_color=[0,200,200,255], get_size=14, get_alignment_baseline="'bottom'", get_pixel_offset=[0,-10]),
                ]
                angles = np.linspace(0, 2*np.pi, 60)
                circle = [{"path": [[
                    rog["lon"] + meters_to_latlon_offset(CFG.breach_radius_m * math.sin(a),
                                                         CFG.breach_radius_m * math.cos(a),
                                                         st.session_state.devices.lat.mean())[1],
                    rog["lat"] + meters_to_latlon_offset(CFG.breach_radius_m * math.sin(a),
                                                         CFG.breach_radius_m * math.cos(a),
                                                         st.session_state.devices.lat.mean())[0]
                ] for a in angles]}]
                layers.append(pdk.Layer("PathLayer", circle, get_path="path", get_color=[0,200,200], width_scale=4, width_min_pixels=1, opacity=0.25))

            # Spoofer overlay
            if scenario.startswith("GPS Spoofing"):
                spf = st.session_state.spoofer
                spf_df = pd.DataFrame([{"lat": spf["lat"], "lon": spf["lon"], "label": "Spoofer"}])
                layers += [
                    pdk.Layer("ScatterplotLayer", data=spf_df, get_position='[lon, lat]',
                              get_fill_color='[255,215,0,240]', get_radius=10),
                    pdk.Layer("TextLayer", data=spf_df, get_position='[lon, lat]', get_text='label',
                              get_color=[255,215,0,255], get_size=14, get_alignment_baseline="'bottom'", get_pixel_offset=[0,-10]),
                ]
                angles = np.linspace(0, 2*np.pi, 60)
                circle = [{"path": [[
                    spf["lon"] + meters_to_latlon_offset(CFG.spoof_radius_m * math.sin(a),
                                                         CFG.spoof_radius_m * math.cos(a),
                                                         st.session_state.devices.lat.mean())[1],
                    spf["lat"] + meters_to_latlon_offset(CFG.spoof_radius_m * math.sin(a),
                                                         CFG.spoof_radius_m * math.cos(a),
                                                         st.session_state.devices.lat.mean())[0]
                ] for a in angles]}]
                layers.append(pdk.Layer("PathLayer", circle, get_path="path", get_color=[255,215,0], width_scale=4, width_min_pixels=1, opacity=0.25))

            view_state = pdk.ViewState(
                latitude=float(st.session_state.devices.lat.mean()),
                longitude=float(st.session_state.devices.lon.mean()),
                zoom=14, pitch=0
            )
            tooltip = {"html": "<b>{device_id}</b> • {type}<br/>Risk: {risk}<br/>SNR: {snr} dB<br/>Loss: {packet_loss}%",
                       "style": {"backgroundColor": "rgba(255,255,255,0.95)", "color": "#111"}}
            st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None, tooltip=tooltip),
                            use_container_width=True)

            # Legend
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
                  <span style="margin-left:auto;opacity:.8;">Dot size = risk</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        fr = pd.DataFrame(list(st.session_state.fleet_records))
        if len(fr)>0:
            for y in ["snr","packet_loss","latency_ms","pos_error_m"]:
                sub = fr.groupby("tick")[y].mean().reset_index()
                fig=px.line(sub, x="tick", y=y, title=f"Fleet avg {y}")
                st.plotly_chart(fig, width='stretch')

    with right:
        leaderboard=[]
        if st.session_state.get("model") is not None:
            for _, r in st.session_state.devices.iterrows():
                feats=st.session_state.last_features.get(r.device_id)
                if not feats: continue
                X=pd.DataFrame([feats]); Xs=st.session_state.scaler.transform(X); Xs_df=to_df(Xs,X.columns)
                prob=float(st.session_state.model.predict_proba(Xs_df)[:,1][0])
                pval=conformal_pvalue(prob) if use_conformal else None
                sev,_=severity(prob,pval)
                leaderboard.append({"device_id":r.device_id,"type":r.type,"prob":prob,"p_value":pval,"severity":sev})
        if leaderboard:
            df_lead=pd.DataFrame(leaderboard).sort_values("prob",ascending=False).head(10)
            st.markdown("### Top risk devices")
            st.dataframe(df_lead, use_container_width=True)

        st.markdown("### Device Inspector (local SHAP)")
        if role in {"Domain Expert", "AI Builder", "Executive"} and st.session_state.get("model") is not None:
            have_feats = [d for d in st.session_state.devices["device_id"].tolist()
                          if d in st.session_state.last_features]
            options = have_feats if have_feats else st.session_state.devices["device_id"].tolist()
            sel = st.selectbox("Select device", options, key="device_inspect_select")

            feats = st.session_state.last_features.get(sel)
            if feats:
                X = pd.DataFrame([feats]).fillna(0.0)
                Xs = st.session_state.scaler.transform(X)
                Xs_df = to_df(Xs, X.columns)
                prob = float(st.session_state.model.predict_proba(Xs_df)[:, 1][0])
                pval = conformal_pvalue(prob) if use_conformal else None

                c1, c2 = st.columns(2)
                with c1: st.metric("Current risk (prob)", f"{prob:.2f}")
                with c2: st.metric("Conf. p-value", f"{pval:.3f}" if pval is not None else "—")

                shap_vec = shap_pos(st.session_state.explainer, Xs_df)[0]
                topk = 8
                pairs = sorted(zip(X.columns, shap_vec), key=lambda kv: abs(kv[1]), reverse=True)[:topk]
                df_shap = pd.DataFrame(pairs, columns=["feature", "impact"])
                fig = px.bar(df_shap.sort_values("impact"), x="impact", y="feature", orientation="h",
                             title=f"Local SHAP — {sel}", labels={"impact": "contribution → anomaly"})
                st.plotly_chart(fig, width='stretch')

                with st.expander("Raw window features (standardized input)"):
                    st.dataframe(Xs_df.T.rename(columns={0: "z-value"}), use_container_width=True)
            else:
                st.info("Waiting for enough samples to build a feature window for this device.")
        else:
            st.caption("Local SHAP hidden for this role. Switch to Domain Expert / AI Builder / Executive to view.")

# ---------- Fleet View
with tab_fleet:
    fr = pd.DataFrame(list(st.session_state.fleet_records))
    if len(fr)>0:
        recent = fr[fr["tick"]>=st.session_state.tick-40]
        cols = ["snr","packet_loss","latency_ms","jitter_ms","pos_error_m","auth_fail_rate","crc_err","throughput_mbps","channel_util",
                "noise_floor_dbm","cca_busy_frac","phy_error_rate","deauth_rate","assoc_churn","eapol_retry_rate","dhcp_fail_rate"]
        cols = [c for c in cols if c in recent.columns]
        mat = recent.groupby("device_id")[cols].mean()
        z = (mat-mat.mean())/mat.std(ddof=0).replace(0,1)
        fig = px.imshow(z.T, color_continuous_scale="RdBu_r", aspect="auto",
                        labels=dict(color="z-score"), title="Fleet heatmap (recent mean z-scores)")
        st.plotly_chart(fig, width='stretch')
    st.dataframe(st.session_state.devices, use_container_width=True)

# ---------- Incidents
with tab_incidents:
    st.subheader("Incidents")
    if len(st.session_state.incidents)==0:
        st.success("No incidents yet.")
    else:
        for i, inc in enumerate(reversed(st.session_state.incidents[-15:]), 1):
            pv = inc.get("p_value")
            pv_str = f"{pv:.3f}" if pv is not None else "—"
            sev, color = inc["severity"], {"High":"red","Medium":"orange","Low":"green"}.get(inc["severity"],"gray")
            title = f"#{len(st.session_state.incidents)-i+1} • {inc['scenario']} • {inc['device_id']} ({inc['type']}) • prob={inc['prob']:.2f} • p={pv_str} • {sev}"
            with st.expander(title):
                badge=f"<span style='background-color:{color}; color:white; padding:2px 8px; border-radius:8px;'>{sev}</span>"
                st.markdown(f"**Severity**: {badge}", unsafe_allow_html=True)
                left,right = st.columns([2,1])

                with left:
                    tabs = st.tabs(["End Users","Domain Experts","Regulators","AI Builders","Executives"])

                    with tabs[0]:
                        st.markdown("#### What happened (simple)")
                        st.write(f"Device **{inc['device_id']}** behaved like **{inc['scenario']}**.")
                        st.markdown("#### What it means")
                        if "Jamming" in inc["scenario"]:
                            st.write("Temporary slow or unstable connection nearby.")
                        elif "GPS Spoofing" in inc["scenario"]:
                            st.write("Location may be inaccurate.")
                        elif "Wi-Fi Breach" in inc["scenario"]:
                            st.write("Unknown/unsafe Wi-Fi nearby disrupted connectivity.")
                        elif "Data Tamper" in inc["scenario"]:
                            st.write("Some readings looked reused or out of range.")
                        st.markdown("#### What to do now")
                        if "Jamming" in inc["scenario"]:
                            st.write("Move 50–100 m away or retry; ops will switch channel.")
                        elif "GPS Spoofing" in inc["scenario"]:
                            st.write("Confirm location via UWB/IMU fusion; slow down in GNSS-denied mode.")
                        elif "Wi-Fi Breach" in inc["scenario"]:
                            st.write("Avoid joining unknown SSIDs; network is enforcing protections.")
                        else:
                            st.write("Retry; if it persists, escalate to ops.")

                    with tabs[1]:
                        st.markdown("#### Operational assessment")
                        st.write(f"- Severity **{sev}** (prob={inc['prob']:.2f}, p={pv_str})")
                        obs=" • ".join([f"{r['feature']}({r['impact']:+.3f})" for r in inc["reasons"]])
                        st.write(f"- Key observables: {obs}")
                        st.markdown("#### Signals")
                        if "Jamming" in inc["scenario"]:
                            st.write("↑ noise_floor_dbm, ↑ cca_busy_frac, ↑ phy_error_rate, ↑ beacon_miss_rate, ↓ snr")
                        elif "Wi-Fi Breach" in inc["scenario"]:
                            st.write("↑ deauth_rate, ↑ assoc_churn, ↑ eapol_retry_rate / dhcp_fail_rate, rogue_rssi_gap>0")
                        elif "Data Tamper" in inc["scenario"]:
                            st.write("↑ dup_ratio, ↑ ts_skew_s, ↑ schema_violation_rate, ↑ hmac_fail_rate (if enabled)")
                        elif "GPS Spoofing" in inc["scenario"]:
                            st.write("↑ pos_error_m with smooth bias/drift")
                        st.markdown("#### Playbook")
                        if "Jamming" in inc["scenario"]:
                            st.write("Channel hop ▸ Spectrum scan ▸ Directional/backup link.")
                        elif "Wi-Fi Breach" in inc["scenario"]:
                            st.write("Quarantine SSID ▸ Rotate credentials ▸ Rogue AP sweep.")
                        elif "Data Tamper" in inc["scenario"]:
                            st.write("Reject stale/invalid payloads ▸ Verify signatures ▸ Audit gateway.")
                        else:
                            st.write("Switch to GNSS-denied nav ▸ Verify time source ▸ Geofence sanity.")

                    with tabs[2]:
                        st.markdown("#### Assurance & governance")
                        st.write(f"- Calibrated confidence: p-value = {pv_str} (target coverage {int(CFG.coverage*100)}%).")
                        st.write("- Audit trail: downloadable incident evidence with model version & inputs.")
                        st.write("- No personal data; technical telemetry only.")
                        evidence = {
                            "ts": inc["ts"], "tick": inc["tick"], "device_id": inc["device_id"], "type": inc["type"],
                            "lat": inc["lat"], "lon": inc["lon"], "scenario": inc["scenario"], "severity": inc["severity"],
                            "prob": inc["prob"], "p_value": inc["p_value"], "model_version": "LightGBM v2.3-demo",
                            "explanations": inc["reasons"]
                        }
                        st.download_button(
                            "Download incident evidence (JSON)",
                            data=json.dumps(evidence, indent=2).encode("utf-8"),
                            file_name=f"incident_{inc['device_id']}_{inc['ts']}.json",
                            mime="application/json",
                            key=f"dl_evidence_{inc['device_id']}_{inc['ts']}"
                        )

                    with tabs[3]:
                        st.markdown("#### Local technical explanation (SHAP)")
                        txt="\n".join([f"- {r['feature']}: {r['impact']:+.3f}" for r in inc["reasons"]])
                        st.markdown(txt)
                        st.caption("Positive impact pushes toward 'anomaly' in standardized feature space.")
                        st.markdown("#### Implementation")
                        st.write("- LightGBM (depth≤3, ~60 trees) on rolling-window features.")
                        st.write("- Conformal calibration for p-values.")
                        st.write("- Realistic RF & Wi-Fi signals; integrity/freshness fields for tamper.")

                    with tabs[4]:
                        st.markdown("#### Executive summary")
                        st.write(f"- Device {inc['device_id']} at risk: **{sev}**; scenario: **{inc['scenario']}**.")
                        st.write("- Impact: transient performance/security issues; mitigations in place.")
                        st.markdown("#### KPIs")
                        st.write("- Incidents, MTTD, Packet loss, Latency, SNR, Deauth rate")

                with right:
                    st.json({k:inc[k] for k in ["prob","p_value","tick","device_id","type","scenario"]})

    if st.session_state.incidents:
        df_inc = pd.DataFrame(st.session_state.incidents)
        df_inc["top_features"] = df_inc["reasons"].apply(lambda r: "; ".join([f"{x['feature']}:{x['impact']:+.3f}" for x in r]))
        csv = df_inc.drop(columns=["reasons"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download incidents CSV", csv, "incidents.csv", "text/csv", key="dl_incidents_csv")

# ---------- Insights
with tab_insights:
    g1,g2 = st.columns(2)
    with g1:
        st.markdown("### Global importance (mean |SHAP|)")
        base = st.session_state.get("baseline")
        if base is not None and len(base)>0:
            shap_mat = shap_pos(st.session_state.explainer, base)
            mean_abs = np.abs(shap_mat).mean(axis=0)
            imp = pd.DataFrame({"feature": base.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False).head(18)
            fig = px.bar(imp, x="mean_abs_shap", y="feature", orientation="h", title="Global feature impact")
            st.plotly_chart(fig, width='stretch')
    with g2:
        fr = pd.DataFrame(list(st.session_state.fleet_records))
        if len(fr)>0:
            a = fr[fr["tick"]<max(1, st.session_state.tick*0.3)]
            b = fr[fr["tick"]>max(0, st.session_state.tick-200)]
            cols = ["snr","packet_loss","latency_ms","jitter_ms","pos_error_m","auth_fail_rate","crc_err","throughput_mbps","channel_util",
                    "noise_floor_dbm","cca_busy_frac","phy_error_rate","deauth_rate","assoc_churn","eapol_retry_rate","dhcp_fail_rate"]
            drift = []
            for c in cols:
                if c not in fr.columns: continue
                mu_a, mu_b = a[c].mean(), b[c].mean()
                sd = a[c].std(ddof=0)+1e-6
                z = (mu_b-mu_a)/sd
                drift.append({"metric":c, "zshift":z})
            if drift:
                df_drift = pd.DataFrame(drift).sort_values("zshift", ascending=False)
                fig = px.bar(df_drift, x="zshift", y="metric", orientation="h", title="Drift (mean shift vs early window)")
                st.plotly_chart(fig, width='stretch')

    st.markdown("### Model card")
    st.json({
        "model": "LightGBM (depth≤3, ~60 trees)",
        "features_per_device": len(feature_cols()),
        "training_metrics": st.session_state.metrics,
        "calibration": f"Conformal p-value (target coverage {int(CFG.coverage*100)}%)",
        "realism": "Distance-based RF, jamming modes, rogue AP behaviors, GNSS spoof scopes, integrity/freshness for tamper",
        "intended_use": "Demonstration of trustworthy anomaly detection in wireless/logistics (Sundsvall/MSU)",
        "limitations": [
            "Synthetic but physics-inspired; not production-ready",
            "Single-site example without true multi-AP roaming"
        ],
        "version": "v2.3-sundsvall"
    })
