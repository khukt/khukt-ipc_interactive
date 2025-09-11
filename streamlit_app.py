# streamlit_app.py
# TRUST AI — Industrial Wireless Security Demo (Sundsvall / Mid Sweden University)
# Two-step ML (Anomaly + Attack Type), Explainable for all user roles
# ---------------------------------------------------------------
import os, math, json, time, random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pydeck as pdk

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier

# Optional XAI
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ---------- Page ----------
st.set_page_config(
    page_title="TRUST AI — Industrial Wireless Security",
    page_icon="🛰️",
    layout="wide"
)

# ----------------------- Configuration -----------------------
@dataclass
class Config:
    n_devices: int = 28
    frac_types: Dict[str, float] = None
    threshold: float = 0.6
    map_center: Tuple[float,float] = (62.3919, 17.3065)  # Sundsvall – Mid Sweden University
    map_bbox: Tuple[float,float,float,float] = (62.388, 62.397, 17.298, 17.317)  # keep on land
    jammer_radius_m: float = 140.0
    spoof_radius_m: float = 220.0
    breach_radius_m: float = 120.0
    tick_ms: int = 1000
    retrain_on_start: bool = False
    rng_seed: int = 1337
    help_mode: bool = True

CFG = Config(frac_types={"AMR": 0.32, "Truck": 0.32, "Sensor": 0.22, "Gateway": 0.14})
random.seed(CFG.rng_seed)
np.random.seed(CFG.rng_seed)

# ----------------------- UI / UX helpers -----------------------
COLOR = {
    "primary":  "#2563EB",
    "muted":    "#64748B",
    "ok":       "#16A34A",
    "warn":     "#F59E0B",
    "danger":   "#DC2626",
    "panel":    "rgba(0,0,0,.03)",
    "badgebg":  "rgba(0,0,0,.06)",
}
ICONS = {
    "app": "🛰️", "map": "🗺️", "insight":"📈", "inc":"⚠️", "gov":"🛡️",
    "AMR":"🤖", "Truck":"🚚", "Sensor":"📟", "Gateway":"📡",
    "Jamming":"📡", "Access Breach":"🔓", "GPS Spoofing":"🧭", "Data Tamper":"🧪",
    "Normal":"✅"
}

def inject_ux(density: str = "Comfortable"):
    pad_top  = "1.6rem" if density=="Comfortable" else "1.0rem"
    pad_bot  = "2.2rem" if density=="Comfortable" else "1.2rem"
    panel_pad= "16px 18px" if density=="Comfortable" else "12px 14px"
    panel_m  = "10px 0 14px 0" if density=="Comfortable" else "6px 0 10px 0"
    st.markdown(f"""
    <style>
      .block-container {{ padding-top: {pad_top}; padding-bottom: {pad_bot}; }}
      .panel {{
        background: {COLOR["panel"]};
        border: 1px solid rgba(0,0,0,.06);
        border-radius: 14px; padding: {panel_pad}; margin: {panel_m};
      }}
      .badge {{ display:inline-flex; align-items:center; gap:.45rem;
               padding:.20rem .60rem; border-radius: 999px; font-size:.90rem;
               background:{COLOR["badgebg"]}; }}
      .badge .dot {{ width:.60rem; height:.60rem; border-radius:999px; display:inline-block; }}
      .dot.ok {{ background:{COLOR["ok"]}; }}
      .dot.warn {{ background:{COLOR["warn"]}; }}
      .dot.danger {{ background:{COLOR["danger"]}; }}
      .subtle {{ color:{COLOR["muted"]}; }}
      .stAlert div[data-baseweb="notification"] {{ border-radius: 12px; }}
    </style>
    """, unsafe_allow_html=True)

def severity_badge(sev:str)->str:
    cls = "ok" if sev=="Low" else ("warn" if sev=="Medium" else "danger")
    return f'<span class="badge"><span class="dot {cls}"></span>{sev}</span>'

def role_chip(role:str)->str:
    return f'<span class="badge">{role}</span>'

def attack_chip(label:str)->str:
    b = (label or "Unknown").replace(" (low conf)","").replace(" (rules)","")
    icon = ICONS.get(b, "⚠️")
    return f'<span class="badge">{icon}&nbsp;{b}</span>'

def section(title:str, icon:str=""):
    st.markdown(f"### {icon} {title}")

# ---------- Units / axis metadata for common metrics ----------
AXES_META = {
    "snr":            {"y_title": "SNR (dB)",                "hover_suffix": " dB"},
    "packet_loss":    {"y_title": "Packet loss (%)",         "hover_suffix": " %"},
    "latency_ms":     {"y_title": "Latency (ms)",            "hover_suffix": " ms"},
    "pos_error_m":    {"y_title": "GNSS position error (m)", "hover_suffix": " m"},
    "jitter_ms":      {"y_title": "Jitter (ms)",             "hover_suffix": " ms"},
    "noise_floor_dbm":{"y_title": "Noise floor (dBm)",       "hover_suffix": " dBm"},
}
def small_line_chart(df, x, y, title, height=240, key=None):
    meta = AXES_META.get(y, {"y_title": y, "hover_suffix": ""})
    fig = px.line(df, x=x, y=y, title=title)
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=38, b=8), showlegend=False)
    fig.update_xaxes(title_text="Tick")
    fig.update_yaxes(title_text=meta["y_title"])
    fig.update_traces(hovertemplate=f"%{{x}} → %{{y:.2f}}{meta['hover_suffix']}")
    st.plotly_chart(fig, use_container_width=True, key=key)

# ----------------------- Attack Glossary -----------------------
ATTACK_GLOSSARY = {
    "Jamming": {
        "name": "Signal Jamming",
        "what": "Someone is blasting radio noise so devices can’t ‘hear’ the base station.",
        "symptoms": "Signal quality drops, lots of retries, everything feels laggy or offline.",
        "impact": "Slower or unstable connections in the area of the jammer.",
        "action": "If mobile, move 50–100 m; try again in 1–2 minutes. The network may auto-change channel.",
        "analogy": "Like trying to talk in a room where someone turned up a loud static radio."
    },
    "Access Breach": {
        "name": "Access Breach (Fake AP/cell or misuse)",
        "what": "A fake Wi-Fi/5G cell or bad configuration tries to trick devices to join or fail auth.",
        "symptoms": "Frequent disconnects, ‘can’t connect’, unusual login or DHCP failures.",
        "impact": "Device might join the wrong network or fail to connect at all.",
        "action": "Use only known site networks; ignore unfamiliar SSIDs/cells. Security is monitoring.",
        "analogy": "Like a fake front desk asking for your badge and then sending you to the wrong room."
    },
    "GPS Spoofing": {
        "name": "GPS/GNSS Spoofing",
        "what": "A transmitter sends fake satellite signals so the device thinks it’s elsewhere.",
        "symptoms": "Map position jumps, heading/speed look wrong, satellite count/quality looks odd.",
        "impact": "Navigation/location tasks may be unsafe if trusted blindly.",
        "action": "Rely on non-GPS fallback (odometry/UWB/IMU), slow down until location stabilizes.",
        "analogy": "Like someone swapping the signs on the road so your map app gets confused."
    },
    "Data Tamper": {
        "name": "Data Tampering",
        "what": "Reported readings are replayed, duplicated, or altered in transit.",
        "symptoms": "Repeat values, strange timestamps, schema/HMAC/CRC failures.",
        "impact": "Dashboards show untrustworthy numbers; decisions may be wrong.",
        "action": "System quarantines suspicious data; check gateway integrity/config and keys.",
        "analogy": "Like someone editing a shipment form before it reaches the office."
    }
}
def explain_attack_plain(label: str):
    if not label: return None
    base = label.replace(" (low conf)", "").replace(" (rules)", "")
    for k in ATTACK_GLOSSARY.keys():
        if base.startswith(k) or base == k:
            return ATTACK_GLOSSARY[k]
    return None

# ----------------------- State & Utilities -----------------------
def json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.ndarray,)): return o.tolist()
    return str(o)

def init_state():
    st.session_state.setdefault("ui_nonce", int(time.time()))
    st.session_state.setdefault("help_mode", CFG.help_mode)
    st.session_state.setdefault("show_quickstart", True)
    st.session_state.setdefault("first_run_toast", False)
    st.session_state.setdefault("devices", build_devices(CFG.n_devices))
    st.session_state.setdefault("seq_counter", {d["device_id"]:0 for d in st.session_state["devices"]})
    st.session_state.setdefault("fleet_records", [])
    st.session_state.setdefault("incidents", [])
    st.session_state.setdefault("incident_labels", {})
    st.session_state.setdefault("metrics", {})
    st.session_state.setdefault("conformal_scores", None)
    st.session_state.setdefault("model", None)
    st.session_state.setdefault("type_model", None)
    st.session_state.setdefault("feature_names", None)
    st.session_state.setdefault("type_names", ["Normal","Jamming","Access Breach","GPS Spoofing","Data Tamper"])
    st.session_state.setdefault("tick", 0)

def is_model_ready():
    return st.session_state.get("model") is not None

def latlon_in_bbox(lat, lon, bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    lat = min(max(lat, lat_min), lat_max)
    lon = min(max(lon, lon_min), lon_max)
    return lat, lon

def build_devices(n):
    types = []
    for t, frac in CFG.frac_types.items():
        types += [t]*int(round(frac*n))
    while len(types) < n: types.append("Sensor")
    random.shuffle(types)
    lat_min, lat_max, lon_min, lon_max = CFG.map_bbox
    devices=[]
    for i in range(n):
        lat = np.random.uniform(lat_min, lat_max)
        lon = np.random.uniform(lon_min, lon_max)
        devices.append({
            "device_id": f"D{str(i+1).zfill(3)}",
            "type": types[i],
            "lat": float(lat), "lon": float(lon),
        })
    return devices

def move_device(dev):
    step = 0.00015 if dev["type"] in ("AMR","Truck") else 0.00005
    dev["lat"] += np.random.normal(0, step)
    dev["lon"] += np.random.normal(0, step)
    dev["lat"], dev["lon"] = latlon_in_bbox(dev["lat"], dev["lon"], CFG.map_bbox)

# ----------------------- Scenario / Attacks -----------------------
SCENARIOS = ["Normal", "Jamming", "Access Breach", "GPS Spoofing", "Data Tamper"]
def meters_to_latlon_offset(m):
    return (m/111000.0, m/55000.0)  # ~at 62°N

def simulate_tick(dev, t, scenario, centers):
    # Baseline (normal)
    snr = float(np.random.normal(22, 4))           # dB
    sinr = float(np.random.normal(18, 5))          # dB
    noise = float(np.random.normal(-92, 3))        # dBm
    loss = float(np.clip(np.random.normal(1.5, 1.2), 0, 40))  # %
    latency = float(np.clip(np.random.normal(28, 10), 5, 250))# ms
    jitter = float(np.clip(np.random.normal(4, 2), 0.1, 60))  # ms
    # GNSS
    pos_err = float(np.clip(np.random.normal(2.5 if dev["type"] in ("AMR","Truck") else 5.0, 2.0), 0, 100))
    hdop = float(np.clip(np.random.normal(1.4, 0.4), 0.8, 5.0))
    sat_ct = int(np.clip(np.random.normal(14, 3), 5, 25))
    # Integrity / schema
    hmac_fail = float(np.clip(np.random.normal(0.05, 0.08), 0, 1))
    schema_violation = float(np.clip(np.random.normal(0.03, 0.05), 0, 1))
    dup_ratio = float(np.clip(np.random.normal(0.02, 0.04), 0, 1))
    # Wi-Fi / attach churn
    deauth_rate = float(np.clip(np.random.normal(0.3, 0.6), 0, 8))
    assoc_churn = float(np.clip(np.random.normal(0.2, 0.5), 0, 10))
    rogue_rssi_gap = float(np.clip(np.random.normal(-2, 4), -30, 30))
    # Move + sequence
    move_device(dev)
    st.session_state.seq_counter[dev["device_id"]] += 1

    lat, lon = dev["lat"], dev["lon"]
    def within(center, radius_m):
        (clat, clon) = center
        dlat, dlon = abs(lat-clat), abs(lon-clon)
        return (dlat*111000)**2 + (dlon*55000)**2 <= (radius_m**2)

    is_attack = 0
    attack_type = "Normal"

    if scenario == "Jamming" and within(centers["jammer"], CFG.jammer_radius_m):
        delta = np.random.uniform(8, 18)
        snr -= delta; sinr -= delta*0.8; noise += np.random.uniform(6, 12)
        loss += np.random.uniform(8, 25); latency += np.random.uniform(40, 120); jitter += np.random.uniform(10, 30)
        is_attack = 1; attack_type = "Jamming"

    if scenario == "Access Breach" and within(centers["rogue"], CFG.breach_radius_m):
        rogue_rssi_gap = float(np.random.uniform(6, 22))
        deauth_rate += np.random.uniform(1.0, 4.5); assoc_churn += np.random.uniform(1.0, 5.0)
        loss += np.random.uniform(2, 10); latency += np.random.uniform(10, 60)
        is_attack = 1; attack_type = "Access Breach"

    if scenario == "GPS Spoofing" and dev["type"] in ("AMR","Truck") and within(centers["spoofer"], CFG.spoof_radius_m):
        pos_err += np.random.uniform(20, 80); hdop += np.random.uniform(1.0, 3.5); sat_ct = int(max(4, sat_ct + np.random.randint(-6, 6)))
        is_attack = 1; attack_type = "GPS Spoofing"

    if scenario == "Data Tamper" and within(centers["tamper"], CFG.breach_radius_m):
        if np.random.rand()<0.5: st.session_state.seq_counter[dev["device_id"]] -= 1  # replay-ish
        hmac_fail += np.random.uniform(0.2, 0.6); schema_violation += np.random.uniform(0.2, 0.5); dup_ratio += np.random.uniform(0.2, 0.6)
        is_attack = 1; attack_type = "Data Tamper"

    return {
        "device_id": dev["device_id"], "type": dev["type"], "tick": t,
        "lat": lat, "lon": lon,
        "snr": snr, "sinr_db": sinr, "noise_floor_dbm": noise,
        "packet_loss": loss, "latency_ms": latency, "jitter_ms": jitter,
        "pos_error_m": pos_err, "gnss_hdop": hdop, "sat_count": sat_ct,
        "hmac_fail_rate": hmac_fail, "schema_violation_rate": schema_violation, "dup_ratio": dup_ratio,
        "deauth_rate": deauth_rate, "assoc_churn": assoc_churn, "rogue_rssi_gap": rogue_rssi_gap,
        "is_attack": is_attack, "attack_type": attack_type
    }

FEATURES = [
    "snr","sinr_db","noise_floor_dbm","packet_loss","latency_ms","jitter_ms",
    "pos_error_m","gnss_hdop","sat_count",
    "hmac_fail_rate","schema_violation_rate","dup_ratio",
    "deauth_rate","assoc_churn","rogue_rssi_gap"
]

# ----------------------- Training -----------------------
def make_centers():
    clat, clon = CFG.map_center
    d1 = meters_to_latlon_offset(60); d2 = meters_to_latlon_offset(120)
    return {
        "jammer": (clat + d1[0], clon - d1[1]),
        "rogue":  (clat - d2[0], clon + d1[1]),
        "spoofer":(clat + d1[0], clon + d2[1]),
        "tamper": (clat - d1[0], clon - d1[1]),
    }

def synthesize_dataset(n_ticks=200, scenario_mix=None, progress_cb=None, pct_start=0, pct_end=70):
    if scenario_mix is None:
        scenario_mix = {"Normal": 0.5, "Jamming":0.15, "Access Breach":0.15, "GPS Spoofing":0.1, "Data Tamper":0.1}
    centers = make_centers()
    D=[]; devices = [d.copy() for d in st.session_state.devices]
    keys = list(scenario_mix.keys())
    probs = np.array([scenario_mix[k] for k in keys]); probs = probs/ probs.sum()
    for t in range(n_ticks):
        scen = np.random.choice(keys, p=probs)
        for dev in devices:
            D.append(simulate_tick(dev, t, scen, centers))
        if progress_cb and t%10==0:
            pct = pct_start + (pct_end - pct_start) * (t+1)/n_ticks
            progress_cb(pct, f"Synthesizing data ({t+1}/{n_ticks}) • scenario={scen}")
    df = pd.DataFrame(D)
    return df

def split_Xy(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y_anom = df["is_attack"].astype(int)
    y_type = df["attack_type"].copy()
    return X, y_anom, y_type

def fit_binary(X_tr, y_tr, X_val, y_val):
    classes = np.array([0,1])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
    cw = {int(c):float(w) for c,w in zip(classes, weights)}
    model = LGBMClassifier(
        n_estimators=220, max_depth=-1, learning_rate=0.08,
        num_leaves=31, min_child_samples=12, subsample=0.8, colsample_bytree=0.9,
        class_weight=cw, random_state=CFG.rng_seed, verbosity=-1  # <- silence logs; no 'verbose' in fit()
    )
    # IMPORTANT: No 'verbose' kwarg here (fixes your crash)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    return model

def fit_type(X_tr, y_tr):
    model = LGBMClassifier(
        n_estimators=200, learning_rate=0.1, num_leaves=31, subsample=0.85, colsample_bytree=0.9,
        random_state=CFG.rng_seed, verbosity=-1
    )
    enc = OrdinalEncoder()
    y_enc = enc.fit_transform(y_tr.to_numpy().reshape(-1,1)).ravel().astype(int)
    model.fit(X_tr, y_enc)
    labels = list(enc.categories_[0])
    return model, labels

def conformal_calibration(probs_cal: np.ndarray, y_cal: np.ndarray):
    alphas = 1.0 - probs_cal[np.arange(len(y_cal)), y_cal]
    return np.asarray(alphas)

def conformal_pvalue(prob_pos: float, cal_alphas: np.ndarray):
    alpha_star = 1.0 - float(prob_pos)
    n = len(cal_alphas); ge = int(np.sum(cal_alphas >= alpha_star))
    return (ge + 1) / (n + 1)

@st.cache_resource(show_spinner=False)
def train_models_cached(seed:int=CFG.rng_seed, ticks:int=260) -> Dict[str,Any]:
    df = synthesize_dataset(n_ticks=ticks)
    X, y_anom, y_type = split_Xy(df)
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y_anom, test_size=0.2, random_state=seed, stratify=y_anom)
    X_train, X_cal, y_train, y_cal = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=seed, stratify=y_tmp)
    model = fit_binary(X_train, y_train, X_cal, y_cal)
    p_cal = model.predict_proba(X_cal)
    cal_scores = conformal_calibration(np.vstack([1-p_cal[:,1], p_cal[:,1]]).T, y_cal.to_numpy())
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    brier = brier_score_loss(y_test, model.predict_proba(X_test)[:,1])
    mask_attack = y_anom==1
    if mask_attack.sum() >= 10:
        type_model, type_labels = fit_type(X[mask_attack], y_type[mask_attack])
    else:
        type_model, type_labels = None, ["Jamming","Access Breach","GPS Spoofing","Data Tamper"]
    global_shap = None
    if HAS_SHAP:
        try:
            expl = shap.TreeExplainer(model)
            samp = X.sample(min(300, len(X)), random_state=seed)
            sv = expl.shap_values(samp)
            if isinstance(sv, list) and len(sv)==2: sv = sv[1]
            mean_abs = np.abs(sv).mean(axis=0)
            global_shap = dict(zip(X.columns.tolist(), mean_abs.tolist()))
        except Exception:
            global_shap = None
    return {"model": model, "type_model": type_model, "type_labels": type_labels,
            "feature_names": FEATURES[:], "cal_scores": cal_scores,
            "auc": float(auc), "brier": float(brier), "global_shap": global_shap}

def train_model_with_progress(n_ticks=350):
    ph = st.empty(); bar = st.progress(0, text="Initializing…"); start = time.time()
    def update(pct, label): bar.progress(min(100, int(pct)), text=label)
    update(5, "Preparing data…")
    df = synthesize_dataset(n_ticks=n_ticks, progress_cb=update, pct_start=5, pct_end=70)
    X, y_anom, y_type = split_Xy(df)
    update(72, "Training anomaly detector…")
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y_anom, test_size=0.2, random_state=CFG.rng_seed, stratify=y_anom)
    X_train, X_cal, y_train, y_cal = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=CFG.rng_seed, stratify=y_tmp)
    model = fit_binary(X_train, y_train, X_cal, y_cal)  # <- no verbose kwarg
    update(83, "Calibrating confidence (conformal)…")
    p_cal = model.predict_proba(X_cal)
    cal_scores = conformal_calibration(np.vstack([1-p_cal[:,1], p_cal[:,1]]).T, y_cal.to_numpy())
    update(88, "Training attack-type classifier…")
    mask_attack = y_anom==1
    type_model, type_labels = (None, ["Jamming","Access Breach","GPS Spoofing","Data Tamper"])
    if mask_attack.sum() >= 10:
        type_model, type_labels = fit_type(X[mask_attack], y_type[mask_attack])
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    brier = brier_score_loss(y_test, model.predict_proba(X_test)[:,1])
    global_shap = None
    if HAS_SHAP:
        try:
            expl = shap.TreeExplainer(model)
            samp = X.sample(min(300, len(X)), random_state=CFG.rng_seed)
            sv = expl.shap_values(samp)
            if isinstance(sv, list) and len(sv)==2: sv = sv[1]
            mean_abs = np.abs(sv).mean(axis=0)
            global_shap = dict(zip(X.columns.tolist(), mean_abs.tolist()))
        except Exception:
            global_shap = None
    st.session_state.model = model
    st.session_state.type_model = type_model
    st.session_state.type_names = type_labels
    st.session_state.feature_names = FEATURES[:]
    st.session_state.conformal_scores = cal_scores
    st.session_state.metrics = {"auc": float(auc), "brier": float(brier)}
    dur = int(time.time() - start)
    update(100, f"Done. Model AUC={auc:.2f}, Brier={brier:.3f} • Duration {dur}s")
    time.sleep(0.4); ph.empty()

# ----------------------- Explanations -----------------------
def confidence_label(prob, pval, th=CFG.threshold):
    if pval is not None:
        if pval <= 0.05:   return "High",  "Confidence from conformal p-value (lower = stronger)."
        elif pval <= 0.20: return "Medium","Confidence from conformal p-value (lower = stronger)."
        else:              return "Low",   "Confidence from conformal p-value (lower = stronger)."
    gap = float(prob) - float(th)
    if gap >= 0.15:  return "High",  f"Risk {prob:.2f} well above threshold {th:.2f}."
    if gap >= 0.05:  return "Medium",f"Risk {prob:.2f} above threshold {th:.2f}."
    return "Low",    f"Risk {prob:.2f} near threshold {th:.2f}."

def build_enduser_symptoms(inc: dict) -> List[str]:
    f = inc.get("features", {}) or {}
    anchors = { "snr":22,"sinr_db":18,"noise_floor_dbm":-92,"packet_loss":1.5,"latency_ms":28,"jitter_ms":4,
                "pos_error_m":3,"gnss_hdop":1.4,"hmac_fail_rate":0.05,"schema_violation_rate":0.03,"dup_ratio":0.02,
                "deauth_rate":0.3,"assoc_churn":0.2,"rogue_rssi_gap":-2}
    stds    = { "snr":4,"sinr_db":5,"noise_floor_dbm":3,"packet_loss":1.2,"latency_ms":10,"jitter_ms":2,
                "pos_error_m":2,"gnss_hdop":0.4,"hmac_fail_rate":0.08,"schema_violation_rate":0.05,"dup_ratio":0.04,
                "deauth_rate":0.6,"assoc_churn":0.5,"rogue_rssi_gap":4}
    def z(name): 
        if name not in f: return 0.0
        return (float(f[name]) - anchors[name])/(stds[name]+1e-6)
    bullets=[]
    if z("noise_floor_dbm") > 2.0 or z("snr") < -2.0 or z("sinr_db") < -2.0: bullets.append("Signal quality **degraded** (interference likely).")
    if z("packet_loss") > 2.0 or z("jitter_ms") > 2.0 or z("latency_ms") > 2.0: bullets.append("**High errors / delay** observed.")
    if z("pos_error_m") > 2.0 or z("gnss_hdop") > 2.0: bullets.append("Location reading **unstable**.")
    if z("hmac_fail_rate") > 2.0 or z("schema_violation_rate") > 2.0 or z("dup_ratio") > 2.0: bullets.append("Data integrity **issues** detected.")
    if z("deauth_rate") > 2.0 or z("assoc_churn") > 2.0 or f.get("rogue_rssi_gap", -10) > 6: bullets.append("Unusual **network (re)auth activity**.")
    if not bullets: bullets.append("Behavior is **unusual** compared to normal for this device.")
    return bullets[:3]

def build_anomaly_explanation(inc: dict) -> str:
    fvals = inc.get("features", {})
    anchors = { "snr":22,"sinr_db":18,"noise_floor_dbm":-92,"packet_loss":1.5,"latency_ms":28,"jitter_ms":4,
                "pos_error_m":3,"gnss_hdop":1.4,"hmac_fail_rate":0.05,"schema_violation_rate":0.03,"dup_ratio":0.02,
                "deauth_rate":0.3,"assoc_churn":0.2,"rogue_rssi_gap":-2}
    stds    = { "snr":4,"sinr_db":5,"noise_floor_dbm":3,"packet_loss":1.2,"latency_ms":10,"jitter_ms":2,
                "pos_error_m":2,"gnss_hdop":0.4,"hmac_fail_rate":0.08,"schema_violation_rate":0.05,"dup_ratio":0.04,
                "deauth_rate":0.6,"assoc_churn":0.5,"rogue_rssi_gap":4}
    z = {k: (float(fvals.get(k, anchors[k]))-anchors[k])/(stds[k]+1e-6) for k in anchors.keys()}
    top = sorted(z.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    parts=[]
    for k,v in top:
        arrow = "↑" if v>0 else "↓"
        parts.append(f"{arrow} **{k}** deviated ({v:+.2f}σ).")
    return " • ".join(parts)

def build_type_explanation(inc: dict) -> str:
    t = (inc.get("type_label") or "Unknown").replace(" (low conf)","")
    if t=="Unknown": return "Model could not confidently decide an attack type."
    info = explain_attack_plain(t)
    if not info: return f"Heuristics and ML both suggest **{t}**."
    return f"Signals align with **{info['name']}**: {info['symptoms']}"

# ----------------------- Runtime detection -----------------------
def fuse_type_probs(ml_probs: Dict[str,float], feats: Dict[str,float]) -> Tuple[str,float,Dict[str,float]]:
    rules = {"Jamming":0.0, "Access Breach":0.0, "GPS Spoofing":0.0, "Data Tamper":0.0}
    if feats.get("snr",30) < 12 or feats.get("noise_floor_dbm",-95) > -85 or feats.get("packet_loss",0) > 8: rules["Jamming"] += 0.35
    if feats.get("rogue_rssi_gap",-10) > 6 or feats.get("deauth_rate",0) > 2.0 or feats.get("assoc_churn",0) > 2.0: rules["Access Breach"] += 0.35
    if feats.get("pos_error_m",0) > 15 or feats.get("gnss_hdop",0) > 3.0: rules["GPS Spoofing"] += 0.35
    if feats.get("hmac_fail_rate",0) > 0.2 or feats.get("schema_violation_rate",0) > 0.2 or feats.get("dup_ratio",0) > 0.2: rules["Data Tamper"] += 0.35
    fused = {k: 0.65*ml_probs.get(k,0.0) + 0.35*rules[k] for k in rules.keys()}
    label = max(fused.items(), key=lambda kv: kv[1])[0]; conf = fused[label]
    return label, conf, {"ML": ml_probs, "Rules": rules, "Fused": fused}

def severity_for(prob: float):
    if prob >= 0.8: return "High"
    if prob >= 0.6: return "Medium"
    return "Low"

def detect_on_row(row: pd.Series) -> Dict[str,Any]:
    if not is_model_ready():
        return {"prob": 0.0, "p_value": None, "severity": "Low",
                "type_label":"Unknown", "type_conf":0.0, "type_parts":{}}
    X = row[FEATURES].to_frame().T
    prob = float(st.session_state.model.predict_proba(X)[:,1][0])
    pval = None
    if st.session_state.conformal_scores is not None:
        pval = float(conformal_pvalue(prob, st.session_state.conformal_scores))
    sev = severity_for(prob)
    ml_probs = {k:0.0 for k in st.session_state.type_names if k!="Normal"}
    if st.session_state.type_model is not None:
        tp = st.session_state.type_model.predict_proba(X)[0]
        labels = st.session_state.type_names
        for i,lab in enumerate(labels):
            if lab=="Normal": continue
            if i < len(tp): ml_probs[lab] = float(tp[i])
    feats = row[FEATURES].to_dict()
    label, conf, parts = fuse_type_probs(ml_probs, feats)
    return {"prob": prob, "p_value": pval, "severity": sev,
            "type_label": label if conf>=0.4 else "Unknown (low conf)",
            "type_conf": conf, "type_parts": parts}

# ----------------------- UI: Quick Start -----------------------
def render_quickstart():
    nonce = st.session_state.ui_nonce
    show = st.session_state.get("show_quickstart", True)
    ready = is_model_ready()
    if not st.session_state.get("first_run_toast") and not ready:
        st.toast("Welcome! Train the model to get started → left sidebar.", icon="👋")
        st.session_state.first_run_toast = True
    with st.expander("🚀 Quick Start (no manual needed)", expanded=(show and not ready)):
        with st.container():
            st.markdown(
                """
1. **Train the model** (left sidebar → *Train / Retrain models*).  
2. **Pick a Scenario** (Jamming, Access Breach, GPS Spoofing, Data Tamper, or Normal).  
3. **Press Auto stream** (or *Step once*) to see live signals.  
4. **Watch the map & KPIs** (risk dots grow; ⚠ marks above-threshold devices).  
5. **Open Incidents** for plain explanations & what-to-do.  
6. **Insights** shows global behavior; **Governance** gives EU AI Act artifacts.
                """.strip()
            )
            c1,c2 = st.columns([1,1])
            with c1:
                if not ready and st.button("▶️ Train now", key=f"qs_train_{nonce}"):
                    train_model_with_progress(n_ticks=300)
            with c2:
                st.checkbox("Hide this next time", value=not show, key=f"qs_hide_{nonce}")
                st.session_state.show_quickstart = not st.session_state[f"qs_hide_{nonce}"]

def require_model_banner():
    if not is_model_ready():
        st.warning("No model yet. Click **Train / Retrain models** in the sidebar to start.", icon="⚠️")
        return True
    return False

# ----------------------- Sidebar -----------------------
init_state()
st.sidebar.header("Controls")

layout_density = st.sidebar.radio("Layout density", ["Comfortable","Compact"], index=0)
# inject CSS AFTER we read density
inject_ux(layout_density)

scenario = st.sidebar.selectbox("Scenario", SCENARIOS, index=0)
st.sidebar.markdown("**Environment**")
cellular_mode = st.sidebar.radio("Network profile", ["🏭 Yard Wi-Fi/private-5G","🚦 Road 5G/LTE"], index=1)
CFG.threshold = st.sidebar.slider("Incident threshold (Risk ≥ alert)", 0.4, 0.9, CFG.threshold, 0.01)
auto_stream = st.sidebar.toggle("Auto stream", value=True)
step_btn = st.sidebar.button("Step once")
st.sidebar.divider()
st.sidebar.header("Models")
if st.sidebar.button("Train / Retrain models"):
    train_model_with_progress(n_ticks=300)
st.sidebar.toggle("Help mode (show hints)", value=st.session_state.help_mode, key="help_mode")

# ----------------------- Header -----------------------
st.title("TRUST AI — Industrial Wireless Security (Sundsvall)")
render_quickstart()

ready = is_model_ready()
inc_counts = {"High":0,"Medium":0,"Low":0}
for _inc in st.session_state.get("incidents", []):
    if _inc["severity"] in inc_counts: inc_counts[_inc["severity"]] += 1
scenario_icon = ICONS.get(scenario, "ℹ️")
role = st.selectbox("Audience profile", ["End User","Domain Expert","Regulator","AI Builder","Executive"], index=0, key="role_sel")

st.markdown(
    f"""
    <div class="panel">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <div style="font-weight:600;">{ICONS['app']} TRUST AI</div>
        <div class="subtle">Scenario:</div>
        <div class="badge">{scenario_icon}&nbsp;{scenario}</div>
        <div class="subtle">Profile:</div>
        <div class="badge">{role}</div>
        <div style="margin-left:auto" class="subtle">Model: {('✅ ready' if ready else '⏳ not trained')}</div>
      </div>
      <div style="margin-top:.5rem; display:flex; gap:14px; flex-wrap:wrap;">
        <span class="subtle">Incidents:</span>
        {"".join([severity_badge(k)+f" <span class='subtle'>{v}</span> " for k,v in inc_counts.items() if v>0]) or "<span class='subtle'>none yet</span>"}
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------- Tabs -----------------------
tab_overview, tab_inc, tab_insight, tab_gov = st.tabs(
    [f"{ICONS['map']} Fleet", f"{ICONS['inc']} Incidents", f"{ICONS['insight']} Insights", f"{ICONS['gov']} Governance"]
)

# ----------------------- Live stream / step -----------------------
def make_centers():  # redefine here to ensure scope
    clat, clon = CFG.map_center
    d1 = meters_to_latlon_offset(60); d2 = meters_to_latlon_offset(120)
    return {
        "jammer": (clat + d1[0], clon - d1[1]),
        "rogue":  (clat - d2[0], clon + d1[1]),
        "spoofer":(clat + d1[0], clon + d2[1]),
        "tamper": (clat - d1[0], clon - d1[1]),
    }

def run_one_tick():
    centers = make_centers()
    t = st.session_state.tick + 1
    st.session_state.tick = t
    recs=[]
    for dev in st.session_state.devices:
        recs.append(simulate_tick(dev, t, scenario, centers))
    df = pd.DataFrame(recs)
    st.session_state.fleet_records.extend(df.to_dict("records"))
    if ready:
        for _, row in df.iterrows():
            out = detect_on_row(row)
            prob = out["prob"]
            if prob >= CFG.threshold:
                st.session_state.incidents.append({
                    "id": f"{row.device_id}_{t}",
                    "device_id": row.device_id, "type": row.type, "tick": int(t),
                    "scenario": scenario, "prob": out["prob"], "p_value": out["p_value"],
                    "severity": out["severity"], "type_label": out["type_label"], "type_conf": out["type_conf"],
                    "lat": float(row.lat), "lon": float(row.lon),
                    "features": {k: json_default(row[k]) for k in FEATURES}
                })

if auto_stream or step_btn:
    run_one_tick()

# ----------------------- Fleet tab -----------------------
with tab_overview:
    # Always render a map (even before any streaming)
    if st.session_state.fleet_records:
        df_map = pd.DataFrame(list(st.session_state.fleet_records))
        df_last = df_map.sort_values("tick").groupby("device_id").tail(1).reset_index(drop=True)
    else:
        df_last = pd.DataFrame(st.session_state.devices).copy()
        df_last["tick"] = 0
        # provide neutral telemetry columns for tooltips
        for col in ["snr","packet_loss"]:
            df_last[col] = np.nan
    if ready and "snr" not in df_last.columns:
        # If user trained before first stream, ensure columns exist
        for col in ["snr","packet_loss"]:
            df_last[col] = np.nan

    # Risk for coloring (0 until model ready + streaming)
    if ready and st.session_state.fleet_records:
        risks=[]
        for _,r in df_last.iterrows():
            X = pd.DataFrame([[r.get(f, np.nan) for f in FEATURES]], columns=FEATURES)
            try:
                risks.append(float(st.session_state.model.predict_proba(X)[:,1][0]))
            except Exception:
                risks.append(0.0)
        df_last["risk"] = risks
    else:
        df_last["risk"] = 0.0

    df_last["label"] = df_last.apply(lambda r: f"{ICONS.get(r.type,'📟')} {r.device_id} ({r.type})", axis=1)

    dev_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_last, get_position='[lon, lat]',
        get_radius=10, get_fill_color='[min(255, int(risk*255)), int((1.0-risk)*180), 40]',
        pickable=True
    )
    text_layer = pdk.Layer(
        "TextLayer",
        data=df_last, get_position='[lon, lat]',
        get_text="label", get_size=12, get_color=[30,30,30],
        get_alignment_baseline="'bottom'"
    )
    warn_df = df_last[df_last["risk"]>=CFG.threshold].copy()
    warn_df["ring_radius"] = 40
    warn_layer = pdk.Layer(
        "ScatterplotLayer", data=warn_df, get_position='[lon, lat]',
        get_fill_color='[0,0,0,0]', get_radius='ring_radius',
        stroked=True, get_line_color=[255,0,0,220], get_line_width=2
    )
    view = pdk.ViewState(latitude=CFG.map_center[0], longitude=CFG.map_center[1], zoom=15, pitch=0)
    tooltip = {
        "html": "<b>{device_id}</b> · {type}<br/>Risk: {risk}<br/>SNR: {snr} dB · Loss: {packet_loss}%<br/><i>Click for details</i>",
        "style": {"backgroundColor":"rgba(255,255,255,.95)","color":"#111"}
    }
    # IMPORTANT: use_container_width (not width='stretch') to avoid pydeck type error
    st.pydeck_chart(pdk.Deck(layers=[dev_layer, warn_layer, text_layer], initial_view_state=view, tooltip=tooltip),
                    use_container_width=True)

    # Fleet charts (2×2 compact)
    fr = pd.DataFrame(list(st.session_state.fleet_records))
    if len(fr)>0:
        section("Fleet KPIs (avg over devices)")
        recent = fr.groupby("tick")[["snr","packet_loss","latency_ms","pos_error_m"]].mean().reset_index()
        nonce = st.session_state.ui_nonce
        c1, c2 = st.columns(2)
        with c1:
            small_line_chart(recent, "tick", "snr", "Fleet avg SNR", key=f"snr_{nonce}")
        with c2:
            small_line_chart(recent, "tick", "packet_loss", "Fleet avg Packet loss", key=f"pl_{nonce}")
        c3, c4 = st.columns(2)
        with c3:
            small_line_chart(recent, "tick", "latency_ms", "Fleet avg Latency", key=f"lat_{nonce}")
        with c4:
            small_line_chart(recent, "tick", "pos_error_m", "Fleet avg GNSS position error", key=f"pos_{nonce}")
        if st.session_state.help_mode:
            st.caption("Axes show units (dB, %, ms, m). Hover for exact values.")

# ----------------------- Incident rendering -----------------------
def render_device_inspector(inc:Dict[str,Any], base_key:str):
    st.markdown(f"**Risk score:** {inc['prob']:.2f}  •  **Severity:** {inc['severity']}")
    if HAS_SHAP and ready:
        try:
            expl = shap.TreeExplainer(st.session_state.model)
            xrow = pd.DataFrame([[inc["features"].get(f, np.nan) for f in FEATURES]], columns=FEATURES)
            sv = expl.shap_values(xrow)
            if isinstance(sv, list) and len(sv)==2: sv = sv[1]
            vals = pd.Series(np.abs(sv[0]), index=FEATURES).sort_values(ascending=False)[:8]
            fig = px.bar(vals[::-1])
            fig.update_layout(height=230, margin=dict(l=8,r=8,t=24,b=8), showlegend=False, title="Top contributing features (local SHAP)")
            st.plotly_chart(fig, use_container_width=True, key=f"shap_local_{base_key}")
        except Exception:
            st.caption("Local SHAP unavailable; showing deviations instead.")
            st.write(build_anomaly_explanation(inc))
    else:
        st.caption("Local SHAP not installed; showing deviations instead.")
        st.write(build_anomaly_explanation(inc))

def render_incident_card(inc:Dict[str,Any], role:str, scope:str):
    base_key = f"{inc['id']}_{scope}_{st.session_state.ui_nonce}"
    pv = inc.get("p_value"); pv_str = f"{pv:.3f}" if pv is not None else "—"
    icon = ICONS.get(inc["scenario"], "⚠️")
    pill = attack_chip(inc.get("type_label","Unknown"))
    st.markdown(
        f"""
        <div class="panel">
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
            <div style="font-weight:600;">{icon} {inc['device_id']} <span class="subtle">({inc['type']})</span></div>
            <div class="badge">Risk {inc['prob']:.2f}</div>
            {severity_badge(inc.get("severity","—"))}
            {pill}
            <div class="subtle">p={pv_str}</div>
          </div>
        """,
        unsafe_allow_html=True
    )
    if role == "End User":
        prob = float(inc.get("prob", 0.0)); pv   = inc.get("p_value")
        conf_lbl, conf_tip = confidence_label(prob, pv)
        b1,b2,b3 = st.columns(3)
        b1.metric("Severity", inc.get("severity","—"))
        b2.metric("Risk score", f"{prob:.2f}")
        b3.metric("Confidence", conf_lbl)
        if st.session_state.help_mode: st.caption(conf_tip)
        st.markdown("#### What we noticed")
        for line in build_enduser_symptoms(inc): st.markdown(f"- {line}")
        st.caption("We describe problems in everyday terms. Technical details are available below.")
        tlabel = (inc.get("type_label") or "Unknown").replace(" (low conf)","")
        info = explain_attack_plain(tlabel)
        st.markdown("#### What this likely means")
        if info:
            st.markdown(f"**{info['name']}** — {info['what']}")
            st.markdown(f"**Typical signs:** {info['symptoms']}")
            st.markdown(f"**Why it matters:** {info['impact']}")
            st.markdown("#### What to do now")
            st.write(info["action"])
        else:
            st.write("System isn’t sure about the cause yet. Retry or move a short distance and try again.")
        with st.expander("Technical details"):
            st.markdown("**Why anomalous (model view):**"); st.markdown(build_anomaly_explanation(inc))
            st.markdown("**Why this type (model view):**"); st.markdown(build_type_explanation(inc))
            render_device_inspector(inc, base_key)
    elif role in ("Domain Expert","AI Builder"):
        c1,c2 = st.columns([1.1,1])
        with c1: render_device_inspector(inc, base_key)
        with c2:
            st.markdown("##### Type attribution (fused)")
            parts = inc.get("type_parts", {})
            if parts:
                fused = pd.Series(parts["Fused"]).sort_values()
                fig = px.bar(fused, title="")
                fig.update_layout(height=220, margin=dict(l=8,r=8,t=24,b=8), showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key=f"fused_{base_key}")
            st.markdown("##### Features"); st.json(inc["features"])
    elif role in ("Regulator","Executive"):
        st.markdown("**Summary**")
        st.markdown(f"- Device **{inc['device_id']}** ({inc['type']}) flagged with **{inc['severity']}** severity.")
        st.markdown(f"- Likely cause: {inc.get('type_label','Unknown')}.")
        st.markdown(f"- Confidence: {confidence_label(inc['prob'], inc.get('p_value'))[0]}.")
        st.markdown("**Action**")
        info = explain_attack_plain((inc.get("type_label") or "Unknown").replace(" (low conf)",""))
        st.write(info["action"] if info else "Retry shortly; escalate if it persists.")
        with st.expander("Details / audit"):
            st.json({k:v for k,v in inc.items() if k not in ("features",)})
            st.json(inc["features"])
    lcol, rcol = st.columns(2)
    with lcol:
        if st.button("Acknowledge", key=f"ack_{base_key}"):
            st.session_state.incident_labels[base_key] = {"ack": True}
    with rcol:
        if st.button("Mark false positive", key=f"fp_{base_key}"):
            st.session_state.incident_labels[base_key] = {"false_positive": True}
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------- Incidents tab -----------------------
with tab_inc:
    if require_model_banner(): st.stop()
    all_inc = list(reversed(st.session_state.incidents))
    if not all_inc:
        st.success("All clear so far — no incidents above the threshold.", icon="✅")
    else:
        st.markdown("#### Filters")
        c1,c2,c3 = st.columns(3)
        with c1:
            sev_sel = st.multiselect("Severity", ["High","Medium","Low"], default=["High","Medium","Low"], key=f"sev_sel_{st.session_state.ui_nonce}")
        with c2:
            type_sel = st.multiselect("Likely type", ["Jamming","Access Breach","GPS Spoofing","Data Tamper","Unknown"], default=["Jamming","Access Breach","GPS Spoofing","Data Tamper","Unknown"], key=f"type_sel_{st.session_state.ui_nonce}")
        with c3:
            who = st.selectbox("Audience view", ["End User","Domain Expert","Regulator","AI Builder","Executive"], index=["End User","Domain Expert","Regulator","AI Builder","Executive"].index(st.session_state.role_sel), key=f"aud_sel_{st.session_state.ui_nonce}")
        for inc in all_inc:
            tlabel = inc.get("type_label","Unknown").replace(" (low conf)","")
            if inc["severity"] in sev_sel and (tlabel in type_sel or (tlabel=="Unknown" and "Unknown" in type_sel)):
                render_incident_card(inc, who, scope="inc")
        def audit_log_json():
            return [ {k: (v if k!="features" else v) for k,v in inc.items()} for inc in st.session_state.incidents ]
        st.download_button(
            "Download incidents (JSON)",
            data=json.dumps(audit_log_json(), indent=2, default=json_default).encode("utf-8"),
            file_name=f"incidents_{int(time.time())}.json",
            mime="application/json",
            key=f"dl_inc_{st.session_state.ui_nonce}"
        )

# ----------------------- Insights tab -----------------------
with tab_insight:
    if require_model_banner(): st.stop()
    section("Model & Data")
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: st.metric("Devices", len(st.session_state.devices))
    with k2: st.metric("Incidents (session)", len(st.session_state.incidents))
    with k3:
        auc = (st.session_state.metrics or {}).get("auc", 0)
        st.metric("Model AUC", f"{auc:.2f}")
        if st.session_state.help_mode: st.caption("**Model AUC**: discrimination; 0.5=random, 1.0=perfect (higher is better).")
    with k4:
        brier = (st.session_state.metrics or {}).get("brier", 0)
        st.metric("Brier score", f"{brier:.3f}")
        if st.session_state.help_mode: st.caption("**Calibration reliability (Brier)**: lower is better.")
    with k5:
        fr = pd.DataFrame(list(st.session_state.fleet_records))
        fleet_risk = 0.0
        if len(fr)>0 and ready:
            last = fr.sort_values("tick").groupby("device_id").tail(1)
            probs=[]
            for _,r in last.iterrows():
                X = pd.DataFrame([[r.get(f, np.nan) for f in FEATURES]], columns=FEATURES)
                try: probs.append(float(st.session_state.model.predict_proba(X)[:,1][0]))
                except Exception: probs.append(0.0)
            fleet_risk = float(np.mean(probs)) if probs else 0.0
        st.metric("Fleet risk (mean prob)", f"{fleet_risk:.2f}")
    gshap = st.session_state.get("global_shap")
    if gshap:
        section("Global feature importance (mean |SHAP|)")
        s = pd.Series(gshap).sort_values()[-12:]
        fig = px.bar(s, title="Higher bars = more influential features globally")
        fig.update_layout(height=320, margin=dict(l=8,r=8,t=38,b=8), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=f"gshap_{st.session_state.ui_nonce}")
        if st.session_state.help_mode:
            st.caption("**Global importance (mean |SHAP|)**: average absolute contribution across many samples.")
    fr = pd.DataFrame(list(st.session_state.fleet_records))
    if len(fr)>0:
        section("Fleet behavior over time (averages)")
        recent = fr.groupby("tick")[["snr","packet_loss","latency_ms","pos_error_m"]].mean().reset_index()
        nonce = st.session_state.ui_nonce + 1
        c1, c2 = st.columns(2)
        with c1:
            small_line_chart(recent, "tick", "snr", "Fleet avg SNR", key=f"ins_snr_{nonce}")
        with c2:
            small_line_chart(recent, "tick", "packet_loss", "Fleet avg Packet loss", key=f"ins_pl_{nonce}")
        c3, c4 = st.columns(2)
        with c3:
            small_line_chart(recent, "tick", "latency_ms", "Fleet avg Latency", key=f"ins_lat_{nonce}")
        with c4:
            small_line_chart(recent, "tick", "pos_error_m", "Fleet avg GNSS position error", key=f"ins_pos_{nonce}")
    section("Attack type glossary (plain language)")
    rows = []
    for k,v in ATTACK_GLOSSARY.items():
        rows.append({"Attack type": v["name"], "What it is": v["what"],
                     "Typical signs": v["symptoms"], "Why it matters": v["impact"], "What to do": v["action"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=240, key=f"gloss_{st.session_state.ui_nonce}")

# ----------------------- Governance tab -----------------------
with tab_gov:
    if not ready: st.info("Train the model to populate governance artifacts (metrics, features, calibration).", icon="ℹ️")
    section("Model transparency")
    st.markdown("""
- **Algorithms**: LightGBM (tree-boosted) binary classifier for anomaly detection; LightGBM multiclass for attack type.  
- **Features**: SNR, SINR, noise floor, packet loss, latency, jitter, GNSS error/HDOP/sat count, integrity rates (HMAC fail, schema violations, duplicates), and access churn (deauth, assoc churn, rogue RSSI gap).  
- **Training data**: Synthetic, generated from realistic industrial wireless patterns in Sundsvall campus bounds.  
- **Calibration**: Conformal p-value from a held-out calibration split.  
- **Explainability**: Global SHAP (feature influence) + local SHAP in incidents (where supported).
    """.strip())
    section("Data transparency")
    st.markdown("""
- **Personal data**: None (synthetic telemetry only).  
- **Provenance**: Generated in-app each session; parameters documented in code.  
- **Retention**: Session-scoped only; downloadable audit log available.
    """.strip())
    if ready:
        meta = {
            "auc": st.session_state.metrics.get("auc"),
            "brier": st.session_state.metrics.get("brier"),
            "features": st.session_state.feature_names,
            "type_labels": st.session_state.type_names,
            "trained_at": int(time.time())
        }
        st.download_button(
            "Download model card (JSON)",
            data=json.dumps(meta, indent=2, default=json_default).encode("utf-8"),
            file_name=f"model_card_{int(time.time())}.json",
            mime="application/json",
            key=f"gov_modelcard_{st.session_state.ui_nonce}"
        )
        if st.session_state.incidents:
            st.download_button(
                "Download audit log (incidents.json)",
                data=json.dumps(st.session_state.incidents, indent=2, default=json_default).encode("utf-8"),
                file_name=f"incidents_audit_log_{int(time.time())}.json",
                mime="application/json",
                key=f"gov_audit_{st.session_state.ui_nonce}"
            )

# ----------------------- Metric hints -----------------------
if st.session_state.help_mode:
    st.caption("""
**Metric hints**  
- **Model AUC**: discrimination; 0.5=random, 1.0=perfect. Higher is better.  
- **Fleet risk (mean prob)**: average anomaly probability; higher ⇒ site-wide stress.  
- **Fleet avg SNR**: link quality; <10 dB shaky, >20 dB healthy.  
- **Global importance (mean |SHAP|)**: average absolute contribution across many samples.  
- **Calibration reliability (Brier)**: overall calibration error (lower is better).
""")
