import math
import time
from collections import deque
from dataclasses import dataclass
import json
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
import warnings

# Silence the SHAP binary-classifier return-shape warning (we already handle it)
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
    # OLD: site_center: tuple = (59.3293, 18.0686)  # Stockholm-ish
    site_center: tuple = (62.3925, 17.3066)        # Sundsvall — Mid Sweden University area
    site_radius_m: float = 400
    jam_radius_m: float = 200
    retrain_on_start: bool = True


CFG = Config()

RAW_FEATURES = [
    "rssi", "snr", "packet_loss", "latency_ms", "jitter_ms",
    "pos_error_m", "auth_fail_rate", "crc_err", "throughput_mbps", "channel_util"
]

DEVICE_TYPES = ["AGV", "Truck", "Sensor", "Gateway"]
MOBILE_TYPES = {"AGV", "Truck"}

# =========================
# Streamlit page
# =========================
st.set_page_config(page_title="TRUST AI — Realistic Wireless Threat Demo", layout="wide")
st.title("TRUST AI — Realistic Wireless Threat Detection (Multi-Device)")
st.caption("Geospatial fleet • Physics-ish RF • LightGBM + SHAP + Conformal • Persona-based XAI • Live Training Progress + ETA")

# Sidebar
with st.sidebar:
    st.header("Demo Controls")
    scenario = st.selectbox(
        "Scenario",
        ["Normal", "Jamming (localized)", "GPS Spoofing (subset)", "Wi-Fi Breach (AP)", "Data Tamper (gateway)"],
        index=0
    )
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
    st.subheader("Display")
    show_map = st.checkbox("Show geospatial map", True)
    show_heatmap = st.checkbox("Show fleet heatmap (metric z-scores)", True)
    type_filter = st.multiselect("Show device types", DEVICE_TYPES, default=DEVICE_TYPES)
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
    # simple diurnal load pattern [0,1]
    return 0.5 + 0.5*math.sin((tick%600)/600*2*math.pi)

def fmt_eta(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "—"
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

def rf_and_network_model(row, tick, scen=None):
    """
    Generate correlated metrics based on distances and load.
    scen: one of ["Normal", "Jamming (localized)", "GPS Spoofing (subset)", "Wi-Fi Breach (AP)", "Data Tamper (gateway)"].
    If None, falls back to current UI selection for streaming.
    """
    if scen is None:
        scen = scenario

    ap = st.session_state.ap
    jam = st.session_state.jammer
    d_ap = max(1.0, haversine_m(row.lat, row.lon, ap["lat"], ap["lon"]))
    d_jam = haversine_m(row.lat, row.lon, jam["lat"], jam["lon"])
    load = time_of_day_load(tick)

    # Basic RF
    rssi = -40 - 18 * math.log10(d_ap) + np.random.normal(0, 2)
    rssi = float(np.clip(rssi, -90, -40))
    base_snr = 35 - 0.008 * d_ap + np.random.normal(0, 1.5)

    jam_penalty = 0.0
    if str(scen).startswith("Jamming"):
        if d_jam <= CFG.jam_radius_m:
            jam_penalty = (CFG.jam_radius_m - d_jam)/CFG.jam_radius_m * np.random.uniform(8, 15)
    snr = max(0.0, base_snr - jam_penalty)

    cong = 0.15*load
    loss = 1/(1+np.exp(0.4*(snr-15))) + cong + np.random.normal(0, 0.02)
    loss = float(np.clip(loss*100, 0, 60))  # %

    latency = 20 + 2.0*loss + 30*load + np.random.normal(0, 6)
    jitter  = 1.0 + 0.05*loss + 8*load + np.random.normal(0, 1.0)
    latency = float(max(3, latency))
    jitter  = float(max(0.2, jitter))

    thr = 90 - 0.8*loss - 40*load + np.random.normal(0, 5)
    thr = float(max(1, thr))

    channel_util = float(np.clip(100*(0.4+0.5*load) + np.random.normal(0, 5), 0, 100))

    auth_fail = np.random.exponential(0.05) + (0.5 if str(scen).startswith("Wi-Fi Breach") else 0)
    crc_err   = np.random.poisson(0.2)

    pos_error = np.random.normal(2.0, 0.7)
    if str(scen).startswith("GPS Spoofing") and row.type in MOBILE_TYPES and np.random.rand()<0.5:
        pos_error += np.random.uniform(15, 40)

    if str(scen).startswith("Data Tamper") and row.type in {"Gateway","Sensor"} and np.random.rand()<0.3:
        crc_err += np.random.poisson(2)
        loss += np.random.uniform(2, 8)
        thr = max(1, thr - np.random.uniform(5, 15))

    return dict(
        rssi=rssi, snr=float(snr),
        packet_loss=float(np.clip(loss, 0, 90)),
        latency_ms=latency, jitter_ms=jitter,
        pos_error_m=float(max(0.3, pos_error)),
        auth_fail_rate=float(max(0, auth_fail)),
        crc_err=int(max(0, crc_err)),
        throughput_mbps=thr, channel_util=channel_util
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
    """Generate rolling-window training data with progress [pct_start..pct_end]."""
    lat0, lon0 = CFG.site_center
    ap_lat, ap_lon = rand_point_near(lat0, lon0, 50)
    jam_lat, jam_lon = rand_point_near(lat0, lon0, 100)
    devs=[]
    for i in range(18):
        d_type=np.random.choice(DEVICE_TYPES, p=[0.5,0.2,0.2,0.1])
        lat,lon = rand_point_near(lat0, lon0, CFG.site_radius_m)
        devs.append(dict(device_id=f"T{i:02d}", type=d_type, lat=lat, lon=lon,
                         speed_mps=(np.random.uniform(0.5,2.5) if d_type in MOBILE_TYPES else 0.0),
                         heading=np.random.uniform(0,2*np.pi)))
    D=pd.DataFrame(devs)
    buf = {d: deque(maxlen=CFG.rolling_len) for d in D.device_id}
    X_rows=[]; y=[]
    t0 = time.time()
    for t in range(n_ticks):
        scen_code = np.random.choice(["Normal","J","GPS","Breach","Tamper"],
                                     p=[0.55,0.15,0.10,0.10,0.10])
        if   scen_code=="J":   sc="Jamming (localized)"
        elif scen_code=="GPS": sc="GPS Spoofing (subset)"
        elif scen_code=="Breach": sc="Wi-Fi Breach (AP)"
        elif scen_code=="Tamper": sc="Data Tamper (gateway)"
        else: sc="Normal"

        # move devices
        for i in D.index:
            if D.at[i,"type"] in MOBILE_TYPES:
                D.at[i,"heading"] += np.random.normal(0,0.3)
                step = D.at[i,"speed_mps"]*np.random.uniform(0.6,1.4)
                dn,de = step*math.cos(D.at[i,"heading"]), step*math.sin(D.at[i,"heading"])
                dlat,dlon = meters_to_latlon_offset(dn,de,D.at[i,"lat"])
                D.at[i,"lat"]+=dlat; D.at[i,"lon"]+=dlon

        # generate metrics under explicit scenario 'sc'
        for _, row in D.iterrows():
            fake_row = type("R",(object,),row.to_dict())()
            ap={"lat":ap_lat,"lon":ap_lon}; jam={"lat":jam_lat,"lon":jam_lon}
            st.session_state.ap=ap; st.session_state.jammer=jam
            metrics = rf_and_network_model(fake_row, t, sc)  # <-- pass scenario
            buf[row.device_id].append(metrics)
            feats = build_window_features(buf[row.device_id])
            if feats:
                X_rows.append(feats)
                y.append(0 if sc=="Normal" else 1)

        # progress + ETA
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

    # ---- Stage 2: Scale + Fit (70..95%) with per-iteration progress
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
        min_data_in_leaf=12,
        min_child_samples=12,   # match to silence warning
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
        pct = 70 + 25*frac  # 70..95
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

    # Suggest threshold that maximizes F1 on test set
    ths = np.linspace(0.30, 0.90, 61)
    best_f1, best_th = 0.0, CFG.threshold
    for th in ths:
        pred = (te_p >= th).astype(int)
        _, _, f1c, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        if f1c > best_f1:
            best_f1, best_th = f1c, th
    st.session_state.suggested_threshold = float(best_th)

    # Finalize
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

# Kickoff training on first load or on demand
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

    fleet_rows=[]
    # generate per-device metrics
    for _, row in st.session_state.devices.iterrows():
        m = rf_and_network_model(row, tick, scenario)  # <-- pass live scenario
        st.session_state.dev_buf[row.device_id].append(m)
        rec = {"tick":tick, "device_id":row.device_id, "type":row.type, "lat":row.lat, "lon":row.lon, **m}
        fleet_rows.append(rec)

    # build features + infer risks
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

    # group incident if widespread
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

    # append fleet rows for charts
    st.session_state.fleet_records.extend(fleet_rows)
    st.session_state.tick += 1

# Run ticks (only after a model exists)
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

# ---------- Overview (map + leaderboard + quick charts)
with tab_overview:
    left, right = st.columns([2,1])

    with left:
        if show_map and st.session_state.get("model") is not None:
            # Base dataframe + optional type filter
            df_map = st.session_state.devices.copy()
            if type_filter and len(type_filter) < len(DEVICE_TYPES):
                df_map = df_map[df_map["type"].isin(type_filter)].copy()

            # Compute risk & latest SNR/Loss for tooltip
            risks, snrs, losses = [], [], []
            for _, r in df_map.iterrows():
                feats = st.session_state.last_features.get(r.device_id, {})
                if feats:
                    X = pd.DataFrame([feats])
                    Xs = st.session_state.scaler.transform(X)
                    Xs_df = to_df(Xs, X.columns)
                    risks.append(float(st.session_state.model.predict_proba(Xs_df)[:, 1][0]))
                else:
                    risks.append(0.0)
                buf = st.session_state.dev_buf.get(r.device_id, [])
                if buf and len(buf) > 0:
                    snrs.append(float(buf[-1].get("snr", np.nan)))
                    losses.append(float(buf[-1].get("packet_loss", np.nan)))
                else:
                    snrs.append(np.nan)
                    losses.append(np.nan)

            df_map["risk"] = risks
            df_map["snr"] = snrs
            df_map["packet_loss"] = losses

            # Colors per type
            type_colors = {
                "AGV":     [0, 128, 255, 220],   # blue
                "Truck":   [255, 165, 0, 220],   # orange
                "Sensor":  [34, 197, 94, 220],   # green
                "Gateway": [147, 51, 234, 220],  # purple
            }
            df_map["fill_color"] = df_map["type"].map(type_colors)
            df_map["label"] = df_map.apply(lambda r: f"{r.device_id} ({r.type})", axis=1)
            df_map["radius"] = 6 + (df_map["risk"] * 16)  # 6..22 px

            layers = []
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=df_map,
                get_position='[lon, lat]', get_fill_color='fill_color',
                get_radius='radius', get_line_color=[0,0,0,140], get_line_width=1, pickable=True
            ))
            layers.append(pdk.Layer(
                "TextLayer", data=df_map,
                get_position='[lon, lat]', get_text='label',
                get_color=[20,20,20,255], get_size=12,
                get_alignment_baseline="'top'", get_pixel_offset=[0,10]
            ))
            ap_df = pd.DataFrame([{"lat": st.session_state.ap["lat"], "lon": st.session_state.ap["lon"], "name": "AP"}])
            layers.append(pdk.Layer("ScatterplotLayer", data=ap_df, get_position='[lon, lat]',
                                    get_fill_color='[30,144,255,240]', get_radius=10))
            layers.append(pdk.Layer("TextLayer", data=ap_df.assign(label="AP"), get_position='[lon, lat]',
                                    get_text='label', get_color=[30,144,255,255], get_size=14,
                                    get_alignment_baseline="'bottom'", get_pixel_offset=[0,-10]))
            if scenario.startswith("Jamming"):
                jam = st.session_state.jammer
                jam_df = pd.DataFrame([{"lat": jam["lat"], "lon": jam["lon"], "name": "Jammer"}])
                layers.append(pdk.Layer("ScatterplotLayer", data=jam_df, get_position='[lon, lat]',
                                        get_fill_color='[255,0,0,240]', get_radius=10))
                layers.append(pdk.Layer("TextLayer", data=jam_df.assign(label="Jammer"), get_position='[lon, lat]',
                                        get_text='label', get_color=[255,0,0,255], get_size=14,
                                        get_alignment_baseline="'bottom'", get_pixel_offset=[0,-10]))
                angles = np.linspace(0, 2*np.pi, 60)
                circle = [{
                    "path": [[
                        jam["lon"] + meters_to_latlon_offset(CFG.jam_radius_m * math.sin(a),
                                                             CFG.jam_radius_m * math.cos(a),
                                                             st.session_state.devices.lat.mean())[1],
                        jam["lat"] + meters_to_latlon_offset(CFG.jam_radius_m * math.sin(a),
                                                             CFG.jam_radius_m * math.cos(a),
                                                             st.session_state.devices.lat.mean())[0]
                    ] for a in angles],
                    "name": "jam_radius"
                }]
                layers.append(pdk.Layer("PathLayer", circle, get_path="path",
                                        get_color=[255,0,0], width_scale=4, width_min_pixels=1, opacity=0.25))
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
                    <span style="width:12px;height:12px;background:#0080FF;border-radius:2px;display:inline-block;"></span> AGV
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
            st.dataframe(df_lead, use_container_width=True)  # dataframe still supports this
        if st.session_state.group_incidents:
            gi = st.session_state.group_incidents[-1]
            st.markdown("### Group incident")
            st.info(f"{gi['affected']} of {gi['fleet']} devices impacted ({gi['ratio']:.0%}) • Scenario: {gi['scenario']}")

# ---------- Fleet View (heatmap + table)
with tab_fleet:
    fr = pd.DataFrame(list(st.session_state.fleet_records))
    if len(fr)>0:
        recent = fr[fr["tick"]>=st.session_state.tick-40]
        mat = recent.groupby("device_id")[["snr","packet_loss","latency_ms","jitter_ms","pos_error_m","auth_fail_rate","crc_err","throughput_mbps","channel_util"]].mean()
        z = (mat-mat.mean())/mat.std(ddof=0).replace(0,1)
        fig = px.imshow(z.T, color_continuous_scale="RdBu_r", aspect="auto",
                        labels=dict(color="z-score"), title="Fleet heatmap (recent mean z-scores)")
        st.plotly_chart(fig, width='stretch')
    st.dataframe(st.session_state.devices, use_container_width=True)

# ---------- Incidents (persona-based explanations)
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
                            st.write("Unknown device may be trying to access the network.")
                        else:
                            st.write("Some data looked inconsistent and may be rejected.")
                        st.markdown("#### What to do now")
                        if "Jamming" in inc["scenario"]:
                            st.write("Move 50–100 m away or retry; operators are switching channel.")
                        elif "GPS Spoofing" in inc["scenario"]:
                            st.write("Confirm location via badge scanner/UWB anchor if available.")
                        elif "Wi-Fi Breach" in inc["scenario"]:
                            st.write("Avoid entering passwords on pop-ups; IT is locking the network.")
                        else:
                            st.write("Retry and contact support if it persists.")
                        if pv is not None: st.caption(f"Confidence (lower = riskier): p-value = {pv_str}")

                    with tabs[1]:
                        st.markdown("#### Operational assessment")
                        st.write(f"- Severity **{sev}** (prob={inc['prob']:.2f}, p={pv_str})")
                        obs=" • ".join([f"{r['feature']}({r['impact']:+.3f})" for r in inc["reasons"]])
                        st.write(f"- Key observables: {obs}")
                        st.markdown("#### Playbook (priority)")
                        if "Jamming" in inc["scenario"]:
                            st.write("1) Channel hop; 2) Spectrum scan near jammer hotspot; 3) Directional/backup link.")
                        elif "GPS Spoofing" in inc["scenario"]:
                            st.write("1) Switch to Wi-Fi/UWB fusion; 2) Verify time source; 3) Geofence sanity check.")
                        elif "Wi-Fi Breach" in inc["scenario"]:
                            st.write("1) Quarantine SSID/VLAN; 2) Rotate keys; 3) Rogue AP sweep.")
                        else:
                            st.write("1) Reject bad packets; 2) Verify signatures; 3) Audit gateway path.")

                    with tabs[2]:
                        st.markdown("#### Assurance & governance")
                        st.write("- Transparent factors (see SHAP below) and plain-language summary.")
                        st.write(f"- Calibrated confidence: conformal p-value = {pv_str} (target coverage {int(CFG.coverage*100)}%).")
                        st.write("- Audit trail: downloadable incident evidence with model version & inputs.")
                        st.write("- Data scope: technical telemetry only; no personal data in this demo.")
                        st.write("- Continuous monitoring: drift checks (Insights ▸ Drift).")
                        evidence = {
                            "ts": inc["ts"], "tick": inc["tick"], "device_id": inc["device_id"], "type": inc["type"],
                            "lat": inc["lat"], "lon": inc["lon"], "scenario": inc["scenario"], "severity": inc["severity"],
                            "prob": inc["prob"], "p_value": inc["p_value"], "model_version": "LightGBM v2.2-demo",
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
                        st.caption("Positive impact pushes toward 'anomaly'; values in standardized space.")
                        st.markdown("#### Implementation")
                        st.write("- Model: LightGBM (depth≤3, ~60 trees), per-device rolling features.")
                        st.write("- Calibration: inductive conformal p-value.")
                        st.write("- Data realism: distance-based RF, localized jamming radius, diurnal load, coupled metrics.")

                    with tabs[4]:
                        st.markdown("#### Executive summary")
                        st.write(f"- Device {inc['device_id']} at risk: **{sev}**; scenario: **{inc['scenario']}**.")
                        st.write("- Impact: short-term performance or location errors; mitigations are ready.")
                        st.markdown("#### Decision & ROI")
                        st.write("- Take action now to avoid SLA breach; cost is low (config scans/changes).")
                        st.write("- Benefits: reduced downtime & safer ops.")
                        st.markdown("#### KPIs")
                        st.write("- Incidents today, Mean Time To Detect, Packet loss, Latency, SNR")

                with right:
                    st.json({k:inc[k] for k in ["prob","p_value","tick","device_id","type","scenario"]})

    if st.session_state.incidents:
        df_inc = pd.DataFrame(st.session_state.incidents)
        df_inc["top_features"] = df_inc["reasons"].apply(lambda r: "; ".join([f"{x['feature']}:{x['impact']:+.3f}" for x in r]))
        csv = df_inc.drop(columns=["reasons"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download incidents CSV", csv, "incidents.csv", "text/csv", key="dl_incidents_csv")

# ---------- Insights (global SHAP + drift)
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
            cols = ["snr","packet_loss","latency_ms","jitter_ms","pos_error_m","auth_fail_rate","crc_err","throughput_mbps","channel_util"]
            drift = []
            for c in cols:
                mu_a, mu_b = a[c].mean(), b[c].mean()
                sd = a[c].std(ddof=0)+1e-6
                z = (mu_b-mu_a)/sd
                drift.append({"metric":c, "zshift":z})
            df_drift = pd.DataFrame(drift).sort_values("zshift", ascending=False)
            fig = px.bar(df_drift, x="zshift", y="metric", orientation="h", title="Drift (mean shift vs early window)")
            st.plotly_chart(fig, width='stretch')

    st.markdown("### Model card")
    st.json({
        "model": "LightGBM (depth≤3, ~60 trees)",
        "features_per_device": len(feature_cols()),
        "training_metrics": st.session_state.metrics,
        "calibration": f"Conformal p-value (target coverage {int(CFG.coverage*100)}%)",
        "realism": "Distance-based RF, localized jamming, diurnal load, coupled metrics, role-specific devices",
        "intended_use": "Demonstration of trustworthy anomaly detection in wireless/logistics",
        "limitations": [
            "Synthetic but physics-inspired; do not use as-is in production",
            "Single-site example without true multi-AP handoff"
        ],
        "version": "v2.2-realistic"
    })
