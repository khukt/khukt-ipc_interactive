import time
from collections import deque
from dataclasses import dataclass
import io
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import shap

# -----------------------------
# Demo constants & schema
# -----------------------------
RAW_FEATURES = [
    "rssi", "snr", "packet_loss", "latency_ms", "jitter_ms",
    "pos_error_m", "auth_fail_rate", "crc_err", "throughput_mbps", "channel_util"
]
WINDOWS = [1, 5]              # seconds
ROLLING_LEN = 5               # keep last 5 seconds of raw for quick stats
SEED = 42
np.random.seed(SEED)

@dataclass
class Config:
    threshold: float = 0.75                 # anomaly probability threshold for incident
    coverage: float = 0.90                  # conformal target coverage
    retrain_every_s: int = 120              # optional scheduled retrain
    train_seconds: int = 120                # synthetic seconds to train on
    cal_seconds: int = 30                   # calibration set size
    eval_seconds: int = 30                  # holdout
    max_points_plot: int = 600              # cap chart memory
    depth: int = 3                          # LightGBM max depth
    n_estimators: int = 60                  # trees
    learning_rate: float = 0.08

CFG = Config()

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="TRUST AI: Wireless Threat Detection (Demo)", layout="wide")
st.title("TRUST AI — Wireless Threat Detection (Demo)")
st.caption("LightGBM + SHAP + Conformal Risk • Multi-level XAI UI")

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Demo Controls")
    scenario = st.selectbox(
        "Scenario",
        ["Normal", "Jamming", "GPS Spoofing", "Wi-Fi Breach", "Data Tamper"],
        index=0
    )
    speed = st.slider("Playback speed (x)", 1, 10, 2, help="Higher = more steps per refresh")
    auto = st.checkbox("Auto stream", value=True)
    reset_btn = st.button("Reset Stream")
    st.divider()
    st.subheader("Model")
    use_conformal = st.checkbox("Conformal risk (calibrated p-value)", value=True)
    st.caption("Conformal wraps the model score to provide a calibrated confidence signal.")
    st.divider()
    st.markdown("**Thresholds**")
    CFG.threshold = st.slider("Incident threshold (model prob.)", 0.50, 0.95, CFG.threshold, 0.01)
    st.caption("Alerts fire when model probability exceeds this threshold.")

# -----------------------------
# Session state init
# -----------------------------
def init_state():
    st.session_state.raw_buffer = deque(maxlen=CFG.max_points_plot)   # whole stream (dict rows)
    st.session_state.window_buf = deque(maxlen=ROLLING_LEN)           # short rolling window (raw)
    st.session_state.incidents = []                                   # list of dicts
    st.session_state.step = 0
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.shap_explainer = None
    st.session_state.conformal_scores = None  # calibration scores (nonconformity)
    st.session_state.train_metrics = {}
    st.session_state.last_train_time = time.time()
    st.session_state.drifts = {}              # feature drift (PSI-ish)
    st.session_state.baseline_distrib = None  # baseline feature ref
    st.session_state.trained = False
    st.session_state.last_prediction = None
    st.session_state.trigger_time = None      # when anomaly starts in the stream

if "raw_buffer" not in st.session_state or reset_btn:
    init_state()

# -----------------------------
# Synthetic data generator
# -----------------------------
def gen_normal_row():
    # Reasonable baselines + noise
    row = {
        "rssi": np.random.normal(-65, 3),
        "snr": np.random.normal(25, 2),
        "packet_loss": max(0, np.random.normal(0.5, 0.3)),
        "latency_ms": max(5, np.random.normal(30, 8)),
        "jitter_ms": max(0.5, np.random.normal(2, 0.7)),
        "pos_error_m": max(0.5, np.random.normal(2.5, 0.7)),
        "auth_fail_rate": max(0, np.random.normal(0.2, 0.2)),
        "crc_err": max(0, np.random.poisson(0.2)),
        "throughput_mbps": max(2, np.random.normal(50, 8)),
        "channel_util": np.clip(np.random.normal(35, 8), 0, 100),
    }
    return row

def inject_anomaly(row, scenario, t_since):
    r = row.copy()
    if scenario == "Jamming":
        # Sudden SNR↓, RSSI noise↑, loss↑, latency/jitter↑
        strength = min(1.0, t_since / 5.0)
        r["snr"] -= 8 * strength + np.random.uniform(0,2)
        r["packet_loss"] += 8 * strength + np.random.uniform(0,2)
        r["latency_ms"] += 40 * strength + np.random.uniform(0,10)
        r["jitter_ms"] += 5 * strength + np.random.uniform(0,2)
        r["channel_util"] += 10 * strength
    elif scenario == "GPS Spoofing":
        strength = min(1.0, t_since / 5.0)
        r["pos_error_m"] += 30 * strength + np.random.uniform(0,10)
        # subtle network oddities when position goes wild
        r["jitter_ms"] += 2 * strength
        r["latency_ms"] += 10 * strength
    elif scenario == "Wi-Fi Breach":
        strength = min(1.0, t_since / 5.0)
        r["auth_fail_rate"] += 8 * strength + np.random.uniform(0,2)
        r["channel_util"] += 20 * strength + np.random.uniform(0,5)
        r["crc_err"] += np.random.poisson(1 + 3*strength)
    elif scenario == "Data Tamper":
        # Impossible jumps / checksum oddities
        if np.random.rand() < 0.6:
            r["throughput_mbps"] *= np.random.uniform(1.5, 2.2)
        r["crc_err"] += np.random.poisson(2)
        r["packet_loss"] += np.random.uniform(3, 10)
    return r

# -----------------------------
# Feature engineering
# -----------------------------
def build_features(buf: deque):
    # We compute per-window stats (1s & 5s); here buf holds last ROLLING_LEN raw rows (~5s).
    if len(buf) == 0:
        return {}
    df = pd.DataFrame(list(buf))
    feats = {}
    for feat in RAW_FEATURES:
        series = df[feat]
        feats[f"{feat}_mean_1s"] = series[-1:].mean()
        feats[f"{feat}_std_1s"] = series[-1:].std(ddof=0) if len(series[-1:]) > 1 else 0.0
        feats[f"{feat}_mean_5s"] = series[-ROLLING_LEN:].mean()
        feats[f"{feat}_std_5s"] = series[-ROLLING_LEN:].std(ddof=0) if len(series) > 1 else 0.0
        # slope over last few points (approx)
        if len(series) >= 3:
            x = np.arange(len(series[-3:]))
            y = series[-3:].to_numpy()
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0
        feats[f"{feat}_slope"] = slope
        # z-score last value vs 5s mean/std
        mu, sd = feats[f"{feat}_mean_5s"], feats[f"{feat}_std_5s"]
        z = 0.0 if sd == 0 else (series.iloc[-1] - mu) / sd
        feats[f"{feat}_z"] = z
        # last jump magnitude
        if len(series) >= 2:
            feats[f"{feat}_jump"] = float(series.iloc[-1] - series.iloc[-2])
        else:
            feats[f"{feat}_jump"] = 0.0
    return feats

def feature_cols():
    cols = []
    for f in RAW_FEATURES:
        cols += [f"{f}_mean_1s", f"{f}_std_1s", f"{f}_mean_5s", f"{f}_std_5s", f"{f}_slope", f"{f}_z", f"{f}_jump"]
    return cols

# -----------------------------
# Training data synthesis
# -----------------------------
def synth_block(seconds, scenario="Normal", label=0):
    rows = []
    t0 = 0
    for s in range(seconds):
        r = gen_normal_row()
        if scenario != "Normal":
            # make anomaly grow after 1s
            r = inject_anomaly(r, scenario, t_since=max(0, s-1))
        rows.append(r)
    df = pd.DataFrame(rows)
    df["label"] = label
    return df

def make_training_data():
    # Mix normal & various anomalies
    nN = CFG.train_seconds
    nA = CFG.train_seconds // 3
    df = pd.concat([
        synth_block(nN, "Normal", 0),
        synth_block(nA, "Jamming", 1),
        synth_block(nA, "GPS Spoofing", 1),
        synth_block(nA, "Wi-Fi Breach", 1),
        synth_block(nA, "Data Tamper", 1),
    ], ignore_index=True)
    # Create rolling-window features as if streaming
    feats_all = []
    buf = deque(maxlen=ROLLING_LEN)
    labels = []
    for i, row in df.iterrows():
        raw = {k: row[k] for k in RAW_FEATURES}
        buf.append(raw)
        feats = build_features(buf)
        feats_all.append(feats)
        labels.append(row["label"])
    X = pd.DataFrame(feats_all).fillna(0.0)
    y = np.array(labels)
    # Split into (train/valid/calibration/test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.40, shuffle=True, random_state=SEED, stratify=y)
    X_cal, X_test, y_cal, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, shuffle=True, random_state=SEED, stratify=y_tmp)
    return (X_train, y_train), (X_cal, y_cal), (X_test, y_test), X

def train_model():
    (X_train, y_train), (X_cal, y_cal), (X_test, y_test), X_all = make_training_data()
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s   = scaler.transform(X_cal)
    X_test_s  = scaler.transform(X_test)

    model = LGBMClassifier(
        n_estimators=CFG.n_estimators,
        max_depth=CFG.depth,
        learning_rate=CFG.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED
    )
    model.fit(X_train_s, y_train)

    # SHAP explainer (TreeExplainer for GBDT)
    explainer = shap.TreeExplainer(model)

    # Conformal calibration scores (nonconformity)
    cal_proba = model.predict_proba(X_cal_s)[:,1]
    # nonconformity: 1 - p(correct class)
    cal_nc = 1 - np.where(y_cal==1, cal_proba, 1 - cal_proba)

    # Metrics
    test_proba = model.predict_proba(X_test_s)[:,1]
    preds = (test_proba >= CFG.threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, test_proba)

    # Baseline distributions (for drift indicator)
    baseline = X_all.sample(n=min(len(X_all), 2000), random_state=SEED)

    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.shap_explainer = explainer
    st.session_state.conformal_scores = cal_nc
    st.session_state.train_metrics = {"precision": prec, "recall": rec, "f1": f1, "auc": auc}
    st.session_state.baseline_distrib = baseline
    st.session_state.trained = True
    st.session_state.last_train_time = time.time()

if not st.session_state.trained:
    with st.spinner("Training model on synthetic data…"):
        train_model()

# -----------------------------
# Conformal risk
# -----------------------------
def conformal_pvalue(prob1, true_label_guess: int):
    """Return calibrated p-value using stored calibration set.
    For streaming we don't know ground-truth in real-time; use anomaly hypothesis (label=1).
    """
    if st.session_state.conformal_scores is None:
        return None
    # nonconformity for "anomaly" hypothesis (label=1)
    nc = 1 - prob1
    cal = st.session_state.conformal_scores
    pval = (np.sum(cal >= nc) + 1) / (len(cal) + 1)
    return float(pval)

# -----------------------------
# Drift check (simple PSI-like)
# -----------------------------
def psi_simple(baseline: pd.Series, current: pd.Series, bins=10):
    # Population Stability Index
    try:
        cuts = np.quantile(baseline, q=np.linspace(0,1,bins+1))
        cuts = np.unique(cuts)
        if len(cuts) < 3:  # too little variance
            return 0.0
        b_counts, _ = np.histogram(baseline, bins=cuts)
        c_counts, _ = np.histogram(current,  bins=cuts)
        b_rat = (b_counts + 1e-6) / (b_counts.sum() + 1e-6)
        c_rat = (c_counts + 1e-6) / (c_counts.sum() + 1e-6)
        psi = np.sum((c_rat - b_rat) * np.log(c_rat / b_rat))
        return float(psi)
    except Exception:
        return 0.0

def update_drift(latest_feat_row: pd.DataFrame):
    if st.session_state.baseline_distrib is None:
        return
    base = st.session_state.baseline_distrib
    cur = pd.concat([pd.DataFrame([latest_feat_row]),], ignore_index=True)
    drifts = {}
    for col in np.random.choice(feature_cols(), size=10, replace=False): # sample 10 for speed
        try:
            drifts[col] = psi_simple(base[col], cur[col])
        except Exception:
            drifts[col] = 0.0
    st.session_state.drifts = drifts

# -----------------------------
# Streaming step
# -----------------------------
def step_once():
    # define when anomaly starts (for non-Normal)
    if st.session_state.trigger_time is None:
        st.session_state.trigger_time = st.session_state.step + 5  # 5 ticks from start

    row = gen_normal_row()
    if scenario != "Normal" and st.session_state.step >= st.session_state.trigger_time:
        t_since = st.session_state.step - st.session_state.trigger_time + 1
        row = inject_anomaly(row, scenario, t_since)
    st.session_state.raw_buffer.append(row)
    st.session_state.window_buf.append(row)

    # Build features
    feats = build_features(st.session_state.window_buf)
    if not feats:
        st.session_state.step += 1
        return

    X_df = pd.DataFrame([feats]).fillna(0.0)
    X_s  = st.session_state.scaler.transform(X_df)
    prob1 = st.session_state.model.predict_proba(X_s)[:,1][0]
    pval = conformal_pvalue(prob1, true_label_guess=1) if use_conformal else None

    # Decide incident
    fired = prob1 >= CFG.threshold
    incident = None
    if fired:
        # Local explanation (top SHAP)
        shap_vals = st.session_state.shap_explainer.shap_values(X_s)[1][0]  # class 1
        shap_pairs = sorted(list(zip(X_df.columns, shap_vals)), key=lambda x: abs(x[1]), reverse=True)[:5]
        reasons = [{"feature": k, "impact": float(v)} for k, v in shap_pairs]
        incident = {
            "ts": int(time.time()),
            "step": st.session_state.step,
            "scenario": scenario,
            "prob": float(prob1),
            "p_value": float(pval) if pval is not None else None,
            "reasons": reasons,
            "raw": row
        }
        st.session_state.incidents.append(incident)

    # Drift update
    update_drift(X_df.iloc[0].to_dict())

    # last prediction
    st.session_state.last_prediction = {
        "prob": float(prob1),
        "p_value": float(pval) if pval is not None else None,
        "fired": bool(fired),
        "features": feats,
        "raw": row
    }
    st.session_state.step += 1

# Run multiple steps per refresh for “speed”
if auto:
    for _ in range(speed):
        step_once()
else:
    if st.button("Step once"):
        step_once()

# -----------------------------
# KPIs row
# -----------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    last_prob = st.session_state.last_prediction["prob"] if st.session_state.last_prediction else 0.0
    st.metric("Current Risk (prob)", f"{last_prob:0.2f}")
with kpi_cols[1]:
    st.metric("Incidents (session)", len(st.session_state.incidents))
with kpi_cols[2]:
    if st.session_state.train_metrics:
        st.metric("Model AUC", f"{st.session_state.train_metrics.get('auc',0):0.2f}")
with kpi_cols[3]:
    if st.session_state.last_prediction and use_conformal:
        pv = st.session_state.last_prediction["p_value"]
        st.metric("Conformal p-value", f"{pv:0.3f}" if pv is not None else "—")

# -----------------------------
# Tabs: Overview / Details / Insights
# -----------------------------
tab_overview, tab_details, tab_insights = st.tabs(["Overview", "Details", "Insights"])

# ---- Overview (business)
with tab_overview:
    left, right = st.columns([2, 1])

    with left:
        # Live charts for a few key signals
        if len(st.session_state.raw_buffer) > 0:
            df = pd.DataFrame(list(st.session_state.raw_buffer))
            df["t"] = np.arange(len(df))
            fig = go.Figure()
            for y in ["snr", "packet_loss", "latency_ms", "pos_error_m"]:
                fig.add_trace(go.Scatter(x=df["t"], y=df[y], mode="lines", name=y))
            fig.update_layout(height=320, title="Live metrics")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Streaming not started yet.")

        # Incident cards
        st.subheader("Incidents")
        if len(st.session_state.incidents) == 0:
            st.success("No incidents fired yet.")
        else:
            for i, inc in enumerate(reversed(st.session_state.incidents[-10:]), 1):
                with st.expander(f"#{len(st.session_state.incidents)-i+1} • {inc['scenario']} • prob={inc['prob']:.2f} • p={inc['p_value']:.3f if inc['p_value'] else '—'}"):
                    c1, c2 = st.columns([2,1])
                    with c1:
                        reasons_txt = "\n".join([f"- {r['feature']}: {r['impact']:+.3f}" for r in inc["reasons"]])
                        st.markdown(f"**Why flagged (top features)**\n\n{reasons_txt}")
                        st.markdown("**Suggested action**")
                        if inc["scenario"] == "Jamming":
                            st.write("Hop channel, run spectrum scan, consider directional antenna.")
                        elif inc["scenario"] == "GPS Spoofing":
                            st.write("Enable multi-source fusion (Wi-Fi/UWB), verify time source, apply geofence sanity checks.")
                        elif inc["scenario"] == "Wi-Fi Breach":
                            st.write("Quarantine SSID/VLAN, rotate keys, check for rogue APs.")
                        else:
                            st.write("Reject tampered data, verify signatures, audit gateway.")
                    with c2:
                        st.json({"prob": inc["prob"], "p_value": inc["p_value"], "step": inc["step"]})

    with right:
        # Trust widget
        st.subheader("Trust widget")
        if st.session_state.last_prediction and use_conformal:
            pval = st.session_state.last_prediction["p_value"]
            label = "Low risk" if (pval and pval > 0.5) else "High risk"
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max(1e-3, (1 - (pval if pval is not None else 0.0))) * 100,
                title={'text': "Calibrated Risk (%)"},
                gauge={'axis': {'range': [0,100]}}
            ))
            st.plotly_chart(gauge, use_container_width=True)
            st.caption(f"Conformal calibrated @ {int(CFG.coverage*100)}% coverage (lower p-value = higher risk).")
        else:
            st.info("Enable conformal in sidebar to show calibrated risk.")

        # Export report
        if st.session_state.incidents:
            df_inc = pd.DataFrame(st.session_state.incidents)
            # flatten reasons for CSV
            df_inc["top_features"] = df_inc["reasons"].apply(lambda r: "; ".join([f"{x['feature']}:{x['impact']:+.3f}" for x in r]))
            csv = df_inc.drop(columns=["reasons"]).to_csv(index=False).encode("utf-8")
            st.download_button("Download incidents CSV", csv, "incidents.csv", "text/csv")

# ---- Details (operator)
with tab_details:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Current vs Baseline (selected metrics)")
        if len(st.session_state.raw_buffer) > 0:
            df = pd.DataFrame(list(st.session_state.raw_buffer))
            df["t"] = np.arange(len(df))
            for y in ["snr", "packet_loss", "latency_ms", "channel_util"]:
                fig = px.line(df, x="t", y=y, title=y)
                st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### Last decision — local explanation")
        if st.session_state.last_prediction:
            feats = st.session_state.last_prediction["features"]
            X_df = pd.DataFrame([feats]).fillna(0.0)
            X_s = st.session_state.scaler.transform(X_df)
            shap_vals = st.session_state.shap_explainer.shap_values(X_s)[1][0]
            pairs = sorted(list(zip(X_df.columns, shap_vals)), key=lambda x: abs(x[1]), reverse=True)[:8]
            df_pairs = pd.DataFrame(pairs, columns=["feature", "impact"])
            fig = px.bar(df_pairs, x="impact", y="feature", orientation="h", title="Top SHAP contributions (class=anomaly)")
            st.plotly_chart(fig, use_container_width=True)
            # Counterfactual hint (very simple)
            st.caption("Counterfactual hint: reduce the strongest positive contributors to lower risk (e.g., lower packet_loss_z / latency_ms_z).")

# ---- Insights (technical)
with tab_insights:
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("### Global importance (mean |SHAP|)")
        if st.session_state.trained:
            # quick global feature importance via SHAP sampling
            base = st.session_state.baseline_distrib
            base_s = st.session_state.scaler.transform(base)
            shap_vals = st.session_state.shap_explainer.shap_values(base_s)[1]
            mean_abs = np.abs(shap_vals).mean(axis=0)
            imp = pd.DataFrame({"feature": base.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False).head(15)
            fig = px.bar(imp, x="mean_abs_shap", y="feature", orientation="h", title="Global feature impact")
            st.plotly_chart(fig, use_container_width=True)
    with g2:
        st.markdown("### Drift monitor (PSI-like)")
        if st.session_state.drifts:
            df_psi = pd.DataFrame(list(st.session_state.drifts.items()), columns=["feature", "psi"])
            fig = px.bar(df_psi.sort_values("psi", ascending=False).head(10), x="psi", y="feature", orientation="h", title="Potential drift (higher = more drift)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to compute drift yet.")

    st.markdown("### Model card (auto-generated)")
    st.json({
        "model": "LightGBM (depth<=3, trees~60)",
        "training": st.session_state.train_metrics,
        "features": len(feature_cols()),
        "explanations": "SHAP (TreeExplainer)",
        "conformal": "Inductive conformal on validation (p-value displayed)",
        "intended_use": "Demo of trustworthy AI for wireless anomaly detection",
        "limitations": [
            "Synthetic data; thresholds illustrative",
            "Single-device view; multi-device aggregation omitted for brevity"
        ],
        "version": "v1-demo"
    })

# -----------------------------
# Footer refresh note
# -----------------------------
st.caption("Tip: Use the sidebar to change scenario or pause streaming. App auto-retrains on load; adjust thresholds anytime.")
