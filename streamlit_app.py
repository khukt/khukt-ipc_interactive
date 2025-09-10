import time
from collections import deque
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

# -----------------------------
# Demo constants & schema Version 0.0.1
# -----------------------------
RAW_FEATURES = [
    "rssi", "snr", "packet_loss", "latency_ms", "jitter_ms",
    "pos_error_m", "auth_fail_rate", "crc_err", "throughput_mbps", "channel_util"
]
ROLLING_LEN = 5  # ~5s short rolling window
SEED = 42
np.random.seed(SEED)

@dataclass
class Config:
    threshold: float = 0.75
    coverage: float = 0.90
    retrain_every_s: int = 120
    train_seconds: int = 120
    cal_seconds: int = 30
    eval_seconds: int = 30
    max_points_plot: int = 600
    depth: int = 3
    n_estimators: int = 60
    learning_rate: float = 0.08

CFG = Config()

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="TRUST AI: Wireless Threat Detection (Demo)", layout="wide")
st.title("TRUST AI — Wireless Threat Detection (Demo)")
st.caption("LightGBM + SHAP + Conformal Risk • Persona-based XAI (End Users • Domain Experts • Regulators • AI Builders • Executives)")

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
# Small helpers
# -----------------------------
def to_df(X_array, columns):
    """Wrap a numpy array with the original feature names as a DataFrame."""
    return pd.DataFrame(X_array, columns=columns)

def shap_pos_class_values(explainer, X_df):
    """Return SHAP values for the anomaly (positive) class, robust across SHAP versions."""
    vals = explainer.shap_values(X_df)
    if isinstance(vals, list):
        use = vals[1] if len(vals) > 1 else vals[0]
        return use  # (n_samples, n_features)
    else:
        return vals   # (n_samples, n_features)

def severity_from(prob, pval):
    """Map model prob & conformal p-value to a severity label and color."""
    high = (prob >= 0.85) or (pval is not None and pval <= 0.05)
    med  = (prob >= 0.70) or (pval is not None and pval <= 0.20)
    if high:   return "High", "red"
    if med:    return "Medium", "orange"
    return "Low", "green"

def bullet(text):
    return f"- {text}"

def scenario_actions(scen):
    if scen == "Jamming":
        return ["Channel hop", "Quick spectrum scan", "Enable directional/backup link"]
    if scen == "GPS Spoofing":
        return ["Switch to multi-source positioning", "Verify time source", "Apply geofence sanity checks"]
    if scen == "Wi-Fi Breach":
        return ["Quarantine SSID/VLAN", "Rotate keys", "Rogue AP scan"]
    return ["Reject bad packets", "Verify signatures", "Audit gateway"]

# -----------------------------
# Session state init
# -----------------------------
def init_state():
    st.session_state.raw_buffer = deque(maxlen=CFG.max_points_plot)   # raw metrics stream
    st.session_state.window_buf = deque(maxlen=ROLLING_LEN)           # last ~5s of raw
    st.session_state.incidents = []
    st.session_state.step = 0
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.shap_explainer = None
    st.session_state.conformal_scores = None  # calibration nonconformity
    st.session_state.train_metrics = {}
    st.session_state.last_train_time = time.time()
    st.session_state.drifts = {}
    st.session_state.baseline_distrib = None  # standardized feature baseline
    st.session_state.trained = False
    st.session_state.last_prediction = None
    st.session_state.trigger_time = None

if "raw_buffer" not in st.session_state or reset_btn:
    init_state()

# -----------------------------
# Synthetic data generator
# -----------------------------
def gen_normal_row():
    return {
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

def inject_anomaly(row, scenario, t_since):
    r = row.copy()
    if scenario == "Jamming":
        strength = min(1.0, t_since / 5.0)
        r["snr"] -= 8 * strength + np.random.uniform(0, 2)
        r["packet_loss"] += 8 * strength + np.random.uniform(0, 2)
        r["latency_ms"] += 40 * strength + np.random.uniform(0, 10)
        r["jitter_ms"] += 5 * strength + np.random.uniform(0, 2)
        r["channel_util"] += 10 * strength
    elif scenario == "GPS Spoofing":
        strength = min(1.0, t_since / 5.0)
        r["pos_error_m"] += 30 * strength + np.random.uniform(0, 10)
        r["jitter_ms"] += 2 * strength
        r["latency_ms"] += 10 * strength
    elif scenario == "Wi-Fi Breach":
        strength = min(1.0, t_since / 5.0)
        r["auth_fail_rate"] += 8 * strength + np.random.uniform(0, 2)
        r["channel_util"] += 20 * strength + np.random.uniform(0, 5)
        r["crc_err"] += np.random.poisson(1 + 3 * strength)
    elif scenario == "Data Tamper":
        if np.random.rand() < 0.6:
            r["throughput_mbps"] *= np.random.uniform(1.5, 2.2)
        r["crc_err"] += np.random.poisson(2)
        r["packet_loss"] += np.random.uniform(3, 10)
    return r

# -----------------------------
# Feature engineering
# -----------------------------
def build_features(buf: deque):
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
        if len(series) >= 3:
            x = np.arange(len(series[-3:]))
            y = series[-3:].to_numpy()
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0
        feats[f"{feat}_slope"] = slope
        mu, sd = feats[f"{feat}_mean_5s"], feats[f"{feat}_std_5s"]
        z = 0.0 if sd == 0 else (series.iloc[-1] - mu) / sd
        feats[f"{feat}_z"] = z
        feats[f"{feat}_jump"] = float(series.iloc[-1] - series.iloc[-2]) if len(series) >= 2 else 0.0
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
    for s in range(seconds):
        r = gen_normal_row()
        if scenario != "Normal":
            r = inject_anomaly(r, scenario, t_since=max(0, s - 1))
        rows.append(r)
    df = pd.DataFrame(rows)
    df["label"] = label
    return df

def make_training_data():
    nN = CFG.train_seconds
    nA = CFG.train_seconds // 3
    df = pd.concat([
        synth_block(nN, "Normal", 0),
        synth_block(nA, "Jamming", 1),
        synth_block(nA, "GPS Spoofing", 1),
        synth_block(nA, "Wi-Fi Breach", 1),
        synth_block(nA, "Data Tamper", 1),
    ], ignore_index=True)

    feats_all, labels = [], []
    buf = deque(maxlen=ROLLING_LEN)
    for _, row in df.iterrows():
        raw = {k: row[k] for k in RAW_FEATURES}
        buf.append(raw)
        feats = build_features(buf)
        feats_all.append(feats)
        labels.append(row["label"])

    X = pd.DataFrame(feats_all).fillna(0.0)
    y = np.array(labels)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.40, shuffle=True, random_state=SEED, stratify=y)
    X_cal, X_test, y_cal, y_test   = train_test_split(X_tmp, y_tmp, test_size=0.50, shuffle=True, random_state=SEED, stratify=y_tmp)
    return (X_train, y_train), (X_cal, y_cal), (X_test, y_test), X

def train_model():
    (X_train, y_train), (X_cal, y_cal), (X_test, y_test), X_all = make_training_data()
    feat_cols = list(X_train.columns)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s   = scaler.transform(X_cal)
    X_test_s  = scaler.transform(X_test)
    X_all_s   = scaler.transform(X_all)

    # Fit & predict with DataFrames to keep feature names
    X_train_s_df = to_df(X_train_s, feat_cols)
    X_cal_s_df   = to_df(X_cal_s,   feat_cols)
    X_test_s_df  = to_df(X_test_s,  feat_cols)
    X_all_s_df   = to_df(X_all_s,   feat_cols)

    model = LGBMClassifier(
        n_estimators=CFG.n_estimators,
        max_depth=CFG.depth,
        learning_rate=CFG.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        min_data_in_leaf=10,
        force_col_wise=True,
        random_state=SEED
    )
    model.fit(X_train_s_df, y_train)

    explainer = shap.TreeExplainer(model)

    # Conformal calibration (nonconformity = 1 - p(correct))
    cal_proba = model.predict_proba(X_cal_s_df)[:, 1]
    cal_nc = 1 - np.where(y_cal == 1, cal_proba, 1 - cal_proba)

    # Metrics
    test_proba = model.predict_proba(X_test_s_df)[:, 1]
    preds = (test_proba >= CFG.threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, test_proba)

    # Baseline (standardized features) for drift
    baseline = X_all_s_df.sample(n=min(len(X_all_s_df), 2000), random_state=SEED)

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
def conformal_pvalue(prob1, true_label_guess: int = 1):
    """Return calibrated p-value using stored calibration set (anomaly hypothesis)."""
    if st.session_state.conformal_scores is None:
        return None
    nc = 1 - prob1  # anomaly hypothesis
    cal = st.session_state.conformal_scores
    pval = (np.sum(cal >= nc) + 1) / (len(cal) + 1)
    return float(pval)

# -----------------------------
# Drift check (simple PSI-like)
# -----------------------------
def psi_simple(baseline: pd.Series, current: pd.Series, bins=10):
    try:
        cuts = np.quantile(baseline, q=np.linspace(0, 1, bins + 1))
        cuts = np.unique(cuts)
        if len(cuts) < 3:
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
    cur = pd.concat([pd.DataFrame([latest_feat_row])], ignore_index=True)
    drifts = {}
    # sample 10 features for speed
    for col in np.random.choice(base.columns, size=min(10, len(base.columns)), replace=False):
        try:
            drifts[col] = psi_simple(base[col], cur[col])
        except Exception:
            drifts[col] = 0.0
    st.session_state.drifts = drifts

# -----------------------------
# Streaming step
# -----------------------------
def step_once():
    if st.session_state.trigger_time is None:
        st.session_state.trigger_time = st.session_state.step + 5

    row = gen_normal_row()
    if scenario != "Normal" and st.session_state.step >= st.session_state.trigger_time:
        t_since = st.session_state.step - st.session_state.trigger_time + 1
        row = inject_anomaly(row, scenario, t_since)

    st.session_state.raw_buffer.append(row)
    st.session_state.window_buf.append(row)

    feats = build_features(st.session_state.window_buf)
    if not feats:
        st.session_state.step += 1
        return

    X_df = pd.DataFrame([feats]).fillna(0.0)
    # standardize and wrap as DF with names
    X_s = st.session_state.scaler.transform(X_df)
    X_s_df = to_df(X_s, X_df.columns)

    prob1 = st.session_state.model.predict_proba(X_s_df)[:, 1][0]
    pval = conformal_pvalue(prob1) if use_conformal else None
    fired = prob1 >= CFG.threshold

    if fired:
        shap_mat = shap_pos_class_values(st.session_state.shap_explainer, X_s_df)
        shap_vec = shap_mat[0]
        shap_pairs = sorted(list(zip(X_s_df.columns, shap_vec)), key=lambda x: abs(x[1]), reverse=True)[:5]
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

    # drift uses standardized space to match baseline
    update_drift(to_df(st.session_state.scaler.transform(X_df), X_df.columns).iloc[0].to_dict())

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
        st.metric("Model AUC", f"{st.session_state.train_metrics.get('auc', 0):0.2f}")
with kpi_cols[3]:
    if st.session_state.last_prediction and use_conformal:
        pv = st.session_state.last_prediction["p_value"]
        st.metric("Conformal p-value", f"{pv:0.3f}" if pv is not None else "—")

# -----------------------------
# Tabs: Overview / Details / Insights
# -----------------------------
tab_overview, tab_details, tab_insights = st.tabs(["Overview", "Details", "Insights"])

# ---- Overview (persona-focused incidents)
with tab_overview:
    left, right = st.columns([2, 1])

    with left:
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

        st.subheader("Incidents")
        if len(st.session_state.incidents) == 0:
            st.success("No incidents fired yet.")
        else:
            for i, inc in enumerate(reversed(st.session_state.incidents[-10:]), 1):
                pv = inc.get("p_value", None)
                pv_str = f"{pv:.3f}" if pv is not None else "—"
                sev, sev_color = severity_from(inc["prob"], pv)
                title = f"#{len(st.session_state.incidents)-i+1} • {inc['scenario']} • prob={inc['prob']:.2f} • p={pv_str} • Severity={sev}"
                with st.expander(title):
                    badge = f"<span style='background-color:{sev_color}; color:white; padding:2px 8px; border-radius:8px;'>{sev}</span>"
                    st.markdown(f"**Severity**: {badge}", unsafe_allow_html=True)

                    c1, c2 = st.columns([2, 1])
                    with c1:
                        tabs = st.tabs([
                            "End Users",
                            "Domain Experts",
                            "Regulators",
                            "AI Builders",
                            "Executives"
                        ])

                        # ------- End Users Impacted by Decisions
                        with tabs[0]:
                            st.markdown("#### What happened (simple)")
                            st.write(f"The system noticed behavior that looks like **{inc['scenario']}**.")
                            st.markdown("#### What it means for me")
                            if inc["scenario"] == "Jamming":
                                st.write("You might experience slower connection or brief dropouts.")
                            elif inc["scenario"] == "GPS Spoofing":
                                st.write("Your location on the map may be wrong for a short time.")
                            elif inc["scenario"] == "Wi-Fi Breach":
                                st.write("An unknown device may be trying to use this network.")
                            else:
                                st.write("Some data looks inconsistent and might be rejected.")
                            st.markdown("#### What to do now")
                            actions = scenario_actions(inc["scenario"])
                            st.write(" • " + " • ".join(actions[:2]))
                            if pv is not None:
                                st.caption(f"Confidence (lower = riskier): p-value = {pv_str}")

                        # ------- Domain Experts & Practitioners
                        with tabs[1]:
                            st.markdown("#### Operational assessment")
                            st.write(bullet(f"Severity **{sev}** (prob={inc['prob']:.2f}, p={pv_str})"))
                            st.write(bullet("Likely root-cause region: RF interference / auth-plane / positioning stack (by scenario)."))
                            st.markdown("#### Recommended playbook (priority)")
                            for idx, a in enumerate(scenario_actions(inc["scenario"]), 1):
                                st.write(f"{idx}. {a}")
                            st.markdown("#### Key observables")
                            obs = []
                            for r in inc["reasons"]:
                                obs.append(f"{r['feature']} ({r['impact']:+.3f})")
                            st.write(" • " + " • ".join(obs))

                        # ------- Regulatory Authorities
                        with tabs[2]:
                            st.markdown("#### Assurance & governance")
                            st.write(bullet("Decision transparency: model shows top contributing factors and plain-language summary."))
                            st.write(bullet(f"Calibrated confidence: conformal p-value = {pv_str} (target coverage {int(CFG.coverage*100)}%)."))
                            st.write(bullet("Audit trail: downloadable incident evidence with model version & inputs."))
                            st.write(bullet("Data scope in this demo: technical telemetry only (no personal data)."))
                            st.write(bullet("Ongoing monitoring: distribution drift checks (see Insights ▸ Drift)."))
                            evidence = {
                                "ts": inc["ts"],
                                "step": inc["step"],
                                "scenario": inc["scenario"],
                                "severity": sev,
                                "prob": inc["prob"],
                                "p_value": inc["p_value"],
                                "model_version": "LightGBM v1.3-demo",
                                "explanations": inc["reasons"],
                                "raw_sample": inc["raw"]
                            }
                            st.download_button(
                                "Download incident evidence (JSON)",
                                data=json.dumps(evidence, indent=2).encode("utf-8"),
                                file_name=f"incident_{inc['ts']}.json",
                                mime="application/json",
                                key=f"dl_evidence_{inc['ts']}_{inc['step']}"   # <-- unique key per incident
                            )

                        # ------- AI Builders, Software Engineers, Researchers & Innovators
                        with tabs[3]:
                            st.markdown("#### Local technical explanation (SHAP)")
                            reasons_txt = "\n".join([f"- {r['feature']}: {r['impact']:+.3f}" for r in inc["reasons"]])
                            st.markdown(reasons_txt)
                            st.caption("Positive impact pushes toward 'anomaly'. Values in standardized feature space.")
                            st.markdown("#### Implementation details")
                            st.write(bullet("Model: LightGBM (depth≤3, ~60 trees), threshold on predicted prob."))
                            st.write(bullet("Calibration: inductive conformal p-value for risk display."))
                            st.write(bullet("Features: short-window means/std/gradients/z-scores over network/position metrics."))

                        # ------- Business Leaders & Executives
                        with tabs[4]:
                            st.markdown("#### Executive summary")
                            st.write(bullet(f"Severity **{sev}**; scenario: **{inc['scenario']}**."))
                            st.write(bullet("Potential impact: short-term service degradation or location inaccuracy; mitigations available."))
                            st.markdown("#### Decision & ROI")
                            st.write(bullet("Immediate action recommended (minutes-level) to reduce risk of SLA breach."))
                            st.write(bullet("Mitigation costs are low (config changes/scans); high benefit by avoiding downtime."))
                            st.markdown("#### KPIs to watch")
                            st.write(bullet("Incidents today, Mean Time To Detect (MTTD), Packet loss, Latency, SNR"))
                            st.caption("This demo uses synthetic data; numbers illustrate the workflow rather than real costs.")

                    with c2:
                        st.json({
                            "prob": inc["prob"],
                            "p_value": inc["p_value"],
                            "step": inc["step"],
                            "scenario": inc["scenario"]
                        })

    with right:
        st.subheader("Trust widget")
        if st.session_state.last_prediction and use_conformal:
            pval = st.session_state.last_prediction["p_value"]
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max(1e-3, (1 - (pval if pval is not None else 0.0))) * 100,
                title={'text': "Calibrated Risk (%)"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(gauge, use_container_width=True)
            st.caption(f"Conformal calibrated @ {int(CFG.coverage*100)}% coverage (lower p-value = higher risk).")
        else:
            st.info("Enable conformal in sidebar to show calibrated risk.")

        if st.session_state.incidents:
            df_inc = pd.DataFrame(st.session_state.incidents)
            df_inc["top_features"] = df_inc["reasons"].apply(
                lambda r: "; ".join([f"{x['feature']}:{x['impact']:+.3f}" for x in r])
            )
            csv = df_inc.drop(columns=["reasons"]).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download incidents CSV",
                csv,
                "incidents.csv",
                "text/csv",
                key="dl_incidents_csv"  # <-- stable unique key
            )

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
        st.markdown("### Last decision — local explanation (technical)")
        if st.session_state.last_prediction:
            feats = st.session_state.last_prediction["features"]
            X_df = pd.DataFrame([feats]).fillna(0.0)
            X_s = st.session_state.scaler.transform(X_df)
            X_s_df = to_df(X_s, X_df.columns)

            shap_mat = shap_pos_class_values(st.session_state.shap_explainer, X_s_df)
            shap_vec = shap_mat[0]
            pairs = sorted(list(zip(X_s_df.columns, shap_vec)), key=lambda x: abs(x[1]), reverse=True)[:8]
            df_pairs = pd.DataFrame(pairs, columns=["feature", "impact"])
            fig = px.bar(df_pairs, x="impact", y="feature", orientation="h",
                         title="Top SHAP contributions (class=anomaly)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Counterfactual hint: reduce the strongest positive contributors to lower risk (e.g., lower packet_loss_z / latency_ms_z).")

# ---- Insights (technical)
with tab_insights:
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("### Global importance (mean |SHAP|)")
        if st.session_state.trained:
            base = st.session_state.baseline_distrib  # standardized DF
            shap_mat = shap_pos_class_values(st.session_state.shap_explainer, base)
            mean_abs = np.abs(shap_mat).mean(axis=0)
            imp = pd.DataFrame({"feature": base.columns, "mean_abs_shap": mean_abs}).sort_values(
                "mean_abs_shap", ascending=False
            ).head(15)
            fig = px.bar(imp, x="mean_abs_shap", y="feature", orientation="h", title="Global feature impact")
            st.plotly_chart(fig, use_container_width=True)
    with g2:
        st.markdown("### Drift monitor (PSI-like)")
        if st.session_state.drifts:
            df_psi = pd.DataFrame(list(st.session_state.drifts.items()), columns=["feature", "psi"])
            fig = px.bar(
                df_psi.sort_values("psi", ascending=False).head(10),
                x="psi", y="feature", orientation="h", title="Potential drift (higher = more drift)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to compute drift yet.")

    st.markdown("### Model card (auto-generated)")
    st.json({
        "model": "LightGBM (depth<=3, trees~60)",
        "training": st.session_state.train_metrics,
        "features": len(feature_cols()),
        "explanations": "SHAP (TreeExplainer, binary-robust)",
        "conformal": "Inductive conformal on validation (p-value displayed)",
        "intended_use": "Demo of trustworthy AI for wireless anomaly detection",
        "limitations": [
            "Synthetic data; thresholds illustrative",
            "Single-device view; multi-device aggregation omitted for brevity"
        ],
        "version": "LightGBM v1.3-demo"
    })

st.caption("Tip: Each incident now includes persona-specific explanations (End Users • Domain Experts • Regulators • AI Builders • Executives).")
