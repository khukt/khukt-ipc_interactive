import time
from collections import deque
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="TRUST AI — Fast Wireless Threat Demo", layout="wide")
st.title("TRUST AI — Wireless Threat Detection (Ultra-Light Demo)")
st.caption("Fast load • Logistic Regression + Interactions • Coefficient-based XAI • Conformal p-value")

# -----------------------------
# Controls (sidebar)
# -----------------------------
with st.sidebar:
    st.header("Demo Controls")
    scenario = st.selectbox(
        "Scenario", ["Normal", "Jamming", "GPS Spoofing", "Wi-Fi Breach", "Data Tamper"], index=0
    )
    speed = st.slider("Playback speed (x)", 1, 20, 5)
    auto = st.checkbox("Auto stream", value=True)
    reset = st.button("Reset stream")

    st.divider()
    st.subheader("Model / Alerts")
    thr = st.slider("Incident threshold (prob.)", 0.50, 0.95, 0.75, 0.01)
    use_conformal = st.checkbox("Show conformal p-value (calibrated risk)", True)

# -----------------------------
# Globals
# -----------------------------
RAW_FEATURES = [
    "rssi", "snr", "packet_loss", "latency_ms", "jitter_ms",
    "pos_error_m", "auth_fail_rate", "crc_err", "throughput_mbps", "channel_util"
]
ROLL = 5                          # rolling window seconds for features
BUF_MAX = 600                     # max points kept for charts
SEED = 7
np.random.seed(SEED)

def init_state():
    st.session_state.raw = deque(maxlen=BUF_MAX)
    st.session_state.roll = deque(maxlen=ROLL)
    st.session_state.incidents = []
    st.session_state.step = 0
    st.session_state.trigger = None

    # Model containers
    st.session_state.scaler = None
    st.session_state.poly = None
    st.session_state.model = None
    st.session_state.feat_names = None
    st.session_state.cal_nc = None
    st.session_state.metrics = {}
    st.session_state.trained = False

    st.session_state.last_pred = None
    st.session_state.baseline = None  # for quick drift-ish comparisons

if "raw" not in st.session_state or reset:
    init_state()

# -----------------------------
# Synthetic data
# -----------------------------
def gen_normal_row():
    return {
        "rssi": np.random.normal(-65, 3),
        "snr": np.random.normal(25, 2),
        "packet_loss": max(0, np.random.normal(0.5, 0.3)),
        "latency_ms": max(5, np.random.normal(30, 8)),
        "jitter_ms": max(0.3, np.random.normal(2, 0.7)),
        "pos_error_m": max(0.3, np.random.normal(2.5, 0.7)),
        "auth_fail_rate": max(0, np.random.normal(0.2, 0.2)),
        "crc_err": max(0, np.random.poisson(0.2)),
        "throughput_mbps": max(2, np.random.normal(50, 8)),
        "channel_util": np.clip(np.random.normal(35, 8), 0, 100),
    }

def inject(row, scenario, t_since):
    r = row.copy()
    if scenario == "Jamming":
        k = min(1.0, t_since/4)
        r["snr"] -= 8*k + np.random.uniform(0,2)
        r["packet_loss"] += 7*k + np.random.uniform(0,2)
        r["latency_ms"] += 30*k + np.random.uniform(0,8)
        r["jitter_ms"] += 4*k + np.random.uniform(0,1.5)
        r["channel_util"] += 10*k
    elif scenario == "GPS Spoofing":
        k = min(1.0, t_since/4)
        r["pos_error_m"] += 25*k + np.random.uniform(0,8)
        r["jitter_ms"] += 1.5*k
        r["latency_ms"] += 8*k
    elif scenario == "Wi-Fi Breach":
        k = min(1.0, t_since/4)
        r["auth_fail_rate"] += 6*k + np.random.uniform(0,2)
        r["channel_util"] += 18*k + np.random.uniform(0,4)
        r["crc_err"] += np.random.poisson(1 + 2*k)
    elif scenario == "Data Tamper":
        if np.random.rand() < 0.7:
            r["throughput_mbps"] *= np.random.uniform(1.5, 2.0)
        r["crc_err"] += np.random.poisson(2)
        r["packet_loss"] += np.random.uniform(3, 8)
    return r

# -----------------------------
# Feature engineering (fast)
# -----------------------------
def build_features(roll_buf):
    if not roll_buf:
        return {}
    df = pd.DataFrame(list(roll_buf))
    feats = {}
    for f in RAW_FEATURES:
        s = df[f]
        feats[f"{f}_mean"] = float(s.mean())
        feats[f"{f}_std"] = float(s.std(ddof=0)) if len(s) > 1 else 0.0
        feats[f"{f}_last"] = float(s.iloc[-1])
        if len(s) >= 2:
            feats[f"{f}_jump"] = float(s.iloc[-1] - s.iloc[-2])
        else:
            feats[f"{f}_jump"] = 0.0
    return feats

def feat_cols():
    cols = []
    for f in RAW_FEATURES:
        cols += [f"{f}_mean", f"{f}_std", f"{f}_last", f"{f}_jump"]
    return cols

# -----------------------------
# Training (tiny, fast)
# -----------------------------
def synth(seconds, scenario="Normal", label=0):
    rows = []
    for t in range(seconds):
        r = gen_normal_row()
        if scenario != "Normal":
            r = inject(r, scenario, t_since=max(0, t-1))
        rows.append(r)
    df = pd.DataFrame(rows)
    df["label"] = label
    return df

def make_dataset():
    # Keep very small for fast boot
    Nn = 90
    Na = 30
    df = pd.concat([
        synth(Nn, "Normal", 0),
        synth(Na, "Jamming", 1),
        synth(Na, "GPS Spoofing", 1),
        synth(Na, "Wi-Fi Breach", 1),
        synth(Na, "Data Tamper", 1),
    ], ignore_index=True)

    roll = deque(maxlen=ROLL)
    feats_all, y = [], []
    for _, row in df.iterrows():
        raw = {k: row[k] for k in RAW_FEATURES}
        roll.append(raw)
        feats_all.append(build_features(roll))
        y.append(int(row["label"]))
    X = pd.DataFrame(feats_all).fillna(0.0)
    y = np.array(y)
    return X, y

def train_model_fast():
    X, y = make_dataset()
    X = X.fillna(0.0)

    # Split: train, cal, test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)

    scaler = StandardScaler()
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

    X_tr_s = scaler.fit_transform(X_tr)
    X_tr_p = poly.fit_transform(X_tr_s)

    # Tiny, fast model
    model = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=300, class_weight="balanced"
    ).fit(X_tr_p, y_tr)

    # Calibrate conformal nonconformity: 1 - p(correct)
    X_cal_p = poly.transform(scaler.transform(X_cal))
    proba_cal = model.predict_proba(X_cal_p)[:, 1]
    cal_nc = 1 - np.where(y_cal == 1, proba_cal, 1 - proba_cal)

    # Metrics
    X_te_p = poly.transform(scaler.transform(X_te))
    p_te = model.predict_proba(X_te_p)[:, 1]
    auc = roc_auc_score(y_te, p_te)
    preds = (p_te >= 0.75).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, preds, average="binary", zero_division=0)

    st.session_state.scaler = scaler
    st.session_state.poly = poly
    st.session_state.model = model
    st.session_state.feat_names = poly.get_feature_names_out(feat_cols()).tolist()
    st.session_state.cal_nc = cal_nc
    st.session_state.metrics = {"AUC": float(auc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    st.session_state.baseline = X.sample(min(1000, len(X)), random_state=SEED)
    st.session_state.trained = True

if not st.session_state.trained:
    with st.spinner("Training ultra-light model…"):
        train_model_fast()

# -----------------------------
# Conformal p-value
# -----------------------------
def conformal_p(prob1):
    cal = st.session_state.cal_nc
    if cal is None: return None
    nc = 1 - prob1
    return float((np.sum(cal >= nc) + 1) / (len(cal) + 1))

# -----------------------------
# Streaming step
# -----------------------------
def step_once():
    if st.session_state.trigger is None:
        st.session_state.trigger = st.session_state.step + 5  # anomaly starts after 5 ticks

    row = gen_normal_row()
    if scenario != "Normal" and st.session_state.step >= st.session_state.trigger:
        row = inject(row, scenario, t_since=st.session_state.step - st.session_state.trigger + 1)

    st.session_state.raw.append(row)
    st.session_state.roll.append(row)

    feats = build_features(st.session_state.roll)
    if not feats:
        st.session_state.step += 1
        return

    X1 = pd.DataFrame([feats]).fillna(0.0)
    X1_p = st.session_state.poly.transform(st.session_state.scaler.transform(X1))
    prob1 = float(st.session_state.model.predict_proba(X1_p)[0, 1])
    pval = conformal_p(prob1) if use_conformal else None
    fired = prob1 >= thr

    # Local explanation (coef * feature value)
    coefs = st.session_state.model.coef_[0]
    contribs = coefs * X1_p[0]
    # Map back to names
    names = st.session_state.feat_names
    pairs = sorted(zip(names, contribs), key=lambda x: abs(x[1]), reverse=True)[:8]
    reasons = [{"feature": n, "impact": float(v)} for n, v in pairs]

    if fired:
        st.session_state.incidents.append({
            "ts": int(time.time()),
            "step": int(st.session_state.step),
            "scenario": scenario,
            "prob": prob1,
            "p_value": pval,
            "reasons": reasons,
            "raw": row
        })

    st.session_state.last_pred = {
        "prob": prob1, "p_value": pval, "fired": fired,
        "reasons": reasons, "raw": row
    }
    st.session_state.step += 1

# Run steps per refresh
if auto:
    for _ in range(speed):
        step_once()
else:
    if st.button("Step once"):
        step_once()

# -----------------------------
# KPI row
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Current Risk (prob)", f"{(st.session_state.last_pred or {}).get('prob', 0.0):.2f}")
with c2:
    st.metric("Incidents (session)", len(st.session_state.incidents))
with c3:
    st.metric("Model AUC", f"{st.session_state.metrics.get('AUC', 0.0):.2f}")
with c4:
    pv = (st.session_state.last_pred or {}).get("p_value", None)
    st.metric("Conformal p-value", f"{pv:.3f}" if pv is not None else "—")

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_details, tab_insights = st.tabs(["Overview", "Details", "Insights"])

# ---- Overview
with tab_overview:
    left, right = st.columns([2,1])
    with left:
        st.subheader("Live Metrics")
        if len(st.session_state.raw) > 0:
            df = pd.DataFrame(list(st.session_state.raw))
            df["t"] = np.arange(len(df))
            chart = alt.Chart(df).transform_fold(
                ["snr", "packet_loss", "latency_ms", "pos_error_m"],
                as_=["metric","value"]
            ).mark_line().encode(
                x="t:Q", y="value:Q", color="metric:N"
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Waiting for stream…")

        st.subheader("Incidents")
        if not st.session_state.incidents:
            st.success("No incidents yet.")
        else:
            for i, inc in enumerate(reversed(st.session_state.incidents[-10:]), 1):
                with st.expander(f"#{len(st.session_state.incidents)-i+1} • {inc['scenario']} • prob={inc['prob']:.2f} • p={inc['p_value']:.3f if inc['p_value'] else '—'}"):
                    colA, colB = st.columns([2,1])
                    with colA:
                        txt = "\n".join([f"- {r['feature']}: {r['impact']:+.3f}" for r in inc["reasons"]])
                        st.markdown("**Why flagged (top contributions)**\n\n" + txt)
                        st.markdown("**Suggested action**")
                        if inc["scenario"] == "Jamming":
                            st.write("Hop channel; run spectrum scan; consider directional antenna.")
                        elif inc["scenario"] == "GPS Spoofing":
                            st.write("Enable multi-source fusion (Wi-Fi/UWB); verify time source; apply geofence checks.")
                        elif inc["scenario"] == "Wi-Fi Breach":
                            st.write("Quarantine SSID/VLAN; rotate keys; check for rogue APs.")
                        else:
                            st.write("Reject tampered data; verify signatures; audit gateway.")
                    with colB:
                        st.json({"prob": inc["prob"], "p_value": inc["p_value"], "step": inc["step"]})

    with right:
        st.subheader("Trust widget")
        if st.session_state.last_pred and use_conformal:
            p = st.session_state.last_pred["p_value"]
            st.metric("Calibrated risk (%)", f"{(1 - (p or 0))*100:.1f}")
            st.caption("Lower p-value = higher risk (inductive conformal).")
        else:
            st.info("Enable conformal in the sidebar.")
        if st.session_state.incidents:
            df_inc = pd.DataFrame(st.session_state.incidents)
            df_inc["top_features"] = df_inc["reasons"].apply(lambda r: "; ".join([f"{x['feature']}:{x['impact']:+.3f}" for x in r]))
            csv = df_inc.drop(columns=["reasons"]).to_csv(index=False).encode("utf-8")
            st.download_button("Download incidents CSV", csv, "incidents.csv", "text/csv")

# ---- Details (operator)
with tab_details:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Selected metrics")
        if len(st.session_state.raw) > 0:
            df = pd.DataFrame(list(st.session_state.raw))
            df["t"] = np.arange(len(df))
            for y in ["snr", "packet_loss", "latency_ms", "channel_util"]:
                st.line_chart(df.set_index("t")[y], height=160)
    with col2:
        st.markdown("### Last decision — local explanation")
        lp = st.session_state.last_pred
        if lp:
            df_pairs = pd.DataFrame(lp["reasons"], columns=["feature","impact"])
            bar = alt.Chart(df_pairs).mark_bar().encode(
                x=alt.X("impact:Q", title="Contribution (coef × value)"),
                y=alt.Y("feature:N", sort="-x", title="Feature term")
            ).properties(height=260)
            st.altair_chart(bar, use_container_width=True)
            st.caption("Counterfactual hint: reducing the largest positive contributors would lower the alert probability.")

# ---- Insights (technical)
with tab_insights:
    st.markdown("### Global importance (mean |contribution| on baseline)")
    if st.session_state.baseline is not None:
        base = st.session_state.baseline.fillna(0.0)
        base_p = st.session_state.poly.transform(st.session_state.scaler.transform(base))
        coefs = st.session_state.model.coef_[0]
        contrib = np.abs(base_p * coefs).mean(axis=0)
        names = st.session_state.feat_names
        imp = pd.DataFrame({"feature": names, "mean_abs_contrib": contrib}).sort_values("mean_abs_contrib", ascending=False).head(15)
        bar = alt.Chart(imp).mark_bar().encode(
            x="mean_abs_contrib:Q", y=alt.Y("feature:N", sort="-x")
        ).properties(height=320)
        st.altair_chart(bar, use_container_width=True)

    st.markdown("### Model card")
    st.json({
        "model": "Logistic Regression with interaction terms (degree=2, interaction_only=True)",
        "training": st.session_state.metrics,
        "features": len(st.session_state.feat_names or []),
        "explanations": "Coefficient × feature value (local & global)",
        "conformal": "Inductive conformal p-value on validation split",
        "intended_use": "Demo of trustworthy AI for wireless anomaly detection",
        "limitations": [
            "Synthetic data; thresholds illustrative",
            "Single-device stream for simplicity"
        ],
        "version": "v2-ultralight"
    })

st.caption("Tip: Switch scenario in the sidebar; this build is optimized for Streamlit Cloud (no Plotly/SHAP).")
