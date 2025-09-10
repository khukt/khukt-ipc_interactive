# ---- ultra-early profiler for import time ----
import time as _t
__APP_T0 = _t.perf_counter()  # very first line
__PERF = {}

def _perf_set(key, seconds):
    __PERF[key] = round(seconds * 1000.0, 2)  # ms

# ------------------- imports (minimal) -------------------
import sys
from collections import deque
import numpy as np
import pandas as pd
import streamlit as st
_perf_set("imports_ms", _t.perf_counter() - __APP_T0)

# ------------------- app config -------------------
_t1 = _t.perf_counter()
st.set_page_config(page_title="TRUST AI — Instant Demo (Profiled)", layout="wide")
st.title("TRUST AI — Wireless Threat Detection (Instant Demo, Profiled)")
st.caption("Zero training • Tiny Logistic Model • Conformal p-value • Built-in profiler")

# ------------------- sidebar (measure) -------------------
_sb_t0 = _t.perf_counter()
with st.sidebar:
    st.header("Demo Controls")
    scenario = st.selectbox("Scenario",
                            ["Normal", "Jamming", "GPS Spoofing", "Wi-Fi Breach", "Data Tamper"], index=0)
    speed = st.slider("Steps per refresh", 1, 50, 8)
    auto = st.checkbox("Auto stream", True)
    thr = st.slider("Incident threshold (prob.)", 0.50, 0.95, 0.75, 0.01)
    st.divider()
    st.subheader("Performance")
    max_points = st.slider("Chart history (points)", 200, 2000, 600, 100,
                           help="Lower = faster")
    show_perf = st.checkbox("Show Performance Debug", True)
    reset = st.button("Reset stream")
_perf_set("sidebar_build_ms", _t.perf_counter() - _sb_t0)

# ------------------- session init (measure) -------------------
_init_t0 = _t.perf_counter()
np.random.seed(3)
RAW = ["rssi","snr","packet_loss","latency_ms","jitter_ms",
       "pos_error_m","auth_fail_rate","crc_err","throughput_mbps","channel_util"]
ROLL = 5  # rolling seconds for features

def init_state():
    st.session_state.raw = deque(maxlen=max_points)
    st.session_state.roll = deque(maxlen=ROLL)
    st.session_state.step = 0
    st.session_state.trigger = None
    st.session_state.incidents = []
    st.session_state.last_pred = None
    # conformal calibration store
    st.session_state.cal_nc = []
    # global importance accumulator
    st.session_state.global_abs_contrib_sum = {}
    st.session_state.global_abs_contrib_count = 0
    # rolling perf stats
    st.session_state.step_times = deque(maxlen=300)  # ms

if "raw" not in st.session_state or reset:
    init_state()
_perf_set("init_state_ms", _t.perf_counter() - _init_t0)

# ------------------- synthetic stream, features -------------------
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
    k = min(1.0, t_since/4)  # smooth ramp
    if scenario == "Jamming":
        r["snr"] -= 8*k + np.random.uniform(0,2)
        r["packet_loss"] += 7*k + np.random.uniform(0,2)
        r["latency_ms"] += 30*k + np.random.uniform(0,8)
        r["jitter_ms"] += 4*k + np.random.uniform(0,1.5)
        r["channel_util"] += 10*k
    elif scenario == "GPS Spoofing":
        r["pos_error_m"] += 25*k + np.random.uniform(0,8)
        r["jitter_ms"] += 1.5*k
        r["latency_ms"] += 8*k
    elif scenario == "Wi-Fi Breach":
        r["auth_fail_rate"] += 6*k + np.random.uniform(0,2)
        r["channel_util"] += 18*k + np.random.uniform(0,4)
        r["crc_err"] += np.random.poisson(1 + 2*k)
    elif scenario == "Data Tamper":
        if np.random.rand() < 0.7:
            r["throughput_mbps"] *= np.random.uniform(1.5, 2.0)
        r["crc_err"] += np.random.poisson(2)
        r["packet_loss"] += np.random.uniform(3, 8)
    return r

def build_features(roll_buf):
    if not roll_buf:
        return {}
    df = pd.DataFrame(list(roll_buf))
    feats = {}
    for f in RAW:
        s = df[f]
        last = float(s.iloc[-1])
        feats[f"{f}_mean"] = float(s.mean())
        feats[f"{f}_std"]  = float(s.std(ddof=0)) if len(s) > 1 else 0.0
        feats[f"{f}_last"] = last
        feats[f"{f}_jump"] = float(last - s.iloc[-2]) if len(s) >= 2 else 0.0
    # interactions (cheap nonlinearity)
    feats["snr_x_packet_loss"] = feats.get("snr_last",0)*feats.get("packet_loss_last",0)
    feats["latency_x_jitter"]  = feats.get("latency_ms_last",0)*feats.get("jitter_ms_last",0)
    feats["poserr_x_latency"]  = feats.get("pos_error_m_last",0)*feats.get("latency_ms_last",0)
    return feats

def feat_names():
    cols = []
    for f in RAW:
        cols += [f"{f}_mean", f"{f}_std", f"{f}_last", f"{f}_jump"]
    cols += ["snr_x_packet_loss","latency_x_jitter","poserr_x_latency"]
    return cols

# ------------------- tiny “AI” model (pre-seeded logistic) -------------------
W = {k: 0.0 for k in feat_names()}
B = -3.2
W.update({
    "snr_last":           -0.12,
    "packet_loss_last":    0.30,
    "latency_ms_last":     0.015,
    "jitter_ms_last":      0.10,
    "pos_error_m_last":    0.08,
    "auth_fail_rate_last": 0.35,
    "crc_err_last":        0.12,
    "channel_util_last":   0.010,
    "throughput_mbps_last":-0.006,
    "snr_mean":           -0.05,
    "packet_loss_mean":    0.12,
    "latency_ms_mean":     0.007,
    "jitter_ms_mean":      0.05,
    "pos_error_m_mean":    0.05,
    "auth_fail_rate_mean": 0.18,
    "snr_jump":           -0.04,
    "packet_loss_jump":    0.08,
    "latency_ms_jump":     0.02,
    "jitter_ms_jump":      0.05,
    "pos_error_m_jump":    0.04,
    "snr_x_packet_loss":   0.012,
    "latency_x_jitter":    0.003,
    "poserr_x_latency":    0.002,
})
def _sigmoid(z): return 1.0/(1.0+np.exp(-z))
def predict_proba(feat_row: dict):
    z = B
    contribs = {}
    for k, w in W.items():
        v = feat_row.get(k, 0.0)
        z += w * v
        contribs[k] = w * v
    return float(_sigmoid(z)), contribs

# ------------------- conformal p-value -------------------
def conformal_p(prob_anom: float):
    cal = st.session_state.cal_nc
    if not cal:
        return None
    nc = 1.0 - prob_anom  # nonconformity for anomaly hypothesis
    cal_arr = np.asarray(cal)
    return float((np.sum(cal_arr >= nc) + 1) / (len(cal_arr) + 1))

# ------------------- streaming step (profiled) -------------------
def step_once():
    t0 = _t.perf_counter()
    if st.session_state.trigger is None:
        st.session_state.trigger = st.session_state.step + 5

    row = gen_normal_row()
    if scenario != "Normal" and st.session_state.step >= st.session_state.trigger:
        row = inject(row, scenario, t_since=st.session_state.step - st.session_state.trigger + 1)

    st.session_state.raw.append(row)
    st.session_state.roll.append(row)

    feats = build_features(st.session_state.roll)
    if feats:
        prob1, contribs = predict_proba(feats)
        fired = (prob1 >= thr)
        # collect calibration early during normal
        if scenario == "Normal" and st.session_state.step < 40:
            st.session_state.cal_nc.append(1.0 - prob1)
        pval = conformal_p(prob1)
        if fired:
            top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            st.session_state.incidents.append({
                "ts": int(_t.time()),
                "step": int(st.session_state.step),
                "scenario": scenario,
                "prob": float(prob1),
                "p_value": pval,
                "reasons": [{"feature": k, "impact": float(v)} for k, v in top],
                "raw": row
            })
        # global importance accumulator
        for k, v in contribs.items():
            st.session_state.global_abs_contrib_sum[k] = st.session_state.global_abs_contrib_sum.get(k, 0.0) + abs(v)
        st.session_state.global_abs_contrib_count += 1
        st.session_state.last_pred = {"prob": prob1, "p_value": pval, "fired": fired,
                                      "reasons": sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:8],
                                      "raw": row}
    st.session_state.step += 1
    # record step time
    st.session_state.step_times.append(round((_t.perf_counter() - t0) * 1000.0, 2))

# run steps this refresh
_run_t0 = _t.perf_counter()
for _ in range(speed if auto else 1):
    step_once()
_perf_set("stream_steps_total_ms", _t.perf_counter() - _run_t0)
if len(st.session_state.step_times):
    __PERF["stream_step_last_ms"] = st.session_state.step_times[-1]
    __PERF["stream_step_avg_ms"]  = round(float(np.mean(st.session_state.step_times)), 2)

# ------------------- KPIs (profile) -------------------
_kpi_t0 = _t.perf_counter()
c1,c2,c3,c4 = st.columns(4)
with c1: st.metric("Current Risk (prob)", f"{(st.session_state.last_pred or {}).get('prob', 0.0):.2f}")
with c2: st.metric("Incidents (session)", len(st.session_state.incidents))
with c3: st.metric("Calibration samples", len(st.session_state.cal_nc))
with c4:
    pv = (st.session_state.last_pred or {}).get("p_value", None)
    st.metric("Conformal p-value", f"{pv:.3f}" if pv is not None else "—")
_perf_set("kpi_render_ms", _t.perf_counter() - _kpi_t0)

# ------------------- Tabs (each section profiled) -------------------
tab_overview, tab_details, tab_insights = st.tabs(["Overview", "Details", "Insights"])

# --- Overview (business)
_ov_t0 = _t.perf_counter()
with tab_overview:
    left, right = st.columns([2,1])

    with left:
        st.subheader("Live Metrics")
        _chart_t0 = _t.perf_counter()
        if len(st.session_state.raw) > 0:
            df = pd.DataFrame(list(st.session_state.raw))
            df["t"] = np.arange(len(df))
            st.line_chart(df.set_index("t")[["snr","packet_loss","latency_ms","pos_error_m"]], height=300)
        else:
            st.info("Waiting for stream…")
        _perf_set("overview_chart_ms", _t.perf_counter() - _chart_t0)

        st.subheader("Incidents")
        _inc_t0 = _t.perf_counter()
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
        _perf_set("overview_incidents_ms", _t.perf_counter() - _inc_t0)

    with right:
        _trust_t0 = _t.perf_counter()
        st.subheader("Trust widget")
        if st.session_state.last_pred:
            p = st.session_state.last_pred["p_value"]
            if p is None and len(st.session_state.cal_nc) < 10:
                st.info("Collecting calibration… (keep ‘Normal’ for a few seconds)")
            else:
                st.metric("Calibrated risk (%)", f"{(1 - (p or 0))*100:.1f}")
                st.caption("Lower p-value = higher risk (inductive conformal).")
        if st.session_state.incidents:
            df_inc = pd.DataFrame(st.session_state.incidents)
            df_inc["top_features"] = df_inc["reasons"].apply(
                lambda r: "; ".join([f"{x['feature']}:{x['impact']:+.3f}" for x in r])
            )
            csv = df_inc.drop(columns=["reasons"]).to_csv(index=False).encode("utf-8")
            st.download_button("Download incidents CSV", csv, "incidents.csv", "text/csv")
        _perf_set("overview_trust_export_ms", _t.perf_counter() - _trust_t0)
_perf_set("overview_total_ms", _t.perf_counter() - _ov_t0)

# --- Details (operator)
_det_t0 = _t.perf_counter()
with tab_details:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Selected metrics")
        _d1 = _t.perf_counter()
        if len(st.session_state.raw) > 0:
            df = pd.DataFrame(list(st.session_state.raw))
            df["t"] = np.arange(len(df))
            for y in ["snr","packet_loss","latency_ms","channel_util"]:
                st.line_chart(df.set_index("t")[[y]], height=140)
        _perf_set("details_selected_metrics_ms", _t.perf_counter() - _d1)
    with col2:
        st.markdown("### Last decision — local explanation")
        _d2 = _t.perf_counter()
        lp = st.session_state.last_pred
        if lp:
            pairs = pd.DataFrame(lp["reasons"], columns=["feature","impact"]).set_index("feature")
            st.bar_chart(pairs, height=260)
            st.caption("Counterfactual hint: lowering the largest positive contributors would reduce alert probability.")
        _perf_set("details_local_explain_ms", _t.perf_counter() - _d2)
_perf_set("details_total_ms", _t.perf_counter() - _det_t0)

# --- Insights (technical)
_ins_t0 = _t.perf_counter()
with tab_insights:
    _i1 = _t.perf_counter()
    st.markdown("### Global importance (mean |contribution|)")
    if st.session_state.global_abs_contrib_count > 0:
        imp = pd.DataFrame([
            {"feature": k, "mean_abs_contrib": v / st.session_state.global_abs_contrib_count}
            for k, v in st.session_state.global_abs_contrib_sum.items()
        ]).sort_values("mean_abs_contrib", ascending=False).head(15).set_index("feature")
        st.bar_chart(imp, height=320)
    _perf_set("insights_global_importance_ms", _t.perf_counter() - _i1)

    _i2 = _t.perf_counter()
    st.markdown("### Model card")
    st.json({
        "model": "Pre-seeded Logistic (linear + 3 interactions)",
        "params": {"intercept": B, "nonzero_weights": int(np.sum([1 for w in W.values() if w != 0]))},
        "features": len(W),
        "explanations": "Local/Global contributions = weight × feature",
        "conformal": "Inductive conformal p-value (built from early Normal samples)",
        "env": {
            "python": sys.version.split()[0],
            "streamlit": st.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "limitations": ["Synthetic data; thresholds illustrative", "Single-device stream for simplicity"],
        "version": "v3-instant-profiled"
    })
    _perf_set("insights_model_card_ms", _t.perf_counter() - _i2)
_perf_set("insights_total_ms", _t.perf_counter() - _ins_t0)

# ------------------- PERF PANEL -------------------
_perf_set("startup_total_ms", _t.perf_counter() - __APP_T0)  # whole-script wall time

if show_perf:
    with st.sidebar.expander("🔧 Performance Debug", expanded=True):
        # Show raw perf dict as a nice table
        perf_df = pd.DataFrame(
            [{"section": k, "ms": v} for k, v in __PERF.items()]
        ).sort_values("ms", ascending=False, ignore_index=True)
        st.dataframe(perf_df, use_container_width=True, height=300)
        # Quick hints
        st.markdown("**Hints**")
        st.markdown("- If `imports_ms` is high: trim dependencies / requirements.txt.")
        st.markdown("- If `overview_chart_ms` is high: reduce `Chart history (points)` in the sidebar.")
        st.markdown("- If `stream_step_avg_ms` is high: lower `Steps per refresh`.")
        st.markdown("- If `startup_total_ms` is high but `imports_ms` low: check repo startup scripts or Cloud build logs.")
