
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

# Optional imports that might fail on some environments; guard gracefully
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

st.set_page_config(page_title="Trustworthy AI Demos (6G/IIoT)", layout="wide")

st.title("Trustworthy AI Demos — 6G / IIoT")
st.caption("Three lightweight, self-contained demos for Explainable AI, tailored to networking & industrial IoT.")

demo = st.sidebar.radio(
    "Choose a demo",
    ["1) Resource Allocation (TreeSHAP)",
     "2) Network Anomaly Detection (LIME)",
     "3) Predictive Maintenance (Counterfactuals)"]
)

# --------- Utility: synthetic data generators --------- #
rng = np.random.default_rng(42)

def gen_resource_alloc(n=800, seed=0):
    r = np.random.default_rng(seed)
    snr = r.normal(loc=10, scale=5, size=n)          # dB
    latency_req = r.uniform(1, 100, size=n)          # ms
    energy = r.uniform(0.1, 1.0, size=n)             # normalized
    mobility = r.uniform(0, 50, size=n)              # km/h
    device_priority = r.integers(0, 3, size=n)       # {0,1,2}
    # Label rule: allocate if (SNR high) & (latency strict) & (priority high) OR (energy high & low mobility)
    score = 0.35*(snr>12) + 0.25*(latency_req<20) + 0.25*(device_priority==2) + 0.15*((energy>0.6)&(mobility<10))
    y = (score + r.normal(0, 0.05, n) > 0.5).astype(int)
    X = pd.DataFrame({
        "snr_db": snr,
        "latency_req_ms": latency_req,
        "energy_norm": energy,
        "mobility_kmh": mobility,
        "device_priority": device_priority
    })
    return X, y

def gen_anomaly(n=1000, seed=1):
    r = np.random.default_rng(seed)
    mean_latency = r.normal(30, 10, size=n)          # ms
    jitter = r.exponential(5, size=n)                # ms
    throughput = r.normal(100, 30, size=n)           # Mbps
    pkt_drop = r.uniform(0, 5, size=n)               # %
    burstiness = r.uniform(0, 1, size=n)             # unitless

    # Attack rule: high jitter or high drop + bursts; or weird combo low throughput + high latency
    score = 0.4*(jitter>10) + 0.3*(pkt_drop>2.5) + 0.2*(burstiness>0.7) + 0.2*((throughput<70)&(mean_latency>45))
    y = (score + r.normal(0, 0.05, n) > 0.5).astype(int)
    X = pd.DataFrame({
        "mean_latency_ms": mean_latency,
        "jitter_ms": jitter,
        "throughput_mbps": throughput,
        "packet_drop_pct": pkt_drop,
        "burstiness": burstiness
    })
    return X, y

def gen_maintenance(n=900, seed=2):
    r = np.random.default_rng(seed)
    temp = r.normal(65, 15, size=n)                  # °C
    vibration = r.normal(25, 8, size=n)              # Hz
    pressure = r.normal(4, 1.2, size=n)              # bar
    age = r.uniform(0, 10, size=n)                   # years

    # Failure rule: high temp, high vibration, high age; pressure extremes
    score = 0.35*(temp>75) + 0.3*(vibration>30) + 0.2*(age>6) + 0.15*((pressure<3)|(pressure>6))
    y = (score + r.normal(0, 0.05, n) > 0.5).astype(int)
    X = pd.DataFrame({
        "temp_c": temp,
        "vibration_hz": vibration,
        "pressure_bar": pressure,
        "age_years": age
    })
    return X, y

# --------- Demo 1: Resource Allocation with TreeSHAP --------- #
if demo.startswith("1"):
    st.header("1) Resource Allocation (TreeSHAP)")
    st.write("**Question**: Why did the RL-inspired policy allocate bandwidth to Device A instead of B? "
             "We train a simple Random Forest on synthetic network features and explain its decision with TreeSHAP.")
    X, y = gen_resource_alloc()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, random_state=0, max_depth=6)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:,1]
    st.write(f"**Accuracy**: {accuracy_score(y_test, (preds>0.5).astype(int)):.3f} • **ROC-AUC**: {roc_auc_score(y_test, preds):.3f}")

    idx = st.slider("Choose a test sample to explain", min_value=0, max_value=len(X_test)-1, value=5, step=1)
    x_instance = X_test.iloc[[idx]]

    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(x_instance)
        st.subheader("Feature Contributions (SHAP)")
        fig, ax = plt.subplots()
        # For binary classifier, shap_values is a list [class0, class1]; pick class 1
        vals = shap_values[1][0]
        order = np.argsort(np.abs(vals))[::-1]
        labels = x_instance.columns[order]
        vals_sorted = vals[order]
        ax.barh(labels, vals_sorted)
        ax.invert_yaxis()
        ax.set_xlabel("SHAP value → contribution to P(allocate_to_A)")
        st.pyplot(fig)
    else:
        st.warning("SHAP not available in this environment; showing permutation importance instead.")
        # Simple permutation importance as a fallback
        base = clf.score(X_test, y_test)
        imps = []
        for col in X_test.columns:
            Xp = X_test.copy()
            Xp[col] = np.random.permutation(Xp[col].values)
            imps.append(base - clf.score(Xp, y_test))
        imp_series = pd.Series(imps, index=X_test.columns).sort_values(ascending=False)
        st.bar_chart(imp_series)

    st.markdown("**Takeaway**: TreeSHAP gives a local explanation per decision, showing which features pushed the model toward allocating bandwidth.")

# --------- Demo 2: Network Anomaly Detection with LIME --------- #
elif demo.startswith("2"):
    st.header("2) Network Anomaly Detection (LIME)")
    st.write("**Question**: Why did the detector flag this traffic as an attack? "
             "We train a Gradient Boosting classifier and use LIME to explain a single prediction.")
    X, y = gen_anomaly()
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.25, random_state=0, stratify=y)

    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:,1]
    st.write(f"**Accuracy**: {accuracy_score(y_test, (preds>0.5).astype(int)):.3f} • **ROC-AUC**: {roc_auc_score(y_test, preds):.3f}")

    idx = st.slider("Choose a test sample to explain", min_value=0, max_value=len(X_test)-1, value=10, step=1)
    x_instance = X_test[idx]

    if LIME_AVAILABLE:
        explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=list(pd.DataFrame(X).columns),
            class_names=["normal","attack"],
            discretize_continuous=True,
            mode="classification"
        )
        exp = explainer.explain_instance(
            x_instance, model.predict_proba, num_features=5, top_labels=1
        )
        st.subheader("Local Explanation (Top 5 contributions)")
        df = pd.DataFrame(exp.as_list(), columns=["Feature", "Contribution"])
        st.table(df)
    else:
        st.warning("LIME not available; showing simple feature sensitivity instead.")
        base = model.predict_proba([x_instance])[0,1]
        sens = []
        cols = list(pd.DataFrame(X).columns)
        for j, name in enumerate(cols):
            x_mod = x_instance.copy()
            x_mod[j] = x_mod[j] + 0.1*np.std(X_train[:,j])
            new = model.predict_proba([x_mod])[0,1]
            sens.append(new - base)
        df = pd.DataFrame({"feature": cols, "delta_prob": sens}).sort_values("delta_prob", ascending=False)
        st.table(df)

    st.markdown("**Takeaway**: LIME provides an intuitive, local surrogate model that explains why this sample was labeled as an anomaly.")

# --------- Demo 3: Predictive Maintenance with Counterfactuals --------- #
else:
    st.header("3) Predictive Maintenance (Counterfactuals)")
    st.write("**Question**: The model predicts failure—*what needs to change to avoid it?* "
             "We train a simple logistic regression and provide a what-if and counterfactual search.")
    X, y = gen_maintenance()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict_proba(X_test)[:,1]
    st.write(f"**Accuracy**: {accuracy_score(y_test, (preds>0.5).astype(int)):.3f} • **ROC-AUC**: {roc_auc_score(y_test, preds):.3f}")

    st.subheader("What-if controls")
    c1, c2, c3, c4 = st.columns(4)
    temp = c1.slider("temp_c", float(X["temp_c"].min()), float(X["temp_c"].max()), float(np.median(X["temp_c"])))
    vib = c2.slider("vibration_hz", float(X["vibration_hz"].min()), float(X["vibration_hz"].max()), float(np.median(X["vibration_hz"])))
    press = c3.slider("pressure_bar", float(X["pressure_bar"].min()), float(X["pressure_bar"].max()), float(np.median(X["pressure_bar"])))
    age = c4.slider("age_years", float(X["age_years"].min()), float(X["age_years"].max()), float(np.median(X["age_years"])))

    x_cur = pd.DataFrame([[temp, vib, press, age]], columns=X.columns)
    prob_fail = pipe.predict_proba(x_cur)[0,1]
    st.metric("Predicted Failure Probability", f"{prob_fail:.2%}")

    st.subheader("Find a simple counterfactual")
    target_prob = st.slider("Target max failure probability", 0.01, 0.49, 0.20, 0.01)

    # Greedy counterfactual: nudge features along direction that reduces failure prob
    # Use model coefficients (in standardized space) to get influence directions
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    coefs = clf.coef_[0]

    def to_std(x_df):
        return (x_df.values - scaler.mean_) / np.sqrt(scaler.var_)

    def to_orig(x_std):
        return x_std * np.sqrt(scaler.var_) + scaler.mean_

    x_std = to_std(x_cur)
    cur_prob = prob_fail
    steps = []
    max_iters = 200
    lr = 0.1  # step size in standardized space

    bounds = np.array([[X[c].min(), X[c].max()] for c in X.columns])

    for it in range(max_iters):
        if cur_prob <= target_prob:
            break
        # Move opposite to positive coefficients (reduce logit)
        direction = -np.sign(coefs)
        x_std = x_std + lr * direction  # nudge each feature
        # Project back to original bounds
        x_new = to_orig(x_std)[0]
        x_new = np.clip(x_new, bounds[:,0], bounds[:,1])
        x_std = (x_new - scaler.mean_) / np.sqrt(scaler.var_)
        prob = pipe.predict_proba(pd.DataFrame([x_new], columns=X.columns))[0,1]
        steps.append((it+1, x_new.copy(), prob))

        # Reduce step size if oscillating
        if len(steps)>2 and steps[-1][2] > steps[-2][2]:
            lr *= 0.5
        cur_prob = prob

    if steps:
        iters, xs, probs = zip(*steps)
        best_idx = int(np.argmin(probs))
        x_cf = xs[best_idx]
        p_cf = probs[best_idx]
        df_compare = pd.DataFrame([x_cur.iloc[0].values, x_cf], columns=X.columns, index=["current","counterfactual"]).T
        st.write("**Suggested counterfactual changes** (minimal greedy):")
        st.dataframe(df_compare.style.format("{:.2f}"))
        st.metric("Counterfactual Failure Probability", f"{p_cf:.2%}")
        st.caption("Note: Simple greedy search; for production, consider DiCE or optimization-based counterfactuals with cost/feasibility constraints.")
    else:
        st.info("Current configuration already below the target failure probability.")

    st.markdown("**Takeaway**: Counterfactuals provide actionable guidance — what needs to change to lower risk.")

st.sidebar.markdown("---")
st.sidebar.markdown("**How this maps to your lectures**")
st.sidebar.markdown('''
- Lecture 1: Motivation & Trustworthy AI framing
- Lecture 2: Methods (TreeSHAP, LIME) + Demos 1 & 2
- Lecture 3: Actionability & Future (Counterfactuals) + Demo 3
''')
