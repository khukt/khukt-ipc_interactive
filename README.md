
# Trustworthy AI Demos — 6G / IIoT (Streamlit)

Three lightweight, self-contained demos tailored to networking & industrial IoT:
1) **Resource Allocation (TreeSHAP)** — local feature attributions for a Random Forest.
2) **Network Anomaly Detection (LIME)** — local surrogate explanations for a gradient boosting classifier.
3) **Predictive Maintenance (Counterfactuals)** — what-if sliders + simple greedy counterfactual search for logistic regression.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push these files (`app.py`, `requirements.txt`) to a public GitHub repo.
2. Go to https://share.streamlit.io → **New app** → pick your repo/branch → **Deploy**.
3. Done. The app auto-installs requirements and runs.

## Notes
- All datasets are synthetic and generated at runtime — no external data needed.
- If SHAP or LIME fail to install on the free tier, the app gracefully falls back to permutation importance / sensitivity analysis.
