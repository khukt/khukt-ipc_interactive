# TRUST AI — Wireless Threat Detection (Streamlit Demo)

A **Streamlit-based interactive demo** for trustworthy anomaly detection in realistic wireless & logistics scenarios.
It simulates Wi‑Fi/private‑5G, cellular, GNSS and integrity signals for a small fleet (AMRs, trucks, sensors, gateways) and
demonstrates **transparent AI**: calibrated anomaly detection, attack typing, role-aware explanations, and EU AI Act–style governance.

> Mid Sweden University · TRUST AI — Realistic Wireless Threats (Sundsvall)  
> Two‑step pipeline: **LightGBM anomaly detector** + **(LightGBM multiclass + rules) for attack typing**  
> Persona‑aware XAI · Conformal risk calibration · Cached models (no retraining on refresh)

---

## ✨ Key Features

- **Realistic telemetry synthesis** (RF/QoS, GNSS, access/auth, cellular, integrity).
- **Binary anomaly detector** with LightGBM, imbalance‑aware training, and **conformal p‑values** for calibrated risk.
- **Attack typing head**: LightGBM multiclass + **domain rule fusion** (Jamming, Access Breach, GPS Spoofing, Data Tamper).
- **Persona‑aware explanations** (End User, Domain Expert, Regulator, AI Builder, Executive).
- **Interactive visuals**: PyDeck geospatial map, risk overlays, KPI heatmaps, incident dashboards.
- **Governance tools**: model card export, audit log download, training explainer, and transparency artifacts.
- **Caching** of trained models to avoid retraining on browser refresh.

---

## 🧰 Tech Stack

- **Python**, **Streamlit**
- **LightGBM** (`LGBMClassifier`) for binary & multiclass stages
- **scikit‑learn** (splits, metrics, scaling)
- **SHAP** (global & local feature attributions)
- **Plotly** (charts) & **PyDeck** (map)

---

## 🗂 Repository Layout (minimal)
```
.
├── app.py                  # Streamlit app (v2.0)
└── README.md               # This file
```
> If you keep extra assets (icons, screenshots) add them under `assets/` and reference below.

---

## 🔧 Installation

```bash
# 1) Clone
git clone https://github.com/<your-org>/trust-ai-wireless-demo.git
cd trust-ai-wireless-demo

# 2) (Recommended) create a virtual env
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt
# or, if you don't have a requirements file yet:
pip install streamlit lightgbm scikit-learn shap plotly pydeck numpy pandas
```

> **Note:** SHAP may build C extensions. If install fails, upgrade pip/setuptools/wheel:  
> `python -m pip install -U pip setuptools wheel`

---

## ▶️ Run

```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**.

---

## 🧪 What the Demo Does

1. **Synthesizes telemetry** for ~30 devices around Sundsvall (lat/lon center configurable).
2. Builds **rolling‑window features** per device (mean/std/min/max/last/slope/z/jump for each raw feature).
3. Trains a **binary LightGBM** detector with class imbalance handling and evaluates (AUC, Precision/Recall/F1, Brier).
4. Calibrates risk with **conformal p‑values** for statistically meaningful severity.
5. Trains a **multiclass LightGBM** to propose **attack type**, then **fuses** with domain rules.
6. Streams the fleet; **incidents** trigger with SHAP‑based explanations and role‑specific guidance.
7. Exposes **governance artifacts** (model card, audit log, data schema) for download.

---

## 🎛️ UI Guide (Sidebar Controls)

- **Comms profile:** Wi‑Fi/private‑5G dominant vs. cellular‑dominant road profile.  
- **Scenario:** `Normal`, `Jamming (localized)`, `Access Breach (AP/gNB)`, `GPS Spoofing (subset)`, `Data Tamper (gateway)`  
  - Scenario‑specific knobs (e.g., jam radius, rogue AP mode, spoofing scope).
- **Playback:** speed, auto stream, reset.  
- **Model:** enable **Conformal risk**, incident threshold (with **suggested threshold** from validation).  
- **Display:** toggle map/heatmap, filter device types.  
- **Viewer role:** switches the **explanations** and **action guidance**.  
- **Help & EU AI Act banner:** quick onboarding, transparency status.

---

## 🛰️ Scenarios & Signals (at a glance)

| Scenario        | Indicative Signals (↑/↓ denote direction) |
|-----------------|--------------------------------------------|
| **Jamming**     | ↑ noise floor, ↑ BLER/PHY errors, ↓ SNR/SINR, ↑ loss/latency |
| **Access Breach** | ↑ deauth, ↑ assoc churn, ↑ 802.1X/DHCP retries, **rogue_rssi_gap > 0**, PCI anomalies |
| **GPS Spoofing** | ↑ pos_error_m, *odd* HDOP (too low/high), ↓ sats, ↑ Doppler var, ↑ clock drift, abnormal C/N0 patterns |
| **Data Tamper** | ↑ dup_ratio, ↑ seq_gap, ↑ timestamp skew, ↑ schema violations, ↑ HMAC/CRC errors |

---

## 🧠 Explainability & Governance

- **Local SHAP** for each incident (device inspector), plus **global SHAP** (mean |SHAP| ranking).  
- **Role‑aware narratives** to keep UX focused (end users vs. experts vs. regulators).  
- **Model card (JSON)** includes: model config, data sources, features, metrics, calibration, intended use, limitations.  
- **Audit log** export of incidents for traceability and evidence.

---

## 📈 Metrics & Calibration

- Binary detector: **AUC, Precision, Recall, F1**, **Brier score** (reliability).  
- Threshold suggestion = argmax F1 over a validation sweep.  
- **Conformal p‑values** provide coverage‑controlled evidence (lower p ⇒ stronger anomaly evidence).  
- Type head uses **temperature scaling** on probabilities + rule fusion with tunable weight (`TYPE_ALPHA`).

---

## 📦 Data Notes

- **Synthetic** physics‑inspired telemetry only; **no personal data**.  
- Rolling windows aggregate per‑device signals into engineered features.  
- Export **data schema** & **incidents** from the Governance tab.

---

## 🐛 Troubleshooting

- **Black screen / layout too tight**: switch to wide mode (default), collapse help, or reduce map overlays.  
- **SHAP import errors**: `pip install -U pip setuptools wheel` then reinstall SHAP.  
- **LightGBM build issues on Windows**: try pre‑built wheels (`pip install lightgbm`) or use conda.  
- **No incidents**: lower the incident threshold or enable an attack scenario; let the stream run a few ticks.  
- **GPU not required**: everything runs on CPU.

---

## 📜 License & Usage

This repository is for **research and educational** demonstration under the TRUST initiative.  
For industrial use, integration into safety control loops, or redistribution, please **contact the authors**.

