# ⚡ Sectoral Energy & Emissions Forecasting Dashboard

A Streamlit-powered interactive dashboard for analyzing UK energy consumption and carbon emissions by sector and end‑use. The tool integrates historical data exploration, forecasting, intervention simulations, and SHAP explainability — with policy overlays, event markers, and a 2030 contributions waterfall.

---

## 🚀 Feature Highlights

### 📊 Trends Explorer
- Line plots of **energy consumption** and **emission intensity** over time.
- Filter by **sector**, **end use**, and **year range**.
- Export summary statistics and the full dataset.

### 🔮 Forecasting (to 2030)
- Forecast `Emissions (ktCO2e)` and `Emission Intensity (ktCO2e/ktoe)` with **ARIMA(1,1,1)** per sector–end‑use.
- Toggle **Policy overlays** to compare BAU vs. policy path.
- **Policy event markers** (vertical dotted lines with labels) indicate BUS/GBIS/ETS/CBAM/PSDS milestones.

### 🧪 Simulation Scenarios
- One‑click scenarios: **Appliance electrification (−20% energy)**, **Services rebound (+20% energy)**, **Industrial process‑heat cuts (−25% emissions)**.
- Custom intervention: select any sector/end‑use and apply a 20% energy change from 2023 onward.
- All simulation charts support **policy overlays** and **event markers**.

### 🧮 2030 “Policy Lever Contributions” Waterfall
- Computes **Δ Emissions vs BAU in 2030** by lever:
  - Domestic/Residential — Space heating (BUS/GBIS)
  - Domestic/Residential — Appliances (Electrification)
  - Industrial — Process heat (CCUS)
  - Services/Commercial — HVAC & Lighting (MEES/PSDS)
- Method: BAU 2030 via ARIMA → apply overlay multipliers → plot reductions as negative bars.

### 🧠 SHAP Explainability
- XGBoost model for emissions with SHAP **summary** and **instance‑level** plots (force / waterfall).

### 🧪 EDA
- Histograms + KDE, correlation heatmap, and downloadable summaries.

---

## 📂 Policy Integration (Overlays & Events)

The dashboard will look for files in either the repo root **or** a `data/` folder:

- `policy_overlays.csv`
- `policy_events.json`

### `policy_overlays.csv` schema

| column | type | notes |
|---|---|---|
| `sector` | string | e.g., `Domestic/Residential`, `Industrial`, `Services/Commercial` |
| `end_use` | string | e.g., `Space heating`, `Appliances`, `Process heat`, `HVAC & Lighting` |
| `year` | integer | must include **2030** for each lever you want in the waterfall |
| `multiplier_energy` | float | scaling to apply to BAU **energy** |
| `multiplier_emissions` | float | scaling to apply to BAU **emissions** |
| `label` | string (optional) | free‑text label shown in legends |

### `policy_events.json` schema

```json
[
  {"date": "2024-11-21", "event": "BUS budget set for 2025/26 (£295m)"},
  {"date": "2026-03-31", "event": "GBIS scheduled end date"},
  {"date": "2026-07-01", "event": "UK ETS domestic maritime inclusion (MRV phase)"},
  {"date": "2027-01-01", "event": "UK CBAM starts"},
  {"date": "2028-03-31", "event": "PSDS Phase 4 delivery window end"}
]
```

### Label normalization (so your files don’t have to be perfect)
The app normalizes differences in spelling/case/spacing:

| Dataset label | Overlay label it maps to |
|---|---|
| `Domestic` → | `Domestic/Residential` |
| `Services`/`Commercial` → | `Services/Commercial` |
| `Process heating` ↔︎ | `Process heat` |
| `Lighting` → | `HVAC & Lighting` *(proxy)* |

If your dataset uses different end‑use names (e.g., `Space Heating` vs `Space heating`), the overlay still binds.

---

## 🧭 Troubleshooting

**“No matching overlays found to compute the waterfall.”**
- Ensure the CSV contains **2030** rows for the four levers listed above.
- Check column names: `sector,end_use,year,multiplier_emissions[,multiplier_energy]`.
- Open **Simulations → Overlay debug / status** to see the loaded path, row count, and sample rows.

**Events not showing?**
- Confirm `policy_events.json` is valid JSON and dates use `YYYY-MM-DD`.

**Overlays not binding to your data?**
- Compare your dataset’s `Sector`/`End Use` to the mapping table above.

## 🧬 Dataset

Default dataset: `Merged_Dataset__Energy___Emissions___HDD.csv`

| Column | Description |
|---|---|
| `Year` | Calendar year |
| `Sector` | e.g., Domestic, Services, Industrial |
| `End Use` | e.g., Space heating, Appliances |
| `Energy Consumption (ktoe)` | Final energy consumption |
| `Emissions (ktCO2e)` | CO₂‑equivalent emissions |
| `Annual_HDD` | Heating Degree Days (weather control) |

You can upload your own data with the similar schema.

## 💻 Run Locally

```bash
# 1) clone
git clone https://github.com/TeresaKamiri/Energy-Forecast-Dashboard.git
cd Energy-Forecast-Dashboard

# 2) (optional) create venv
python -m venv dsp-env
.\dsp-env\Scripts\activate  # Linux: source dsp-env/bin/activate  # To activate.

# 3) install deps
pip install -r requirements.txt

# 4) run
streamlit run app.py
```

**Core dependencies:** `streamlit`, `pandas`, `plotly`, `statsmodels`, `xgboost`, `shap`, `matplotlib`, `seaborn`.

---

## 📦 Repo Structure

```
energy-dashboard/
├─ app.py                 # Streamlit dashboard
├─ requirements.txt       # Python dependencies
├─ Merged_Dataset__Energy___Emissions___HDD.csv  # Default data file
├─ policy_overlays.csv    # ↖ optional
├─ policy_events.json     # ↖ optional
└─ README.md
```


## 🤝 Contribution
PRs welcome — especially on additional policy levers, improved mappings, or alternative forecasting models (Prophet/TBATS).
