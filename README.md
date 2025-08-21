
# ⚡ Sectoral Energy & Emissions Forecasting Dashboard

A Streamlit-powered interactive dashboard for analyzing UK energy consumption and carbon emissions by sector and end-use. The tool integrates historical data exploration, forecasting, intervention simulations, and SHAP explainability to support evidence-based sustainability decisions.

---

## 🚀 Features

### 📊 Trends Explorer
- Line plots of **energy consumption** and **emission intensity** over time.
- Filter by **sector**, **end use**, and **year range**.
- Export summary statistics and the full dataset.

### 🔮 Forecasting
- Forecast ahead to 2030:
  - `Energy Consumption (ktoe)`
  - `Emissions (ktCO2e)`
- Uses **ARIMA (1,1,1)** model per sector-end use pair.
- Displays ARIMA model summary (AIC, coefficients).
- Optional time-series plot overlays forecast on history.

### 🧪 Simulation Scenarios
- Apply what-if scenarios and visualize impacts:
  - **Electrification of Appliances** (20% savings)
  - **Services Rebound** (20% increase)
  - **Targeted Industrial Emission Cut** (25% reduction)
- Interactive simulation: choose sector/end-use and simulate 20% reduction in energy from 2023 onward.
- Impact shown on both energy and emissions plots.

### 🧠 SHAP Explainability
- Trained **XGBoost regression model** for emissions using:
  - Energy consumption
  - Heating Degree Days (HDD)
  - Emission intensity
  - One-hot encoded sectors
- Displays:
  - SHAP summary plot (beeswarm)
  - Row-level force plots via interactive row slider

### 📥 Upload Your Own Dataset
- Upload your own `.csv` with similar structure (`Year`, `Sector`, `End Use`, etc.).
- The dashboard adapts to uploaded data instantly.
- All tabs and models are re-run with user data.

### 📈 EDA (Exploratory Data Analysis)
- Histogram + KDE plots for any numeric column
- Correlation heatmap to identify interdependencies
- Quick export of transformed dataset

---

## 🧬 Dataset

Default dataset: `Merged_Dataset__Energy___Emissions___HDD.csv`

| Column | Description |
|--------|-------------|
| `Year` | Calendar year |
| `Sector` | Sector (e.g., Domestic, Services, Industrial) |
| `End Use` | End use category (e.g., Space Heating, Appliances) |
| `Energy Consumption (ktoe)` | Final energy consumption |
| `Emissions (ktCO2e)` | CO₂-equivalent emissions |
| `Annual_HDD` | Annual Heating Degree Days (weather control) |

You may upload your own data with a similar schema to explore your scenario.

---

## 💻 Running Locally

### 1. Clone the repo

```bash
git clone https://github.com/TeresaKamiri/Energy-Forecast-Dashboard.git
cd Energy-Forecast-Dashboard
```

### 2. Create virtual environment

```bash
python -m venv dsp-env
source dsp-env/bin/activate  # or `.\dsp-env\Scripts\activate` on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch app

```bash
streamlit run app.py
```

---

## 📦 File Structure

```
energy-dashboard/
│
├── app.py                  # Main Streamlit dashboard
├── requirements.txt        # Python dependencies
├── Merged_Dataset__*.csv   # Default data file
└── README.md               # You're here
```

---

## 📌 Example Enhancements

- [ ] Export dashboard snapshots as PDF or PNG
- [ ] Add Prophet forecast model toggle
- [ ] Allow filtering by specific fuel type (e.g., electricity only)
- [ ] Include water heating or cooking use-cases

---

## 🤝 Contribution

Got ideas for better simulation models, explainability layers, or visual layouts? Fork it, try it, and PR it!

---

## 🧠 Citation

This dashboard was developed as part of a master's project titled:

> “**Sectoral Forecasting of Energy Demand and Carbon Emissions: Predictive Modelling Across Industrial, Commercial, and Residential Use Cases**”
