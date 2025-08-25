import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
import shap
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from typing import List

# Page config
st.set_page_config(page_title="Energy Dashboard", layout="wide")
st.title("Energy Dashboard: Trends, Forecasting, Simulations & SHAP Explainability")

# Normalizers (dataset + overlays)
def _norm(s: str) -> str:
    return str(s).strip().lower()

def _norm_sector(s: str) -> str:
    s = _norm(s)
    if 'industrial' in s:
        return 'industrial'
    if 'service' in s or 'commercial' in s:
        return 'services/commercial'
    if 'residential' in s or 'domestic' in s:
        return 'domestic/residential'
    return s

def _norm_end_use(s: str) -> str:
    s = _norm(s)
    if 'process heat' in s or 'process heating' in s:
        return 'process heat'
    if 'space heat' in s:
        return 'space heating'
    if 'appliance' in s:
        return 'appliances'
    if 'hvac' in s or 'lighting' in s:
        return 'hvac & lighting'
    return s

# Data loading (+ normalized cols)
@st.cache_data
def load_data():
    df = pd.read_csv("Merged_Dataset__Energy___Emissions___HDD.csv")
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.dropna(subset=['Year'], inplace=True)
    df['sector_norm'] = df['Sector'].apply(_norm_sector)
    df['end_use_norm'] = df['End Use'].apply(_norm_end_use)
    return df

df = load_data()
sectors = sorted(df['Sector'].dropna().unique().tolist())
end_uses = sorted(df['End Use'].dropna().unique().tolist())

# Global year filter
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())
year_range = st.sidebar.slider("Filter Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])].copy()
# recompute norms after filter
df['sector_norm'] = df['Sector'].apply(_norm_sector)
df['end_use_norm'] = df['End Use'].apply(_norm_end_use)

with st.sidebar:
    st.title("⚡ Energy Dashboard")
    selected_tab = st.radio("Navigate", ["Sector Overview", "Trends", "Forecasting", "Simulations", "SHAP Explainability", "EDA"]) 

# Robust loaders for policy files (CSV + JSON)
def _try_read_csv(paths: List[str]) -> pd.DataFrame:
    for p in paths:
        try:
            po = pd.read_csv(p)
            st.session_state['__policy_csv_path'] = p
            return po
        except Exception:
            continue
    return pd.DataFrame()

def _try_read_json(paths: List[str]):
    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
                st.session_state['__policy_json_path'] = p
                return data
        except Exception:
            continue
    return []

_policy_csv_paths = [
    "policy_overlays.csv", "./policy_overlays.csv",
    "data/policy_overlays.csv", "./data/policy_overlays.csv",
]
_policy_json_paths = [
    "policy_events.json", "./policy_events.json",
    "data/policy_events.json", "./data/policy_events.json",
]

policy_overlays_raw = _try_read_csv(_policy_csv_paths)
policy_events = _try_read_json(_policy_json_paths)

if not policy_overlays_raw.empty:
    po = policy_overlays_raw.copy()
    po.columns = [c.strip().lower() for c in po.columns]
    # ensure essential cols
    for col in ['sector','end_use','year','multiplier_emissions']:
        if col not in po.columns:
            st.warning(f"policy_overlays.csv is missing column: {col}")
    if 'multiplier_energy' not in po.columns:
        po['multiplier_energy'] = po['multiplier_emissions']
    po['year'] = pd.to_numeric(po['year'], errors='coerce').astype('Int64')
    po['sector_norm'] = po['sector'].apply(_norm_sector)
    po['end_use_norm']  = po['end_use'].apply(_norm_end_use)
    policy_overlays = po
else:
    policy_overlays = pd.DataFrame(columns=['sector','end_use','year','multiplier_emissions','multiplier_energy','sector_norm','end_use_norm'])

# Forecast helpers
def build_long(series_df: pd.DataFrame, value_col: str):
    out = pd.DataFrame(columns=["Year","Value","Type","Metric"])
    model = None
    if series_df.empty or value_col not in series_df.columns:
        return out, model
    ts = series_df[["Year", value_col]].dropna().sort_values('Year')
    if ts.empty:
        return out, model

    last_year = int(ts['Year'].max())
    steps = 2030 - last_year

    hist_df = ts.rename(columns={value_col: "Value"})
    hist_df["Type"] = "Historical"
    hist_df["Metric"] = value_col

    if len(ts) >= 5:
        try:
            m = ARIMA(ts.set_index('Year')[value_col], order=(1,1,1)).fit()
            model = m
            if steps > 0:
                fc = m.forecast(steps=steps)
                fc_years = np.arange(last_year+1, 2031)
                fc_df = pd.DataFrame({"Year": fc_years, "Value": fc.values, "Type":"Forecast", "Metric": value_col})
                out = pd.concat([hist_df, fc_df], ignore_index=True)
                return out, model
        except Exception:
            pass
    return hist_df, model

@st.cache_data(show_spinner=False)
def get_overlay_row(sector_norm: str, end_use_norm: str, year: int):
    if policy_overlays.empty:
        return None
    ov = policy_overlays[(policy_overlays['sector_norm'] == sector_norm) & (policy_overlays['end_use_norm'] == end_use_norm)]
    if ov.empty:
        return None
    cand = ov[ov['year'] == year]
    if not cand.empty:
        return cand.iloc[0]
    # nearest year fallback
    try:
        idx = (ov['year'] - year).abs().idxmin()
        return ov.loc[idx]
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def bau2030_norm(df_in: pd.DataFrame, sector_norm: str, end_norm: str) -> float:
    """Robust BAU 2030 using normalized labels + sensible fallbacks."""
    sub = df_in[(df_in['sector_norm']==sector_norm) & (df_in['end_use_norm']==end_norm)]
    if sub.empty:
        sec = df_in[df_in['sector_norm']==sector_norm]
        cand = sec[sec['End Use'].str.contains('Lighting|HVAC', case=False, na=False)]
        if cand.empty:
            ts = sec.groupby('Year')['Emissions (ktCO2e)'].sum().reset_index()
        else:
            ts = cand.groupby('Year')['Emissions (ktCO2e)'].sum().reset_index()
    else:
        ts = sub[['Year','Emissions (ktCO2e)']].copy()

    ts = ts.dropna().sort_values('Year')
    if ts.empty:
        return float('nan')
    y = ts.set_index('Year')['Emissions (ktCO2e)']
    if len(y) < 5:
        return float(y.loc[2030]) if 2030 in y.index else float(y.iloc[-1])
    try:
        m = ARIMA(y, order=(1,1,1)).fit()
        steps = 2030 - int(y.index.max())
        if steps <= 0:
            return float(y.iloc[-1])
        fc = m.forecast(steps=steps)
        return float(fc.iloc[-1])
    except Exception:
        return float(y.iloc[-1])

# Plot helper: overlays + vertical policy events
def add_policy_overlays(fig: go.Figure, sector: str, end_use: str, base_long: pd.DataFrame, metric: str = 'emissions') -> go.Figure:
    """Add dashed policy path + vertical policy events to a Plotly fig.
    For forecasting views, the overlay follows the BAU series year-by-year:
    y_overlay(year) = BAU_value(year) * policy_multiplier(year).
    metric ∈ {'emissions','energy','intensity'} controls which multiplier(s) are used.
    """
    sector_o, end_use_o = _norm_sector(sector), _norm_end_use(end_use)
    ov_year_min = ov_year_max = None

    # ---- Build overlay path by merging overlay multipliers with BAU/base series ----
    if not policy_overlays.empty and not base_long.empty:
        ov = policy_overlays[(policy_overlays['sector_norm'] == sector_o) & (policy_overlays['end_use_norm'] == end_use_o)].copy()
        if not ov.empty:
            # Select appropriate multiplier(s)
            col_e = 'multiplier_energy'
            col_c = 'multiplier_emissions'
            needed_cols = [col_e, col_c]
            for c in needed_cols:
                if c not in ov.columns:
                    ov[c] = np.nan
            ov['year'] = pd.to_numeric(ov['year'], errors='coerce')
            base = base_long[['Year','Value']].dropna().copy()
            base['Year'] = pd.to_numeric(base['Year'], errors='coerce')

            # Choose multiplier logic
            if metric == 'energy':
                ov['mult'] = pd.to_numeric(ov[col_e], errors='coerce')
            elif metric == 'intensity':
                # Intensity ≈ Emissions/Energy, so apply ratio of multipliers when available
                me = pd.to_numeric(ov[col_c], errors='coerce')
                en = pd.to_numeric(ov[col_e], errors='coerce')
                ov['mult'] = (me / en).replace([np.inf, -np.inf], np.nan)
            else:  # 'emissions'
                ov['mult'] = pd.to_numeric(ov[col_c], errors='coerce')

            merged = pd.merge(ov[['year','mult']], base, left_on='year', right_on='Year', how='left')
            merged = merged.dropna(subset=['year','mult','Value']).sort_values('year')
            if not merged.empty:
                y_overlay = merged['Value'].astype(float) * merged['mult'].astype(float)
                fig.add_trace(go.Scatter(x=merged['year'], y=y_overlay, mode="lines+markers", name="Policy path", line=dict(dash="dash")))
                ov_year_min, ov_year_max = int(merged['year'].min()), int(merged['year'].max())

    # ---- Event markers ----
    if policy_events:
        y_max = None
        if 'Value' in base_long.columns and base_long['Value'].notna().any():
            y_max = float(base_long['Value'].max())
        if y_max is None:
            try:
                y_max = max([np.nanmax(getattr(tr, 'y', np.array([0]))) for tr in fig.data])
            except Exception:
                y_max = 0
        for e in policy_events:
            try:
                year = int(str(e.get("date", "").split("-")[0]))
            except Exception:
                year = None
            if year:
                fig.add_vline(x=year, line_width=1, line_dash="dot", line_color="gray")
                fig.add_annotation(x=year, y=y_max, text=e.get("event",""), showarrow=False, yshift=10, font=dict(size=9))

    # ---- Ensure x‑axis includes overlay years ----
    try:
        base_year_min = int(pd.to_numeric(base_long.get('Year'), errors='coerce').dropna().min()) if 'Year' in base_long else None
        base_year_max = int(pd.to_numeric(base_long.get('Year'), errors='coerce').dropna().max()) if 'Year' in base_long else None
        xmin_candidates = [v for v in [base_year_min, ov_year_min] if v is not None]
        xmax_candidates = [v for v in [base_year_max, ov_year_max] if v is not None]
        if xmin_candidates and xmax_candidates:
            fig.update_xaxes(range=[min(xmin_candidates), max(xmax_candidates)])
    except Exception:
        pass

    return fig

# Tabs
if selected_tab == "Trends":
    st.subheader("📊 Sectoral Trends and Emission Intensity")
    selected_sector = st.selectbox("Select Sector", sectors)
    df_sector = df[df['Sector'] == selected_sector].copy()

    fig1 = px.line(df_sector, x="Year", y="Energy Consumption (ktoe)", color="End Use", title=f"Energy Consumption Trends — {selected_sector}")
    st.plotly_chart(fig1, use_container_width=True)

    df_sector["Emission Intensity"] = df_sector["Emissions (ktCO2e)"] / df_sector["Energy Consumption (ktoe)"]
    fig2 = px.line(df_sector, x="Year", y="Emission Intensity", color="End Use", title=f"Emission Intensity (ktCO2e/ktoe) — {selected_sector}")
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Summary & Download"):
        st.dataframe(df_sector)
        csv = df.to_csv(index=False).encode()
        st.download_button("Download merged dataset", csv, "merged_energy_emissions.csv", "text/csv")

elif selected_tab == "Forecasting":
    st.subheader("🔮 Forecast to 2030: Emissions & Emission Intensity (Historical + Forecast)")

    selected_sector = st.selectbox("Select Sector", sectors, key="forecast_sector")
    selected_end_use = st.selectbox("Choose End Use", end_uses)
    show_policy = st.checkbox("Show policy overlays & events", value=True)

    df_series = df[(df['Sector'] == selected_sector) & (df['End Use'] == selected_end_use)].copy()
    df_series = df_series.sort_values("Year")
    df_series = df_series[df_series['Energy Consumption (ktoe)'] > 0]
    df_series["Emission Intensity"] = df_series["Emissions (ktCO2e)"] / df_series["Energy Consumption (ktoe)"]
    df_series.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_series.dropna(subset=["Emissions (ktCO2e)", "Emission Intensity"], inplace=True)

    if df_series.empty:
        st.info("No valid rows after filtering (check selected sector/end use and year range).")
        st.stop()

    st.write(f"**Context:** {selected_sector} → {selected_end_use}. Historical years: {int(df_series['Year'].min())}–{int(df_series['Year'].max())}. Target: 2030.")

    emissions_long, emissions_model = build_long(df_series, "Emissions (ktCO2e)")
    intensity_long, intensity_model = build_long(df_series, "Emission Intensity")

    # Emissions chart with overlays
    st.markdown("### 📉 Emissions (ktCO₂e): Historical + Forecast to 2030")
    if not emissions_long.empty:
        fig_emis = go.Figure()
        for t in emissions_long["Type"].unique():
            cur = emissions_long[emissions_long['Type'] == t]
            fig_emis.add_trace(go.Scatter(x=cur['Year'], y=cur['Value'], mode="lines+markers", name=f"Emissions ({t})"))
        if show_policy:
            fig_emis = add_policy_overlays(fig_emis, selected_sector, selected_end_use, emissions_long, metric='emissions')
        fig_emis.update_layout(xaxis_title="Year", yaxis_title="Emissions (ktCO₂e)")
        st.plotly_chart(fig_emis, use_container_width=True)

    # Emission Intensity with overlays
    st.markdown("### ⚖️ Emission Intensity (ktCO₂e/ktoe): Historical + Forecast to 2030")
    if not intensity_long.empty:
        fig_int = go.Figure()
        for t in intensity_long["Type"].unique():
            cur = intensity_long[intensity_long['Type'] == t]
            fig_int.add_trace(go.Scatter(x=cur['Year'], y=cur['Value'], mode="lines+markers", name=f"Intensity ({t})"))
        if show_policy:
            fig_int = add_policy_overlays(fig_int, selected_sector, selected_end_use, intensity_long, metric='emissions')
        fig_int.update_layout(xaxis_title="Year", yaxis_title="ktCO₂e/ktoe")
        st.plotly_chart(fig_int, use_container_width=True)

    with st.expander("Model diagnostics"):
        st.write("Simple ARIMA(1,1,1); horizon capped at 2030.")
        colA, colB = st.columns(2)
        with colA:
            if st.checkbox("Show ARIMA model summary — Emissions"):
                st.text(emissions_model.summary().as_text() if emissions_model is not None else "Unavailable")
        with colB:
            if st.checkbox("Show ARIMA model summary — Emission Intensity"):
                st.text(intensity_model.summary().as_text() if intensity_model is not None else "Unavailable")

elif selected_tab == "Simulations":
    st.subheader("⚡ Simulation Scenarios")
    sim_type = st.selectbox("Choose Scenario", ["Electrification of Appliances", "Services Rebound", "Targeted Emission Cut"])

    df_sim = df.copy()
    if sim_type == "Electrification of Appliances":
        df_sim.loc[(df_sim["Sector"].str.contains("Domestic", case=False, na=False)) & (df_sim["End Use"].str.contains("Appliance", case=False, na=False)), "Energy Consumption (ktoe)"] *= 0.8
        st.write("⚙️ 20% energy savings from appliance electrification.")
        fig = px.line(df_sim[df_sim["Sector"].str.contains("Domestic", case=False, na=False)], x="Year", y="Energy Consumption (ktoe)", color="End Use", title="Domestic energy with appliance electrification")
        base_long = df_sim[(df_sim['sector_norm'] == 'domestic/residential') & (df_sim['end_use_norm'] == 'appliances')][['Year','Energy Consumption (ktoe)']].rename(columns={'Energy Consumption (ktoe)':'Value'})
        base_long['Type'] = 'Scenario'
        fig = add_policy_overlays(fig, 'Domestic', 'Appliances', base_long, metric='energy')
        st.plotly_chart(fig, use_container_width=True)

    elif sim_type == "Services Rebound":
        df_sim.loc[df_sim["Sector"].str.contains("Services|Commercial", case=False, na=False), "Energy Consumption (ktoe)"] *= 1.2
        st.write("📈 20% energy rebound in the Services sector.")
        fig = px.line(df_sim[df_sim["Sector"].str.contains("Services|Commercial", case=False, na=False)], x="Year", y="Energy Consumption (ktoe)", color="End Use", title="Services energy with rebound")
        base_long = df_sim[df_sim['sector_norm'] == 'services/commercial'][['Year','Energy Consumption (ktoe)']].groupby('Year').sum().reset_index().rename(columns={'Energy Consumption (ktoe)':'Value'})
        base_long['Type'] = 'Scenario'
        fig = add_policy_overlays(fig, 'Services', 'HVAC & Lighting', base_long, metric='energy')
        st.plotly_chart(fig, use_container_width=True)

    elif sim_type == "Targeted Emission Cut":
        df_sim.loc[(df_sim["Sector"].str.contains("Industrial", case=False, na=False)) & (df_sim["End Use"].str.contains("Process", case=False, na=False)), "Emissions (ktCO2e)"] *= 0.75
        st.write("🔻 25% cut in emissions for industrial process heating.")
        fig = px.line(df_sim[df_sim["Sector"].str.contains("Industrial", case=False, na=False)], x="Year", y="Emissions (ktCO2e)", color="End Use", title="Industrial emissions with targeted cut")
        base_long = df_sim[(df_sim['sector_norm'] == 'industrial') & (df_sim['end_use_norm'] == 'process heat')][['Year','Emissions (ktCO2e)']].rename(columns={'Emissions (ktCO2e)':'Value'})
        base_long['Type'] = 'Scenario'
        fig = add_policy_overlays(fig, 'Industrial', 'Process heating', base_long, metric='emissions')
        st.plotly_chart(fig, use_container_width=True)

    # ---- Custom intervention ----
    st.subheader("🧪 Intervention & Rebound Simulation")
    sel_sector = st.selectbox("Select Sector", sectors, key="sim_sector")
    sel_end_use = st.selectbox("Choose End Use", end_uses, key="sim_end_use")
    local = df[(df['Sector'] == sel_sector) & (df['End Use'] == sel_end_use)].copy()

    st.markdown("**Scenario**: Reduce energy by 20% from 2023 onwards")
    cutoff_year = 2023
    mask = local['Year'] >= cutoff_year
    local.sort_values('Year', inplace=True)
    if 'Energy Consumption (ktoe)' in local.columns:
        local.loc[mask, 'Energy Consumption (ktoe)'] *= 0.8
    if {'Emissions (ktCO2e)', 'Energy Consumption (ktoe)'} <= set(local.columns):
        intensity = (local['Emissions (ktCO2e)'] / local['Energy Consumption (ktoe)']).mean()
        local['Emissions (ktCO2e)'] = local['Energy Consumption (ktoe)'] * intensity

    fig4 = px.line(local, x='Year', y='Energy Consumption (ktoe)', title='Energy after 20% Reduction Scenario')
    base_long_energy = local[['Year','Energy Consumption (ktoe)']].rename(columns={'Energy Consumption (ktoe)':'Value'})
    base_long_energy['Type'] = 'Scenario'
    fig4 = add_policy_overlays(fig4, sel_sector, sel_end_use, base_long_energy, metric='energy')
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.line(local, x='Year', y='Emissions (ktCO2e)', title='Emissions Impact (Scenario)')
    base_long_emis = local[['Year','Emissions (ktCO2e)']].rename(columns={'Emissions (ktCO2e)':'Value'})
    base_long_emis['Type'] = 'Scenario'
    fig5 = add_policy_overlays(fig5, sel_sector, sel_end_use, base_long_emis, metric='emissions')
    st.plotly_chart(fig5, use_container_width=True)

    # ---- 2030 Waterfall ----
    st.subheader("📉 2030 Net Emissions vs BAU — Policy Lever Contributions")

    with st.expander("Overlay debug / status"):
        st.write("CSV path:", st.session_state.get('__policy_csv_path', 'not found'))
        st.write("JSON path:", st.session_state.get('__policy_json_path', 'not found'))
        st.write("Overlay rows:", len(policy_overlays))
        if not policy_overlays.empty:
            st.dataframe(policy_overlays[['sector','end_use','year','multiplier_emissions']].head(12))
            st.write("Unique overlay pairs (normalized):", policy_overlays[['sector_norm','end_use_norm']].drop_duplicates().values.tolist())

    levers = [
        {"name":"Domestic Space Heating (BUS/GBIS)",       "ov_sector":"domestic/residential", "ov_end":"space heating"},
        {"name":"Domestic Appliances (Electrification)",   "ov_sector":"domestic/residential", "ov_end":"appliances"},
        {"name":"Industrial Process Heat (CCUS)",          "ov_sector":"industrial",            "ov_end":"process heat"},
        {"name":"Services HVAC & Lighting (MEES/PSDS)",    "ov_sector":"services/commercial",  "ov_end":"hvac & lighting"},
    ]

    contrib_names, contrib_vals = [], []
    missing = []
    for lv in levers:
        bau2030 = bau2030_norm(df, lv['ov_sector'], lv['ov_end'])
        if not np.isfinite(bau2030):
            missing.append((lv['name'], 'no BAU match in dataset (even after fallback)'))
            continue
        row2030 = get_overlay_row(lv['ov_sector'], lv['ov_end'], 2030)
        if row2030 is None:
            missing.append((lv['name'], 'no overlay row at/near 2030'))
            continue
        mult = float(row2030.get('multiplier_emissions', 1.0))
        overlay_val = bau2030 * mult
        delta = bau2030 - overlay_val
        if np.isfinite(delta) and abs(delta) > 0:
            contrib_names.append(lv['name'])
            contrib_vals.append(-delta)

    if contrib_names:
        measures = ["relative"] * len(contrib_names) + ["total"]
        xs = contrib_names + ["Net reduction"]
        ys = contrib_vals + [0]
        figw = go.Figure(go.Waterfall(
            name="Policy contributions",
            orientation="v",
            measure=measures,
            x=xs,
            y=ys,
            text=[f"{v:.0f}" for v in contrib_vals] + [""],
            textposition="outside",
            connector={"line": {"dash": "dot"}}
        ))
        figw.update_layout(
            xaxis_title="Lever",
            yaxis_title="Δ Emissions vs BAU in 2030 (ktCO₂e)",
            waterfallgap=0.3,
            showlegend=False,
        )
        figw.update_yaxes(zeroline=True, zerolinewidth=1)
        st.plotly_chart(figw, use_container_width=True)
    else:
        st.info("No matching overlays found to compute the waterfall. Confirm CSV paths and ensure 2030 rows exist for the four overlay pairs (see debug panel).")

    if missing:
        with st.expander("Why some bars are missing?"):
            for name, reason in missing:
                st.write(f"• {name}: {reason}")

elif selected_tab == "SHAP Explainability":
    st.subheader("🧠 SHAP Explainability for Emissions")

    df_shap = df.dropna(subset=["Energy Consumption (ktoe)", "Annual_HDD", "Emissions (ktCO2e)"]).copy()
    df_shap["Emission Intensity"] = df_shap["Emissions (ktCO2e)"] / df_shap["Energy Consumption (ktoe)"]
    df_shap.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_shap.dropna(inplace=True)

    df_encoded = pd.get_dummies(df_shap, columns=["Sector"], drop_first=True)

    features = ['Energy Consumption (ktoe)', 'Annual_HDD', 'Emission Intensity'] + [col for col in df_encoded.columns if col.startswith("Sector_")]
    X = df_encoded[features]
    y = df_encoded["Emissions (ktCO2e)"]

    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    st.write("Feature Importance (SHAP Summary)")
    fig, _ = plt.subplots()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig)

    st.subheader("🔍 SHAP Interpretability")
    features2 = ['Energy Consumption (ktoe)', 'Annual_HDD']
    df_shap2 = df.dropna(subset=features2 + ['Emissions (ktCO2e)'])
    X2 = df_shap2[features2]
    y2 = df_shap2['Emissions (ktCO2e)']
    model2 = xgb.XGBRegressor()
    model2.fit(X2, y2)
    explainer2 = shap.Explainer(model2)
    shap_vals2 = explainer2(X2)

    selected_row = st.slider("Select Row Index", 0, len(X2)-1, 0)
    st.write("Displaying SHAP force plot for selected row:")
    try:
        force_obj = shap.plots.force(shap_vals2[selected_row], matplotlib=False)
        html = shap.getjs() + force_obj.html()
        st.components.v1.html(html, height=400)
    except Exception as e:
        st.info(f"Interactive force plot unavailable ({e}). Showing a static waterfall instead.")
        fig2, _ = plt.subplots()
        shap.plots.waterfall(shap_vals2[selected_row], show=False)
        st.pyplot(fig2)

elif selected_tab == "EDA":
    st.subheader("📊 Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.markdown("### 📈 Distribution Plots")
    selected_dist_col = st.selectbox("Select Numeric Column for Distribution", numeric_cols)
    fig_dist, ax = plt.subplots()
    sns.histplot(df[selected_dist_col], kde=True, ax=ax)
    st.pyplot(fig_dist)

    st.markdown("### 🔗 Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

    with st.expander("Summary tables"):
        yearly_summary = df.groupby("Year").agg({
            "Energy Consumption (ktoe)": "sum",
            "Emissions (ktCO2e)": "sum"
        }).reset_index()
        st.dataframe(yearly_summary)
        summary_stats = df[[c for c in numeric_cols if c != "Year"]].describe().T
        st.dataframe(summary_stats)
        st.write("**Year Range:**", int(df["Year"].min()), "to", int(df["Year"].max()))

elif selected_tab == "Sector Overview":
    st.header("🔭 Sector-Level Overview")
    st.subheader("📊 Sectoral Trends and Emission Intensity")

    selected_sector = st.selectbox("Select Sector", sectors, key="sector_overview")

    sector_df = df[df['Sector'] == selected_sector].sort_values(by="Year")
    sector_df = sector_df[sector_df['Energy Consumption (ktoe)'] > 0]
    sector_df['Emission Intensity'] = sector_df['Emissions (ktCO2e)'] / sector_df['Energy Consumption (ktoe)']
    sector_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    sector_df.dropna(subset=['Emission Intensity'], inplace=True)
    yearly_summary = sector_df.groupby("Year").agg({
        'Energy Consumption (ktoe)': 'sum',
        'Emissions (ktCO2e)': 'sum',
        'Emission Intensity': 'mean'
    }).reset_index()

    st.subheader("Energy & Emissions Over Time")
    fig1 = px.line(yearly_summary, x="Year", y="Energy Consumption (ktoe)", title="Energy Consumption")
    fig2 = px.line(yearly_summary, x="Year", y="Emissions (ktCO2e)", title="Carbon Emissions")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Emission Intensity Trend (ktCO₂e/ktoe)")
    fig3 = px.line(yearly_summary, x="Year", y="Emission Intensity", title="Emission Intensity")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📄 Summary Table")
    st.dataframe(yearly_summary)
    st.download_button("Download Summary CSV", yearly_summary.to_csv(index=False), file_name=f"{selected_sector}_summary.csv")
