import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import xgboost as xgb
import shap
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    df = pd.read_csv("Merged_Dataset__Energy___Emissions___HDD.csv")
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.dropna(subset=['Year'], inplace=True)
    return df

df = load_data()
sectors = df['Sector'].dropna().unique().tolist()
end_uses = df['End Use'].dropna().unique().tolist()

st.set_page_config(page_title="Energy Dashboard", layout="wide")
st.title("Energy Dashboard: Trends, Forecasting, Simulations & SHAP Explainability")

# After loading data
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())
year_range = st.sidebar.slider("Filter Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]


with st.sidebar:
    st.title("⚡ Energy Dashboard")
    selected_tab = st.radio("Navigate", ["Sector Overview", "Trends", "Forecasting", "Simulations", "SHAP Explainability", "EDA"])

if selected_tab == "Trends":
    st.subheader("📊 Sectoral Trends and Emission Intensity")
    selected_sector = st.selectbox("Select Sector", sectors)
    df_sector = df[df['Sector'] == selected_sector]

    fig1 = px.line(df_sector, x="Year", y="Energy Consumption (ktoe)", color="End Use",
                   title=f"Energy Consumption Trends - {selected_sector}")
    st.plotly_chart(fig1)

    df_sector["Emission Intensity"] = df_sector["Emissions (ktCO2e)"] / df_sector["Energy Consumption (ktoe)"]
    fig2 = px.line(df_sector, x="Year", y="Emission Intensity", color="End Use",
                   title=f"Emission Intensity (ktCO2e/ktoe) - {selected_sector}")
    st.plotly_chart(fig2)

    if st.checkbox("Show Summary Statistics"):
        st.dataframe(df_sector)

    if st.checkbox("Download Merged Dataset"):
        csv = df.to_csv(index=False).encode()
        st.download_button("Download as CSV", csv, "merged_energy_emissions.csv", "text/csv")

if selected_tab == "Forecasting":
    st.subheader("🔮 Forecast to 2030: Emissions & Emission Intensity (Historical + Forecast)")

    selected_sector = st.selectbox("Select Sector", sectors, key="forecast_sector")
    selected_end_use = st.selectbox("Choose End Use", end_uses)

    # 1) Prepare series for the selected (sector, end_use)
    df_series = df[(df['Sector'] == selected_sector) & (df['End Use'] == selected_end_use)].copy()
    df_series = df_series.sort_values("Year")

    # Guard against zero division and invalid values for intensity
    df_series = df_series[df_series['Energy Consumption (ktoe)'] > 0]
    df_series["Emission Intensity"] = df_series["Emissions (ktCO2e)"] / df_series["Energy Consumption (ktoe)"]
    df_series.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_series.dropna(subset=["Emissions (ktCO2e)", "Emission Intensity"], inplace=True)

    if df_series.empty:
        st.info("No valid rows after filtering (check selected sector/end use and year range).")
        st.stop()

    # Context line
    st.write(
        f"**Context:** {selected_sector} → {selected_end_use}. "
        f"Historical years: {int(df_series['Year'].min())}–{int(df_series['Year'].max())}. "
        f"Target horizon: 2030."
    )

    # 2) Helper: forecast a single metric to 2030 (returns long-form Historical+Forecast AND the fitted model)
    def forecast_to_2030(series_df: pd.DataFrame, value_col: str):
        """
        Fit ARIMA(1,1,1) on the given value_col and forecast to 2030 using 'Year' as the time index.
        Returns: (long_df, fitted_model_or_None)
        """
        if series_df.empty or value_col not in series_df.columns:
            return pd.DataFrame(columns=["Year", "Value", "Type", "Metric"]), None

        ts = series_df[["Year", value_col]].dropna()
        if ts.empty:
            return pd.DataFrame(columns=["Year", "Value", "Type", "Metric"]), None

        ts = ts.set_index(pd.Index(ts["Year"], name="Year"))[[value_col]]

        last_year = int(ts.index.max())
        steps = 2030 - last_year

        # Historical part (always returned)
        hist_df = ts.reset_index().rename(columns={value_col: "Value"})
        hist_df["Type"] = "Historical"
        hist_df["Metric"] = value_col

        fitted = None
        if len(ts.dropna()) >= 5:
            # Fit model for summary even if no forecast horizon remains
            try:
                model = ARIMA(ts, order=(1, 1, 1))
                fitted = model.fit()
            except Exception as e:
                st.warning(f"ARIMA fit failed for {value_col}: {e}")

        # Forecast only if horizon remains and we have a fitted model
        if steps > 0 and fitted is not None:
            try:
                fc = fitted.forecast(steps=steps)
                fc_years = np.arange(last_year + 1, 2031)
                fc_df = pd.DataFrame({
                    "Year": fc_years,
                    "Value": fc.values,
                    "Type": "Forecast",
                    "Metric": value_col
                })
                out = pd.concat([hist_df, fc_df], ignore_index=True)
                return out, fitted
            except Exception as e:
                st.warning(f"Forecast failed for {value_col}: {e}")
                return hist_df, fitted

        return hist_df, fitted

    # 3) Build combined long-form data + models for summaries
    emissions_long, emissions_model   = forecast_to_2030(df_series, "Emissions (ktCO2e)")
    intensity_long, intensity_model   = forecast_to_2030(df_series, "Emission Intensity")

    # 4) Plot Emissions (historical + forecast)
    st.markdown("### 📉 Emissions (ktCO₂e): Historical + Forecast to 2030")
    if not emissions_long.empty:
        fig_emis = px.line(
            emissions_long, x="Year", y="Value", color="Type",
            title=f"{selected_end_use} — Emissions Forecast to 2030 ({selected_sector})",
            markers=True
        )
        st.plotly_chart(fig_emis, use_container_width=True)
    else:
        st.info("No emissions data available for the selected filters.")

    # 5) Plot Emission Intensity (historical + forecast)
    st.markdown("### ⚖️ Emission Intensity (ktCO₂e/ktoe): Historical + Forecast to 2030")
    if not intensity_long.empty:
        fig_int = px.line(
            intensity_long, x="Year", y="Value", color="Type",
            title=f"{selected_end_use} — Emission Intensity Forecast to 2030 ({selected_sector})",
            markers=True
        )
        st.plotly_chart(fig_int, use_container_width=True)
    else:
        st.info("No emission intensity data available for the selected filters.")

    # 6) Optional diagnostics + input peek
    with st.expander("Model diagnostics"):
        st.write("Simple ARIMA(1,1,1) used for both series; horizon capped at 2030.")
        colA, colB = st.columns(2)

        with colA:
            if st.checkbox("Show ARIMA model summary — Emissions"):
                if emissions_model is not None:
                    st.text(emissions_model.summary().as_text())
                else:
                    st.info("Emissions model summary unavailable (insufficient data or fit failed).")

        with colB:
            if st.checkbox("Show ARIMA model summary — Emission Intensity"):
                if intensity_model is not None:
                    st.text(intensity_model.summary().as_text())
                else:
                    st.info("Intensity model summary unavailable (insufficient data or fit failed).")

    if st.checkbox("Show Summary Table"):
        st.dataframe(df_series)
    if st.checkbox("Download Merged Dataset"):
        csv = df.to_csv(index=False).encode()
        st.download_button("Download as CSV", csv, "merged_energy_emissions.csv", "text/csv")

elif selected_tab == "Simulations":
    st.subheader("⚡ Simulation Scenarios")
    sim_type = st.selectbox("Choose Scenario", ["Electrification of Appliances", "Services Rebound", "Targeted Emission Cut"])

    if sim_type == "Electrification of Appliances":
        df_sim = df.copy()
        df_sim.loc[(df_sim["Sector"] == "Domestic") & (df_sim["End Use"] == "Appliances"), "Energy Consumption (ktoe)"] *= 0.8
        st.write("⚙️ 20% energy savings from appliance electrification.")
        fig = px.line(df_sim[df_sim["Sector"] == "Domestic"], x="Year", y="Energy Consumption (ktoe)", color="End Use")
        st.plotly_chart(fig)

    elif sim_type == "Services Rebound":
        df_sim = df.copy()
        df_sim.loc[df_sim["Sector"] == "Services", "Energy Consumption (ktoe)"] *= 1.2
        st.write("📈 20% energy rebound in the Services sector.")
        fig = px.line(df_sim[df_sim["Sector"] == "Services"], x="Year", y="Energy Consumption (ktoe)", color="End Use")
        st.plotly_chart(fig)

    elif sim_type == "Targeted Emission Cut":
        df_sim = df.copy()
        df_sim.loc[(df_sim["Sector"] == "Industrial") & (df_sim["End Use"] == "Process heating"), "Emissions (ktCO2e)"] *= 0.75
        st.write("🔻 25% cut in emissions for industrial process heating.")
        fig = px.line(df_sim[df_sim["Sector"] == "Industrial"], x="Year", y="Emissions (ktCO2e)", color="End Use")
        st.plotly_chart(fig)

    st.subheader("🧪 Intervention & Rebound Simulation")
    selected_sector = st.selectbox("Select Sector", sectors, key="sim_sector")
    selected_end_use = st.selectbox("Choose End Use", end_uses, key="sim_end_use")
    df_sim = df[(df['Sector'] == selected_sector) & (df['End Use'] == selected_end_use)].copy()

    st.markdown("**Scenario**: Reduce energy by 20% from 2023 onwards")
    cutoff_year = 2023
    mask = df_sim['Year'] >= cutoff_year
    df_sim.loc[mask, 'Energy Consumption (ktoe)'] *= 0.8
    df_sim['Emissions (ktCO2e)'] = df_sim['Energy Consumption (ktoe)'] *         (df_sim['Emissions (ktCO2e)'] / df_sim['Energy Consumption (ktoe)']).mean()

    fig4 = px.line(df_sim, x='Year', y='Energy Consumption (ktoe)', title='Energy after 20% Reduction Scenario')
    st.plotly_chart(fig4)
    fig5 = px.line(df_sim, x='Year', y='Emissions (ktCO2e)', title='Emissions Impact (Scenario)')
    st.plotly_chart(fig5)

    if st.checkbox("Show Summary Statistics"):
        st.dataframe(df_sim)

    if st.checkbox("Download Merged Dataset"):
        csv = df.to_csv(index=False).encode()
        st.download_button("Download as CSV", csv, "merged_energy_emissions.csv", "text/csv")


if selected_tab == "SHAP Explainability":
    st.subheader("🧠 SHAP Explainability for Emissions")

    # Create Emission Intensity feature
    df_shap = df.dropna(subset=["Energy Consumption (ktoe)", "Annual_HDD", "Emissions (ktCO2e)"]).copy()
    df_shap["Emission Intensity"] = df_shap["Emissions (ktCO2e)"] / df_shap["Energy Consumption (ktoe)"]

    # One-hot encode sector
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
    fig, ax = plt.subplots()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig)


    st.subheader("🔍 SHAP Interpretability")

    features = ['Energy Consumption (ktoe)', 'Annual_HDD']
    df_shap = df.dropna(subset=features + ['Emissions (ktCO2e)'])

    X = df_shap[features]
    y = df_shap['Emissions (ktCO2e)']
    model = xgb.XGBRegressor()
    model.fit(X, y)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    selected_row = st.slider("Select Row Index", 0, len(X)-1, 0)
    st.write("Displaying SHAP force plot for selected row:")
    try:
        # Use JS-based force plot and embed the JS bundle explicitly
        force_obj = shap.plots.force(shap_values[selected_row], matplotlib=False)
        html = shap.getjs() + force_obj.html()
        st.components.v1.html(html, height=400)
    except Exception as e:
        # Fallback to a static, thread-safe matplotlib waterfall plot
        st.info(f"Interactive force plot unavailable ({e}). Showing a static waterfall instead.")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[selected_row], show=False)
        st.pyplot(fig)


    if st.checkbox("Show Summary Statistics"):
        st.dataframe(df_shap)

    if st.checkbox("Download Merged Dataset"):
        csv = df.to_csv(index=False).encode()
        st.download_button("Download as CSV", csv, "merged_energy_emissions.csv", "text/csv")

if selected_tab == "EDA":
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

    if st.checkbox("Show Summary"):
        yearly_summary = df.groupby("Year").agg({
            "Energy Consumption (ktoe)": "sum",
            "Emissions (ktCO2e)": "sum"
        }).reset_index()
        st.dataframe(yearly_summary)


    if st.checkbox("Show Summary Statistics"):
        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "Year"]
        summary_stats = df[numeric_cols].describe().T

        st.dataframe(summary_stats)

        # Show year range separately
        st.write("**Year Range:**", int(df["Year"].min()), "to", int(df["Year"].max()))


    if st.checkbox("Download Dataset"):
        csv = df.to_csv(index=False).encode()
        st.download_button("Download as CSV", csv, "your_dataset.csv", "text/csv")


if selected_tab == "Sector Overview":
    st.header("🔭 Sector-Level Overview")
    st.subheader("📊 Sectoral Trends and Emission Intensity")
    
    selected_sector = st.selectbox("Select Sector", sectors, key="sector_overview")

    # Filtered data for sector
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

    # Line charts
    st.subheader("Energy & Emissions Over Time")
    fig1 = px.line(yearly_summary, x="Year", y="Energy Consumption (ktoe)", title="Energy Consumption")
    fig2 = px.line(yearly_summary, x="Year", y="Emissions (ktCO2e)", title="Carbon Emissions")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

    # Emission intensity trend
    st.subheader("Emission Intensity Trend (ktCO₂e/ktoe)")
    fig3 = px.line(yearly_summary, x="Year", y="Emission Intensity", title="Emission Intensity")
    st.plotly_chart(fig3, use_container_width=True)

    # Year-wise Summary Table
    st.subheader("📄 Summary Table")
    st.dataframe(yearly_summary)
    st.download_button("Download Summary CSV", yearly_summary.to_csv(index=False), file_name=f"{selected_sector}_summary.csv")

    
# SHAP Summary (sectoral influence)
st.subheader("💡 SHAP Feature Importance Summary (Emissions Prediction)")
try:
    df['Emission Intensity'] = df['Emissions (ktCO2e)'] / df['Energy Consumption (ktoe)']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Emission Intensity'], inplace=True)

    feature_cols = ['Energy Consumption (ktoe)', 'Annual_HDD', 'Emission Intensity'] + [col for col in df.columns if col.startswith("Sector_")]
    df_encoded = pd.get_dummies(df, columns=['Sector'], prefix='Sector')
    y = df_encoded['Emissions (ktCO2e)']

    missing = [col for col in feature_cols if col not in df_encoded.columns]
    if missing:
        st.warning(f"SHAP input missing columns: {missing}")
    else:
        X = df_encoded[feature_cols]
        model = xgb.XGBRegressor()
        model.fit(X, y)
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        st.write("SHAP Summary Plot:")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(fig)
except Exception as e:
        st.warning(f"SHAP plot failed: {e}")