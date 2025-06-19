# app.py

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from math import sqrt

st.set_page_config(page_title="Sales Forecasting App", layout="wide")

st.title("📈 Sales Forecasting Time Series App")
st.markdown("""
This app analyzes and forecasts store sales using a SARIMA model. It incorporates holiday and promotion effects, allowing store-wise forecasting.
""")

# Load CSV
df = pd.read_csv("predicting_sales_time_series.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df.set_index('date', inplace=True)

with st.expander("🔍 Preview Dataset"):
    st.dataframe(df.head())

# Sidebar selections
store_ids = df['store_id'].unique()
selected_store = st.sidebar.selectbox("Select Store ID", store_ids)
show_outliers = st.sidebar.checkbox("Remove Sales Outliers", value=True)
use_lags = st.sidebar.checkbox("Include Lag Features", value=True)

store_df = df[df['store_id'] == selected_store].copy()

# Remove outliers
if show_outliers:
    q1, q3 = store_df['sales'].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    store_df = store_df[(store_df['sales'] >= lower) & (store_df['sales'] <= upper)]

# Optional lag features
if use_lags:
    store_df['lag_1'] = store_df['sales'].shift(1)
    store_df.dropna(inplace=True)
    exog_vars = store_df[['is_holiday', 'promotion', 'lag_1']]
else:
    exog_vars = store_df[['is_holiday', 'promotion']]

target = store_df['sales']
train_size = int(len(store_df) * 0.85)
train, test = target[:train_size], target[train_size:]
exog_train, exog_test = exog_vars[:train_size], exog_vars[train_size:]

# Modeling
try:
    model = SARIMAX(train,
                    exog=exog_train,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)

    pred = results.get_prediction(start=test.index[0],
                                  end=test.index[-1],
                                  exog=exog_test,
                                  dynamic=False)
    forecast = pred.predicted_mean

    rmse = sqrt(mean_squared_error(test, forecast))
    st.success(f"✅ Model trained. RMSE: {rmse:.2f}")

    # Combine data for plotting
    train_test_df = pd.concat([train, test])
    train_test_index = train.index.to_list() + test.index.to_list()
    train_test_type = ["Train"] * len(train) + ["Test"] * len(test)

    forecast_plot_df = pd.DataFrame({
        "date": train_test_index,
        "sales": train_test_df.values,
        "type": train_test_type
    })

    forecast_df_plot = pd.DataFrame({
        "date": forecast.index,
        "sales": forecast.values,
        "type": ["Forecast"] * len(forecast)
    })

    combined_plot_df = pd.concat([forecast_plot_df, forecast_df_plot])

    # Plotly chart
    st.subheader("📊 Forecast vs Actual")
    fig = go.Figure()
    for category in combined_plot_df["type"].unique():
        subset = combined_plot_df[combined_plot_df["type"] == category]
        fig.add_trace(go.Scatter(
            x=subset["date"], y=subset["sales"],
            mode="lines", name=category
        ))
    fig.update_layout(
        title=f"Sales Forecast vs Actual (Store {selected_store})",
        xaxis_title="Date", yaxis_title="Sales",
        hovermode="x unified", template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Downloadable forecast
    forecast_df = pd.DataFrame({
        "date": forecast.index,
        "forecasted_sales": forecast.values,
        "actual_sales": test.values
    })
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast CSV", csv, "sales_forecast.csv", "text/csv")

except Exception as e:
    st.error(f"❌ Model training failed: {e}")
