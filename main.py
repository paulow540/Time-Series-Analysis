import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from math import sqrt

st.set_page_config(page_title="Sales Forecasting App", layout="wide")

st.title("📈 Sales Forecasting Time Series App")
st.markdown("""
This app performs time series forecasting using SARIMA based on store-level sales, holidays, and promotions.
Upload your dataset or use the default sample to get started.
""")

# Load dataset
df = pd.read_csv("predicting_sales_time_series.csv")

with st.expander("📄 Sample of Dataset"):
    st.write(df.head())

# --- Data Cleaning ---
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df = df.set_index('date')

# Optional: remove outliers in sales
q_low, q_high = df['sales'].quantile([0.01, 0.99])
df = df[(df['sales'] >= q_low) & (df['sales'] <= q_high)]

# Feature engineering: Day of Week (optional)
df['day_of_week'] = df.index.dayofweek  # 0 = Monday

# Sidebar store selection
store_ids = df['store_id'].unique()
selected_store = st.sidebar.selectbox("🏪 Select Store ID", store_ids)

store_df = df[df['store_id'] == selected_store]

# Plot sales
st.subheader(f"📊 Sales Trend - Store {selected_store}")
st.line_chart(store_df['sales'])

# Forecasting setup
exog_vars = store_df[['is_holiday', 'promotion']]
target = store_df['sales']

train_size = int(len(store_df) * 0.85)
train, test = target[:train_size], target[train_size:]
exog_train, exog_test = exog_vars[:train_size], exog_vars[train_size:]

try:
    # SARIMA model
    model = SARIMAX(train,
                    exog=exog_train,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)

    # pred = results.get_prediction(start=test.index[0],
    #                               end=test.index[-1],
    #                               exog=exog_test,
    #                               dynamic=False)
    pred = results.get_prediction(start=len(train),
                              end=len(train) + len(test) - 1,
                              exog=exog_test,
                              dynamic=False)

    forecast = pred.predicted_mean

    rmse = sqrt(mean_squared_error(test, forecast))
    st.success(f"📉 RMSE: {rmse:.2f}")

    # Combine for plot
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

    # Plot with Plotly
    st.subheader("📈 Forecast vs Actual (Interactive)")
    fig = go.Figure()

    for category in combined_plot_df["type"].unique():
        subset = combined_plot_df[combined_plot_df["type"] == category]
        fig.add_trace(go.Scatter(
            x=subset["date"], y=subset["sales"],
            mode="lines",
            name=category
        ))

    fig.update_layout(
        title=f"Sales Forecast vs Actual (Store {selected_store})",
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Download forecast
    forecast_df = pd.DataFrame({
        "date": forecast.index,
        "forecasted_sales": forecast.values,
        "actual_sales": test.values
    })

    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast CSV", csv, "sales_forecast.csv", "text/csv")

except Exception as e:
    st.error(f"⚠️ Model training failed: {e}")
