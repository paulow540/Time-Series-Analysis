# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt

st.set_page_config(page_title="Sales Forecasting App", layout="wide")

st.title("📈 Sales Forecasting Time Series App")
st.markdown("Upload your sales data and forecast future sales using SARIMA with exogenous variables (holiday, promotion).")

# File uploader
uploaded_file = st.file_uploader("C:\\Users\\Administrator\\Desktop\\Femi\\predicting_sales_time_series.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.set_index('date')

    # Sidebar store selection
    store_ids = df['store_id'].unique()
    selected_store = st.sidebar.selectbox("Select Store ID", store_ids)

    store_df = df[df['store_id'] == selected_store]

    # Plot sales
    st.subheader(f"Sales Trend - Store {selected_store}")
    st.line_chart(store_df['sales'])

    # Forecasting
    exog_vars = store_df[['is_holiday', 'promotion']]
    target = store_df['sales']

    train_size = int(len(store_df) * 0.85)
    train, test = target[:train_size], target[train_size:]
    exog_train, exog_test = exog_vars[:train_size], exog_vars[train_size:]

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
        st.success(f"📊 RMSE: {rmse:.2f}")

        # Plot forecast vs actual
        st.subheader("Forecast vs Actual")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(train.index, train, label='Train')
        ax.plot(test.index, test, label='Test')
        ax.plot(forecast.index, forecast, label='Forecast', linestyle='--')
        ax.set_title('Sales Forecast vs Actual')
        ax.legend()
        st.pyplot(fig)

        # Forecast download
        forecast_df = pd.DataFrame({
            "date": forecast.index,
            "forecasted_sales": forecast.values,
            "actual_sales": test.values
        })

        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast CSV", csv, "sales_forecast.csv", "text/csv")

    except Exception as e:
        st.error(f"Model training failed: {e}")
