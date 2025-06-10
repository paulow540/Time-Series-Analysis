import streamlit as st

# Importing required libraries
import numpy as np
import pandas as pd, datetime
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import plotly.express as px
# get_ipython().run_line_magic('matplotlib', 'inline')
from time import time
import os
from math import sqrt
import scipy.stats as stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import  ARIMA
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf, pacf
from plotly.subplots import make_subplots


# from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from math import sqrt


st.title("Time Series Analysis Forecasting  and Predicting sales Analysis")

# Importing store data
store = pd.read_csv('C:\\Users\\Administrator\\Desktop\\Femi\\rossmann-store-sales\\store.csv')
# Importing train data
train = pd.read_csv('C:\\Users\\Administrator\\Desktop\\Femi\\rossmann-store-sales\\train.csv', index_col='Date', parse_dates = True)
# Importing test data
test = pd.read_csv('C:\\Users\\Administrator\\Desktop\\Femi\\rossmann-store-sales\\test.csv')


# Extracting year, month, day and week, and making new column
train['Year'] = train.index.year
train['Month'] = train.index.month
train['Day'] = train.index.day
train['WeekOfYear'] = train.index.isocalendar().week

train['SalePerCustomer'] = train['Sales']/train['Customers']



# Checking data when the stores were closed
train_store_closed = train[(train.Open == 0)]



# Checking days when the stores were closed
train_store_closed.hist('DayOfWeek')



# bar1, bar2 = st.tabs(["SchoolHoliday","StateHoliday"])

# with bar1:
#     # Checking whether there was a school holiday when the store was closed
#     st.write("Checking whether there was a school holiday when the store was closed")
#     tsc_schoolHoliday = train_store_closed['SchoolHoliday'].value_counts() #.plot(kind='bar')
#     st.bar_chart(tsc_schoolHoliday )

# with bar2:
#     # Checking whether there was a state holiday when the store was closed
#     st.write("Checking whether there was a state holiday when the store was closed")
#     tsc_stateHoliday = train_store_closed['StateHoliday'].value_counts() #.plot(kind='bar')
#     st.bar_chart(tsc_stateHoliday )


# No. of days with closed stores
# train[(train.Open == 0)].shape[0]

# No. of days when store was opened but zero sales - might be because of external factors or refurbishmnent
# train[(train.Open == 1) & (train.Sales == 0)].shape[0]

# Replacing missing values for Competiton distance with median
store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace=True)

# No info about other columns - so replcae by 0
store.fillna(0, inplace=True)

# Assuming stores open in test
test.fillna(1, inplace=True)

# Joining the tables
with st.expander("click to check the stores dataset"):
    train_store_joined = pd.merge(train, store, on='Store', how='inner')


    # Distribution of sales and customers across store types
    train_store_joined.groupby('StoreType')[['Customers', 'Sales', 'SalePerCustomer']].sum().sort_values('Sales', ascending=False)

    # fig = px.bar(train_store_joined, 
    #              x= train_store_joined.index,
    #              y= "Sales",                     
    #              )


    # Closed and zero-sales obseravtions
    train_store_joined[(train_store_joined.Open ==0) | (train_store_joined.Sales==0)]

    # Open & Sales >0 stores
    train_store_joined_open = train_store_joined[~((train_store_joined.Open ==0) | (train_store_joined.Sales==0))]


bar1, bar2, bar3= st.tabs(["Weekly Sales Trend for different store","Forecasting a Time Series", "Model Prediction and validation"])
with bar1:
    # Resample weekly sales data
    # Data Preparation: input should be float type
    train['Sales'] = train['Sales'] * 1.0
    two = st.sidebar.number_input("Enter a Weekly Sales Trend for  Stores 2", value=2)
    sales_a = train[train.Store == two]['Sales'].resample('W').sum()
    eight = st.sidebar.number_input("Enter a Weekly Sales Trend for  Stores 85", value=85)
    sales_b = train[train.Store == eight]['Sales'].resample('W').sum()
    one = st.sidebar.number_input("Enter a Weekly Sales Trend for  Stores 1", value=1)
    sales_c = train[train.Store == one]['Sales'].resample('W').sum()
    three = st.sidebar.number_input("Enter a Weekly Sales Trend for  Stores 13", value=13)
    sales_d = train[train.Store == three]['Sales'].resample('W').sum()


    # Create 4-row subplot
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=(f"Store {two}", f"Store {eight}", f"Store {one}", f"Store {three}"))

    fig.add_trace(go.Scatter(x=sales_a.index, y=sales_a.values, name=f"Store {two}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sales_b.index, y=sales_b.values, name=f"Store {eight}"), row=2, col=1)
    fig.add_trace(go.Scatter(x=sales_c.index, y=sales_c.values, name=f"Store {one}"), row=3, col=1)
    fig.add_trace(go.Scatter(x=sales_d.index, y=sales_d.values, name=f"Store {three}"), row=4, col=1)

    fig.update_layout(height=1000, width=900, title_text="Weekly Sales Trend for Selected Stores")
    st.plotly_chart(fig, use_container_width=True)



with bar2:
    # Streamlit version of the autocorrelation plot function
    def auto_corr(sales):
        lag_acf = acf(sales, nlags=30)
        lag_pacf = pacf(sales, nlags=20, method='ols')

        # ACF Plot
        fig1, ax1 = plt.subplots()
        ax1.plot(lag_acf)
        ax1.axhline(y=0, linestyle='--', color='red')
        ax1.axhline(y=1.96 / np.sqrt(len(sales)), linestyle='--', color='red')
        ax1.axhline(y=-1.96 / np.sqrt(len(sales)), linestyle='--', color='red')
        ax1.set_title('ACF')
        st.pyplot(fig1)

        # PACF Plot
        fig2, ax2 = plt.subplots()
        ax2.plot(lag_pacf)
        ax2.axhline(y=0, linestyle='--', color='red')
        ax2.axhline(y=1.96 / np.sqrt(len(sales)), linestyle='--', color='red')
        ax2.axhline(y=-1.96 / np.sqrt(len(sales)), linestyle='--', color='red')
        ax2.set_title('PACF')
        st.pyplot(fig2)

        st.title("Autocorrelation and Partial Autocorrelation")

        # Simulated or uploaded time series data
        # You can replace this with file upload logic
        sales = pd.Series(np.random.randn(100))

    # # Summing sales on per week basis
    # Only select numeric columns before resampling
    train_arima = train.select_dtypes(include='number').resample("W").mean()

    # Focus on just the 'Sales' column
    train_arima = train_arima[["Sales"]]

     # Define the p, d and q parameters to take any value between 0 and 3
    p = d = q = range(0, 2)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    if st.button("Show ACF & PACF"):
        auto_corr(sales_a)

  
with bar3:
    # Determing p,d,q combinations with AIC scores.
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(train_arima,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()

                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    # Fitting the data to SARIMA model
    model_sarima = sm.tsa.statespace.SARIMAX(train_arima,
                                    order=(1, 1, 1),
                                    seasonal_order=(0, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results_sarima = model_sarima.fit()

    
    st.dataframe(results_sarima.summary().tables[1])
    st.write("This table shows how well the model fits past sales patterns. Numbers in P>|z| show if effects are meaningful: small values (like 0.000) mean strong evidence. Large ones (like 0.184) mean weak or uncertain impact.")

     # Extract residuals
    residuals = results_sarima.resid

    # 1. Residual Time Series
    # fig1 = go.Figure()
    # fig1.add_trace(go.Scatter(y=residuals, mode='lines', name='Residuals'))
    # fig1.update_layout(title="Residuals over Time", xaxis_title="Time", yaxis_title="Residuals")

    # # 2. Histogram of Residuals
    # fig2 = px.histogram(residuals, nbins=30, title="Histogram of Residuals")

    # # 3. Q-Q Plot
    # qq = stats.probplot(residuals, dist="norm")
    # fig3 = go.Figure()
    # fig3.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Data'))
    # fig3.add_trace(go.Scatter(x=qq[0][0], y=qq[0][0], mode='lines', name='Ideal', line=dict(color='red')))
    # fig3.update_layout(title="Q-Q Plot", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")

    # # 4. ACF Plot of Residuals
    # acf_vals = acf(residuals, nlags=40)
    # fig4 = go.Figure()
    # fig4.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals))
    # fig4.update_layout(title="ACF of Residuals", xaxis_title="Lag", yaxis_title="ACF")

    # # Show all plots
    # fig1.show()
    # fig2.show()
    # fig3.show()
    # fig4.show()



    # Model Prediction and validation
    # Predictions are performed for the 11th Jan' 2015 onwards of the train data.
    date = st.date_input("Enter the time to predict", "2015-01-11")

    pred = results_sarima.get_prediction(start=pd.to_datetime(date), dynamic = False)

    # Get confidence intervals of forecasts
    # Generate predictions
    pred = results_sarima.get_prediction(start=pd.to_datetime(date), dynamic=False)
    pred_ci = pred.conf_int()

    # Extract forecasted and actual data
    forecasted = pred.predicted_mean
    actual = train_arima["2015-01-11":]

    # Create Plotly figure
    fig = go.Figure()

    # Observed values
    fig.add_trace(go.Scatter(
        x=train_arima["2014":].index,
        y=train_arima["2014":].values,
        mode='lines',
        name='Observed',
        line=dict(color='blue')
    ))

    # Predicted values
    fig.add_trace(go.Scatter(
        x=forecasted.index,
        y=forecasted.values,
        mode='lines',
        name='One-step ahead Forecast',
        line=dict(color='orange')
    ))

    # Confidence interval (shaded area)
    fig.add_trace(go.Scatter(
        x=pred_ci.index.tolist() + pred_ci.index[::-1].tolist(),
        y=pred_ci.iloc[:, 0].tolist() + pred_ci.iloc[:, 1][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(0,0,0,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='Confidence Interval'
    ))

    # Final layout tweaks
    fig.update_layout(
        title="SARIMA Forecast with Confidence Interval",
        xaxis_title="Date",
        yaxis_title="Sales",
        width=1000,
        height=600
    )

    fig.show()

    # RMSE Calculation
    rms_arima = sqrt(mean_squared_error(actual, forecasted))
    print("Root Mean Squared Error:", rms_arima)
