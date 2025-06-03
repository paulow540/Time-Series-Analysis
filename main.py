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
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import  ARIMA
from datetime import datetime
# from sklearn import model_selection
# from sklearn.metrics import mean_squared_error, r2_score
# from pandas import DataFrame
# import xgboost as xgb
# # from fbprophet import Prophet
# import warnings
# warnings.filterwarnings('ignore')


st.title("Time Series Analysis Forecasting  and Predicting sales Analysis")

# Importing store data
store = pd.read_csv('store.csv')
# Importing train data
train = pd.read_csv('train.csv', index_col='Date', parse_dates = True)
# Importing test data
test = pd.read_csv('test.csv')


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



bar1, bar2 = st.tabs(["SchoolHoliday","StateHoliday"])

with bar1:
    # Checking whether there was a school holiday when the store was closed
    st.write("Checking whether there was a school holiday when the store was closed")
    tsc_schoolHoliday = train_store_closed['SchoolHoliday'].value_counts() #.plot(kind='bar')
    st.bar_chart(tsc_schoolHoliday )

with bar2:
    # Checking whether there was a state holiday when the store was closed
    st.write("Checking whether there was a state holiday when the store was closed")
    tsc_stateHoliday = train_store_closed['StateHoliday'].value_counts() #.plot(kind='bar')
    st.bar_chart(tsc_stateHoliday )


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
train_store_joined = pd.merge(train, store, on='Store', how='inner')
train_store_joined.head()

# Distribution of sales and customers across store types
train_store_joined.groupby('StoreType')[['Customers', 'Sales', 'SalePerCustomer']].sum().sort_values('Sales', ascending=False)

fig = px.bar(train_store_joined, 
             x= train_store_joined.index,
             y= "Sales",         
             
             
             )


# Closed and zero-sales obseravtions
train_store_joined[(train_store_joined.Open ==0) | (train_store_joined.Sales==0)]

# Open & Sales >0 stores
train_store_joined_open = train_store_joined[~((train_store_joined.Open ==0) | (train_store_joined.Sales==0))]



# # Only use numeric columns for correlation
# numeric_data = train_store_joined.select_dtypes(include='number')
# # Plotly heatmap
# fig = px.imshow(
#     numeric_data.corr(),
#     text_auto=True,
#     aspect="auto",
#     color_continuous_scale="RdBu_r",
#     title="Correlation Heatmap of Numeric Features",
# )

# Data Preparation: input should be float type
train['Sales'] = train['Sales'] * 1.0



# Assigning one store from each category



import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Resample weekly sales data
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
                    subplot_titles=("Store 2", "Store 85", "Store 1", "Store 13"))

fig.add_trace(go.Scatter(x=sales_a.index, y=sales_a.values, name="Store 2"), row=1, col=1)
fig.add_trace(go.Scatter(x=sales_b.index, y=sales_b.values, name="Store 85"), row=2, col=1)
fig.add_trace(go.Scatter(x=sales_c.index, y=sales_c.values, name="Store 1"), row=3, col=1)
fig.add_trace(go.Scatter(x=sales_d.index, y=sales_d.values, name="Store 13"), row=4, col=1)

fig.update_layout(height=1000, width=900, title_text="Weekly Sales Trend for Selected Stores")
st.plotly_chart(fig, use_container_width=True)






















# # Show in Streamlit
# st.plotly_chart(fig, use_container_width=True)

# sale = st.sidebar.slider("Sales",train_store_joined_open["Sales"].min(), train_store_joined_open["Sales"].min(),value=100.2)

# fig = px.line(
#     data_frame=train_store_joined_open,
#     x="Month",
#     y="Sales",
#     color="Promo",     # Facet by 'Promo' horizontally
#     facet_col="Promo2",    # Differentiate by 'Promo2'
#     facet_row="Year",      # Facet by 'Year' vertically
#                          # You can change to "strip", "violin", etc.
#     markers=True 
# )
# # # Show in Streamlit
# st.plotly_chart(fig, use_container_width=True)

# Plot with Plotly Express
# fig = px.scatter(
#     train_store_joined_open,
#     x="Month",
#     y="Sales",
#     color="Promo2",
#     facet_col="Promo",
#     facet_row="Year",
#     title="Sales by Month, Promo, and Year",
#     trendline="lowess",  # Optional: adds a smoothed line
#     height=800
# )

# # Show in Streamlit
# st.plotly_chart(fig, use_container_width=True)

# ===============================================================================
#  Time Series Analysis & Predictive Modelling

# pd.plotting.register_matplotlib_converters()

# # Data Preparation: input should be float type
# train['Sales'] = train['Sales'] * 1.0

# # Assigning one store from each category
# sales_a = train[train.Store == 2]['Sales']
# sales_b = train[train.Store == 85]['Sales'].sort_index(ascending = True)
# sales_c = train[train.Store == 1]['Sales']
# sales_d = train[train.Store == 13]['Sales']

# f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (12, 13))

# # Trend
# sales_a.resample('W').sum().plot(ax = ax1)
# sales_b.resample('W').sum().plot(ax = ax2)
# sales_c.resample('W').sum().plot(ax = ax3)
# sales_d.resample('W').sum().plot(ax = ax4)

# st.pyplot(fig=f)
