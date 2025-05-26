import streamlit as st

# Importing required libraries
import numpy as np
import pandas as pd, datetime
import seaborn as sns
# from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import plotly.express as px
# get_ipython().run_line_magic('matplotlib', 'inline')
from time import time
import os
from math import sqrt
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# import itertools
# import statsmodels.api as sm
# from statsmodels.tsa.stattools import acf,pacf
# from statsmodels.tsa.arima_model import  ARIMA
# from sklearn import model_selection
# from sklearn.metrics import mean_squared_error, r2_score
# from pandas import DataFrame
# import xgboost as xgb
# # from fbprophet import Prophet
# import warnings
# warnings.filterwarnings('ignore')


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



# Checking whether there was a school holiday when the store was closed
st.sidebar.write("Checking whether there was a school holiday when the store was closed")
tsc_schoolHoliday = train_store_closed['SchoolHoliday'].value_counts() #.plot(kind='bar')
st.sidebar.bar_chart(tsc_schoolHoliday )


# Checking whether there was a state holiday when the store was closed
st.sidebar.write("Checking whether there was a state holiday when the store was closed")
tsc_stateHoliday = train_store_closed['StateHoliday'].value_counts() #.plot(kind='bar')
st.sidebar.bar_chart(tsc_stateHoliday )


# No. of days with closed stores
train[(train.Open == 0)].shape[0]

# No. of days when store was opened but zero sales - might be because of external factors or refurbishmnent
train[(train.Open == 1) & (train.Sales == 0)].shape[0]

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


# Closed and zero-sales obseravtions
train_store_joined[(train_store_joined.Open ==0) | (train_store_joined.Sales==0)].shape

# Open & Sales >0 stores
train_store_joined_open = train_store_joined[~((train_store_joined.Open ==0) | (train_store_joined.Sales==0))]



# Only use numeric columns for correlation
numeric_data = train_store_joined.select_dtypes(include='number')
# Plotly heatmap
fig = px.imshow(
    numeric_data.corr(),
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Correlation Heatmap of Numeric Features",
)

# Show in Streamlit
st.plotly_chart(fig, use_container_width=True)




# # Sales trend over the months
# st.write("Sales trend over the months")
# sns.catplot(
#     data=train_store_joined_open,
#     x="Month",
#     y="Sales",
#     col="Promo",     # Facet by 'Promo' horizontally
#     hue="Promo2",    # Differentiate by 'Promo2'
#     row="Year",      # Facet by 'Year' vertically
#     kind="point"       # You can change to "strip", "violin", etc.
# )

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
