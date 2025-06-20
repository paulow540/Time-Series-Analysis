# 📈 Sales Forecasting Time Series App
This Streamlit web app allows you to analysis a sales dataset and perform time series forecasting using the SARIMA model. It forecasts future sales based on historical sales, holidays, and promotions.

## 🔍 Features
- 📤 Import  your custom sales dataset (CSV)

- 🏪 Select any store by store_id

- 📊 View historical sales trends

- ⏳ Forecast future sales using SARIMA

- 💡 Uses is_holiday and promotion as exogenous factors

- 📥 Download forecast results as CSV

## 🧾 Dataset Requirements
Make sure your CSV file contains the following columns:

### Column	Type	Description
### 📊 Dataset Columns

| Column Name   | Type     | Description                              |
|---------------|----------|------------------------------------------|
| store_id      | Integer  | Unique identifier for each store         |
| sales         | Integer  | Daily sales values (target variable)     |
| date          | Date     | Date in YYYY-MM-DD format                |
| day_of_week   | String   | Name of the day (e.g., Monday)           |
| is_holiday    | 0 or 1   | 1 if the day is a holiday, 0 otherwise   |
| promotion     | 0 or 1   | 1 if promotion is active, 0 otherwise    |

## 🚀 How to Use
1. Import your CSV file.

2. Choose the store ID from the sidebar.

3. The app will:

    - Show a sales trend graph

    - Train a SARIMA model

     - Forecast sales for the future period

4. View and download your forecasted results.

# 💻 How to Run Locally
1. Clone this repo or copy the code into a file called app.py

2. Install dependencies:

    ```bash
    pip install streamlit pandas numpy matplotlib scikit-learn statsmodels
    
3. Run the app:
   
   ```bash
    streamlit run app.py
  
## 📦 Sample requirements.txt

    streamlit
    pandas
    numpy
    matplotlib
    scikit-learn
    statsmodels
    
## 🧠 Author
Developed with ❤️ using Python and Streamlit.



   




