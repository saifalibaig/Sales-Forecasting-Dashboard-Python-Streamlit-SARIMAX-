# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="📈 Sales Forecasting App", layout="wide")
st.title("📈 Sales Forecasting Dashboard")

# -----------------------------
# Zip file upload
# -----------------------------
train_zip = st.file_uploader("Upload train.zip (contains train.csv)", type="zip")
test_zip = st.file_uploader("Upload test.zip (contains test.csv)", type="zip")

@st.cache_data
def load_zip_csv(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as z:
        csv_filename = z.namelist()[0]
        df = pd.read_csv(z.open(csv_filename))
    return df

if train_zip and test_zip:
    train = load_zip_csv(train_zip)
    test = load_zip_csv(test_zip)

    st.subheader("📊 Raw Data Preview")
    st.write(train.head())

    # -----------------------------
    # Preprocess
    # -----------------------------
    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    train = train.groupby("Date").agg({"Sales": "sum"}).asfreq("D")
    train["Sales"] = train["Sales"].fillna(0)

    test = test.drop_duplicates(subset=["Date"])
    test = test.set_index("Date").asfreq("D")

    horizon = len(test)

    # -----------------------------
    # Evaluation helper
    # -----------------------------
    def evaluate(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        st.write(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        return {"model": name, "mae": mae, "rmse": rmse}

    results = []

    st.subheader("🔹 Forecast Models")

    # -----------------------------
    # Holt-Winters
    # -----------------------------
    st.write("Running Holt-Winters Forecast...")
    hw_model = ExponentialSmoothing(train["Sales"], trend="add", seasonal="add", seasonal_periods=12)
    hw_fit = hw_model.fit()
    hw_forecast = hw_fit.forecast(horizon).round()

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(train.index, train["Sales"], label="Train")
    ax.plot(test.index, hw_forecast, label="HW Forecast", color="orange")
    ax.legend()
    ax.set_title("Holt-Winters Forecast")
    st.pyplot(fig)

    results.append(evaluate(train["Sales"][-horizon:], hw_forecast, "Holt-Winters"))

    # -----------------------------
    # SARIMA
    # -----------------------------
    st.write("Running SARIMA Forecast...")
    sarima_model = SARIMAX(train["Sales"], order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fit.forecast(horizon).round()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(train.index, train["Sales"], label="Train")
    ax.plot(test.index, sarima_forecast, label="SARIMA Forecast", color="green")
    ax.legend()
    ax.set_title("SARIMA Forecast")
    st.pyplot(fig)

    results.append(evaluate(train["Sales"][-horizon:], sarima_forecast, "SARIMA"))

    # -----------------------------
    # Prophet
    # -----------------------------
    st.write("Running Prophet Forecast...")
    prophet_df = train.reset_index()[["Date","Sales"]].rename(columns={"Date":"ds","Sales":"y"})
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    prophet_model.fit(prophet_df)

    future = prophet_model.make_future_dataframe(periods=horizon, freq="D")
    prophet_forecast = prophet_model.predict(future)

    prophet_pred = prophet_forecast.set_index("ds")["yhat"].iloc[-horizon:].round()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(train.index, train["Sales"], label="Train")
    ax.plot(test.index, prophet_pred, label="Prophet Forecast", color="red")
    ax.legend()
    ax.set_title("Prophet Forecast")
    st.pyplot(fig)

    results.append(evaluate(train["Sales"][-horizon:], prophet_pred, "Prophet"))

    # -----------------------------
    # Model Comparison
    # -----------------------------
    results_df = pd.DataFrame(results).sort_values(by="rmse")
    st.subheader("📌 Model Comparison (sorted by RMSE)")
    st.dataframe(results_df)

    best_model = results_df.iloc[0]["model"]
    st.write(f"✅ Best model based on RMSE: **{best_model}**")

    # -----------------------------
    # Final Forecast
    # -----------------------------
    future_horizon = st.number_input("Enter future forecast horizon (days)", min_value=1, max_value=365, value=180)

    if best_model == "Holt-Winters":
        final_forecast = hw_fit.forecast(future_horizon).round()
    elif best_model == "SARIMA":
        final_forecast = sarima_fit.forecast(future_horizon).round()
    else:  # Prophet
        future = prophet_model.make_future_dataframe(periods=future_horizon, freq="D")
        prophet_forecast = prophet_model.predict(future)
        final_forecast = prophet_forecast.set_index("ds")["yhat"].iloc[-future_horizon:].round()

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(train.index, train["Sales"], label="Train")
    ax.plot(pd.date_range(start=train.index[-1]+pd.Timedelta(days=1), periods=future_horizon), final_forecast,
            label=f"{best_model} Forecast", color="purple")
    ax.legend()
    ax.set_title("Final Forecast")
    st.pyplot(fig)

    st.subheader("📄 Final Forecast Data")
    forecast_dates = pd.date_range(start=train.index[-1]+pd.Timedelta(days=1), periods=future_horizon)
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecasted_Sales": final_forecast.astype(int)})
    st.dataframe(forecast_df)
