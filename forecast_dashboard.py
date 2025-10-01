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
import itertools

st.set_page_config(page_title="📈 Auto-Tuned Sales Forecasting", layout="wide")
st.title("📈 Auto-Tuned Sales Forecasting Dashboard")

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

    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    train = train.groupby("Date").agg({"Sales": "sum"}).asfreq("D")
    train["Sales"] = train["Sales"].fillna(0)

    test = test.drop_duplicates(subset=["Date"])
    test = test.set_index("Date").asfreq("D")

    horizon = len(test)

    def evaluate(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        st.write(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        return {"model": name, "mae": mae, "rmse": rmse}

    results = []
    st.subheader("🔹 Forecast Models with Auto-Tuning")

    # Holt-Winters Auto-Tuning
    st.write("🔹 Running Holt-Winters Auto-Tuning...")
    best_mae = float("inf")
    best_hw_model = None
    hw_options = list(itertools.product(["add", "mul"], ["add", "mul"], [7, 12]))
    for trend, seasonal, period in hw_options:
        try:
            model = ExponentialSmoothing(train["Sales"], trend=trend, seasonal=seasonal, seasonal_periods=period).fit()
            pred = model.forecast(horizon).round()
            mae = mean_absolute_error(train["Sales"][-horizon:], pred)
            if mae < best_mae:
                best_mae = mae
                best_hw_model = model
                best_hw_forecast = pred
        except:
            continue
    hw_forecast = best_hw_forecast

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(train.index, train["Sales"], label="Train")
    ax.plot(test.index, hw_forecast, label="HW Forecast", color="orange")
    ax.legend()
    ax.set_title("Holt-Winters Forecast")
    st.pyplot(fig)
    results.append(evaluate(train["Sales"][-horizon:], hw_forecast, "Holt-Winters"))

    # SARIMA Auto-Tuning
    st.write("🔹 Running SARIMA Auto-Tuning...")
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
    best_rmse = float("inf")
    best_sarima_model = None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(train["Sales"], order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                pred = model.forecast(horizon).round()
                rmse = np.sqrt(mean_squared_error(train["Sales"][-horizon:], pred))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_sarima_model = model
                    best_sarima_forecast = pred
            except:
                continue
    sarima_forecast = best_sarima_forecast

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(train.index, train["Sales"], label="Train")
    ax.plot(test.index, sarima_forecast, label="SARIMA Forecast", color="green")
    ax.legend()
    ax.set_title("SARIMA Forecast")
    st.pyplot(fig)
    results.append(evaluate(train["Sales"][-horizon:], sarima_forecast, "SARIMA"))

    # Prophet Auto-Tuning
    st.write("🔹 Running Prophet Auto-Tuning...")
    prophet_df = train.reset_index()[["Date","Sales"]].rename(columns={"Date":"ds","Sales":"y"})
    best_mae = float("inf")
    best_prophet_model = None
    best_prophet_forecast = None
    for cps in [0.01, 0.1, 0.5]:
        for seasonality_mode in ["additive", "multiplicative"]:
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                            seasonality_mode=seasonality_mode, changepoint_prior_scale=cps)
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=horizon, freq="D")
            forecast = model.predict(future).set_index("ds")["yhat"].iloc[-horizon:].round()
            mae = mean_absolute_error(train["Sales"][-horizon:], forecast)
            if mae < best_mae:
                best_mae = mae
                best_prophet_model = model
                best_prophet_forecast = forecast
    prophet_pred = best_prophet_forecast

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(train.index, train["Sales"], label="Train")
    ax.plot(test.index, prophet_pred, label="Prophet Forecast", color="red")
    ax.legend()
    ax.set_title("Prophet Forecast")
    st.pyplot(fig)
    results.append(evaluate(train["Sales"][-horizon:], prophet_pred, "Prophet"))

    # Model Comparison
    results_df = pd.DataFrame(results).sort_values(by="rmse")
    st.subheader("📌 Model Comparison (sorted by RMSE)")
    st.dataframe(results_df)

    best_model = results_df.iloc[0]["model"]
    st.write(f"✅ Best model based on RMSE: **{best_model}**")

    future_horizon = st.number_input("Enter future forecast horizon (days)", min_value=1, max_value=365, value=180)

    if best_model == "Holt-Winters":
        final_forecast = best_hw_model.forecast(future_horizon).round()
    elif best_model == "SARIMA":
        final_forecast = best_sarima_model.forecast(future_horizon).round()
    else:  # Prophet
        future = best_prophet_model.make_future_dataframe(periods=future_horizon, freq="D")
        prophet_forecast = best_prophet_model.predict(future)
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
