# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import itertools
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="📈 Auto-Tuned Forecasting", layout="wide")
st.title("📈 Advanced Auto-Tuned Sales Forecasting Dashboard")

train_zip = st.file_uploader("Upload train.zip (contains train.csv)", type="zip")
test_zip = st.file_uploader("Upload test.zip (contains test.csv)", type="zip")

@st.cache_data
def load_zip_csv(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as z:
        csv_filename = z.namelist()[0]
        df = pd.read_csv(z.open(csv_filename))
    return df

def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"model": name, "mae": mae, "rmse": rmse}

if train_zip and test_zip:
    train = load_zip_csv(train_zip)
    test = load_zip_csv(test_zip)

    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    train = train.groupby("Date").agg({"Sales": "sum"}).asfreq("D")
    train["Sales"] = train["Sales"].fillna(0)

    test = test.drop_duplicates(subset=["Date"])
    test = test.set_index("Date").asfreq("D")

    horizon = len(test)
    results = {}

    st.subheader("🔹 Holt-Winters Auto-Tuning")
    best_mae_hw = float("inf")
    for trend in ["add", "mul"]:
        for seasonal in ["add", "mul"]:
            for period in [7, 12, 30]:
                try:
                    model = ExponentialSmoothing(train["Sales"], trend=trend, seasonal=seasonal, seasonal_periods=period).fit()
                    forecast = model.forecast(horizon).round()
                    mae = mean_absolute_error(train["Sales"][-horizon:], forecast)
                    if mae < best_mae_hw:
                        best_mae_hw = mae
                        hw_model = model
                        hw_forecast = forecast
                except:
                    continue
    results["Holt-Winters"] = evaluate(train["Sales"][-horizon:], hw_forecast, "Holt-Winters")
    fig, ax = plt.subplots()
    ax.plot(train.index, train["Sales"], label="Train")
    ax.plot(test.index, hw_forecast, label="HW Forecast", color="orange")
    ax.legend(); st.pyplot(fig)

    st.subheader("🔹 SARIMA Auto-Tuning")
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in pdq for s in [7, 12, 30]]
    best_aic = float("inf")
    for param in pdq:
        for seasonal_order in seasonal_pdq:
            try:
                mod = SARIMAX(train["Sales"], order=param, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False)
                forecast = res.forecast(horizon).round()
                if res.aic < best_aic:
                    best_aic = res.aic
                    sarima_model = res
                    sarima_forecast = forecast
            except:
                continue
    results["SARIMA"] = evaluate(train["Sales"][-horizon:], sarima_forecast, "SARIMA")
    fig, ax = plt.subplots()
    ax.plot(train.index, train["Sales"], label="Train")
    ax.plot(test.index, sarima_forecast, label="SARIMA Forecast", color="green")
    ax.legend(); st.pyplot(fig)

    st.subheader("🔹 Prophet Auto-Tuning")
    prophet_df = train.reset_index()[["Date", "Sales"]].rename(columns={"Date": "ds", "Sales": "y"})
    best_mae_prophet = float("inf")
    for cps in [0.01, 0.1, 0.5]:
        for mode in ["additive", "multiplicative"]:
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                            seasonality_mode=mode, changepoint_prior_scale=cps)
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=horizon, freq="D")
            forecast = model.predict(future).set_index("ds")["yhat"].iloc[-horizon:].round()
            mae = mean_absolute_error(train["Sales"][-horizon:], forecast)
            if mae < best_mae_prophet:
                best_mae_prophet = mae
                prophet_model = model
                prophet_forecast = forecast
    results["Prophet"] = evaluate(train["Sales"][-horizon:], prophet_forecast, "Prophet")
    fig, ax = plt.subplots()
    ax.plot(train.index, train["Sales"], label="Train")
    ax.plot(test.index, prophet_forecast, label="Prophet Forecast", color="red")
    ax.legend(); st.pyplot(fig)

    st.subheader("📊 Model Comparison")
    results_df = pd.DataFrame(results.values()).sort_values(by="rmse")
    st.dataframe(results_df)
    best_model_name = results_df.iloc[0]["model"]

    st.write(f"✅ Best model based on RMSE: **{best_model_name}**")

    st.subheader("🔹 Ensemble Forecast (Weighted by RMSE)")
    inv_rmse = {k: 1/v["rmse"] for k,v in results.items()}
    total_inv_rmse = sum(inv_rmse.values())
    weights = {k: v/total_inv_rmse for k,v in inv_rmse.items()}
    ensemble_forecast = (hw_forecast*weights["Holt-Winters"] + sarima_forecast*weights["SARIMA"] + prophet_forecast*weights["Prophet"]).round()

    fig, ax = plt.subplots()
    ax.plot(train.index, train["Sales"], label="Train")
    ax.plot(test.index, ensemble_forecast, label="Ensemble Forecast", color="purple")
    ax.legend(); st.pyplot(fig)

    st.subheader("📄 Final Forecast Data")
    forecast_dates = pd.date_range(start=train.index[-1]+pd.Timedelta(days=1), periods=horizon)
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecasted_Sales": ensemble_forecast.astype(int)})
    st.dataframe(forecast_df)
