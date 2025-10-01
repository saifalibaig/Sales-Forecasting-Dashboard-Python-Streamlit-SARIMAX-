import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from prophet import Prophet
import numpy as np

st.set_page_config(page_title="📈 Sales Forecasting App", layout="wide")
st.title("📈 Sales Forecasting Dashboard")

# -----------------------------
# Load train & test data from zip
# -----------------------------
@st.cache_data
def load_zip_csv(path):
    with zipfile.ZipFile(path, 'r') as z:
        csv_filename = z.namelist()[0]
        df = pd.read_csv(z.open(csv_filename))
    return df

train = load_zip_csv("train.zip")
test = load_zip_csv("test.zip")

train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])
train = train.set_index('Date')
test = test.set_index('Date')

st.subheader("📊 Data Preview")
st.write(train.head())

# -----------------------------
# Model selection
# -----------------------------
model_choice = st.radio(
    "Select Forecasting Model",
    ["Naive Forecast", "Moving Average", "SES", "Holt", "ARIMA", "SARIMA", "Prophet"]
)

# -----------------------------
# Forecast Horizon
# -----------------------------
forecast_steps = st.number_input(
    "Forecast Steps",
    min_value=1,
    max_value=max(len(test), 1),
    value=min(len(test), 30)
)

# -----------------------------
# Run Forecast with preset parameters
# -----------------------------
if st.button("Run Forecast"):
    with st.spinner(f"Running {model_choice}..."):
        if model_choice == "Naive Forecast":
            forecast = [train['Sales'].iloc[-1]] * forecast_steps

        elif model_choice == "Moving Average":
            window = 7  # preset best window
            forecast = [train['Sales'].tail(window).mean()] * forecast_steps

        elif model_choice == "SES":
            ses = SimpleExpSmoothing(train['Sales']).fit(smoothing_level=0.3, optimized=False)
            forecast = ses.forecast(forecast_steps)

        elif model_choice == "Holt":
            holt = Holt(train['Sales']).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
            forecast = holt.forecast(forecast_steps)

        elif model_choice == "ARIMA":
            model = sm.tsa.ARIMA(train['Sales'], order=(2, 1, 2))  # preset parameters
            res = model.fit()
            forecast = res.forecast(steps=forecast_steps)

        elif model_choice == "SARIMA":
            model = sm.tsa.statespace.SARIMAX(
                train['Sales'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)
            forecast = res.forecast(steps=forecast_steps)

        elif model_choice == "Prophet":
            prophet_df = train.reset_index().rename(columns={"date": "ds", "Sales": "y"})
            m = Prophet(daily_seasonality=True)
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=forecast_steps)
            forecast_df = m.predict(future)
            forecast = forecast_df['yhat'][-forecast_steps:].values

    # -----------------------------
    # Results
    # -----------------------------
    forecast_dates = pd.date_range(start=train.index[-1], periods=forecast_steps+1, freq='D')[1:]
    results = pd.DataFrame({
        'date': forecast_dates,
        'forecasted_Sales': forecast
    })

    st.subheader(f"🔮 {model_choice} Forecast")
    st.write(results)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train['Sales'], label="Train")
    ax.plot(results['date'], results['forecasted_Sales'], label="Forecast", color="red")
    ax.legend()
    st.pyplot(fig)
