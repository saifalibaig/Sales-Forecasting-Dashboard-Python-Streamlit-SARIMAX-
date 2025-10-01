import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from prophet import Prophet
import numpy as np

st.set_page_config(page_title="📈 Sales Forecasting App", layout="wide")
st.title("📈 Sales Forecasting Dashboard (Multiple Models)")

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

train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])
train = train.set_index('date')
test = test.set_index('date')

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
# Forecast Horizon (Fixed Issue)
# -----------------------------
forecast_steps = st.number_input(
    "Forecast Steps",
    min_value=1,
    max_value=max(len(test), 1),
    value=min(len(test), 30)
)

# -----------------------------
# Run Forecast
# -----------------------------
if st.button("Run Forecast"):
    with st.spinner(f"Running {model_choice}..."):
        if model_choice == "Naive Forecast":
            forecast = [train['Sales'].iloc[-1]] * forecast_steps

        elif model_choice == "Moving Average":
            window = st.slider("Moving Average Window", 2, 30, 7)
            forecast = [train['Sales'].tail(window).mean()] * forecast_steps

        elif model_choice == "SES":
            alpha = st.slider("Smoothing level (alpha)", 0.01, 1.0, 0.3)
            ses = SimpleExpSmoothing(train['Sales']).fit(smoothing_level=alpha, optimized=False)
            forecast = ses.forecast(forecast_steps)

        elif model_choice == "Holt":
            holt = Holt(train['Sales']).fit()
            forecast = holt.forecast(forecast_steps)

        elif model_choice == "ARIMA":
            p = st.slider("p", 0, 3, 1)
            d = st.slider("d", 0, 2, 1)
            q = st.slider("q", 0, 3, 1)
            model = sm.tsa.ARIMA(train['Sales'], order=(p, d, q))
            res = model.fit()
            forecast = res.forecast(steps=forecast_steps)

        elif model_choice == "SARIMA":
            p = st.slider("p", 0, 2, 1)
            d = st.slider("d", 0, 1, 1)
            q = st.slider("q", 0, 2, 1)
            P = st.slider("P", 0, 2, 1)
            D = st.slider("D", 0, 1, 1)
            Q = st.slider("Q", 0, 2, 1)
            seasonal_period = st.number_input("Seasonal Period", value=12)
            model = sm.tsa.statespace.SARIMAX(
                train['Sales'],
                order=(p, d, q),
                seasonal_order=(P, D, Q, seasonal_period),
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
