# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import zipfile

st.set_page_config(page_title="📈 Sales Forecasting App (SARIMAX)", layout="wide")
st.title("📈 Sales Forecasting App (SARIMAX)")

# -----------------------------
# Load train & test from local zip files
# -----------------------------
@st.cache_data
def load_zip_csv(path):
    with zipfile.ZipFile(path, 'r') as z:
        csv_filename = z.namelist()[0]  # assumes 1 CSV per zip
        df = pd.read_csv(z.open(csv_filename))
    return df

train = load_zip_csv("train.zip")
test = load_zip_csv("test.zip")

# Convert to datetime
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])
train = train.set_index('Date')
test = test.set_index('Date')

# Fill missing values
train['Sales'] = train['Sales'].fillna(method='ffill').fillna(0)

st.subheader("📊 Data Preview")
st.write("**Train Data:**", train.head())
st.write("**Test Data:**", test.head())

# -----------------------------
# Model parameter controls
# -----------------------------
st.sidebar.header("SARIMAX Parameters")
p = st.sidebar.slider("p", 0, 3, 1)
d = st.sidebar.slider("d", 0, 2, 1)
q = st.sidebar.slider("q", 0, 3, 1)
P = st.sidebar.slider("P", 0, 3, 1)
D = st.sidebar.slider("D", 0, 2, 1)
Q = st.sidebar.slider("Q", 0, 3, 1)
seasonal_period = st.sidebar.number_input("Seasonal period", value=12)

# -----------------------------
# Run Forecast
# -----------------------------
if st.button("Run Forecast"):
    with st.spinner("Training SARIMAX model..."):
        model = sm.tsa.statespace.SARIMAX(
            train['Sales'],
            order=(p, d, q),
            seasonal_order=(P, D, Q, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    forecast = model.get_forecast(steps=len(test))
    fc_mean = forecast.predicted_mean
    fc_ci = forecast.conf_int()

    results = pd.DataFrame({
        'date': test.index,
        'forecasted_Sales': fc_mean.values,
        'lower_ci': fc_ci.iloc[:, 0].values,
        'upper_ci': fc_ci.iloc[:, 1].values
    })

    st.subheader("🔮 Forecasted Sales")
    st.write(results)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train['Sales'], label="Train")
    ax.plot(results['date'], results['forecasted_Sales'], label="Forecast", color="red")
    ax.fill_between(results['date'], results['lower_ci'], results['upper_ci'],
                    color="pink", alpha=0.3)
    ax.legend()
    st.pyplot(fig)
