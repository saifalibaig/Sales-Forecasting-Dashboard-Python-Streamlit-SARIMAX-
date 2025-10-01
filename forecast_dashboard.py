# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import statsmodels.api as sm

st.title("📈 Sales Forecasting App (SARIMAX)")

# Upload data
train_file = st.file_uploader("Upload training CSV (with 'date' and 'Sales')", type="csv")
test_file = st.file_uploader("Upload test CSV (with 'date' only)", type="csv")

if train_file is not None and test_file is not None:
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # Convert to datetime
    train['date'] = pd.to_datetime(train['date'])
    test['date'] = pd.to_datetime(test['date'])
    train = train.set_index('date')
    test = test.set_index('date')

    # Fill missing
    train['Sales'] = train['Sales'].fillna(method='ffill').fillna(0)

    # Model params (basic UI controls)
    p = st.slider("p", 0, 3, 1)
    d = st.slider("d", 0, 2, 1)
    q = st.slider("q", 0, 3, 1)
    P = st.slider("P", 0, 3, 1)
    D = st.slider("D", 0, 2, 1)
    Q = st.slider("Q", 0, 3, 1)
    seasonal_period = st.number_input("Seasonal period", value=12)

    if st.button("Run Forecast"):
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
            'lower_ci': fc_ci.iloc[:,0].values,
            'upper_ci': fc_ci.iloc[:,1].values
        })

        st.subheader("Forecasted Sales")
        st.write(results)

        # Plot
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(train.index, train['Sales'], label="Train")
        ax.plot(results['date'], results['forecasted_Sales'], label="Forecast", color="red")
        ax.fill_between(results['date'], results['lower_ci'], results['upper_ci'],
                        color="pink", alpha=0.3)
        ax.legend()
        st.pyplot(fig)
