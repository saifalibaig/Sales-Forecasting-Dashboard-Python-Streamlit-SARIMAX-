# 📈 Sales Forecasting Dashboard

An interactive **Streamlit-based application** for forecasting sales using multiple time series models.  
The dashboard enables users to upload their sales data, visualize historical trends, compare forecasting models, and generate future sales predictions.  

🔗 **Live App**: [Sales Forecasting Dashboard](https://sarimax-forecast-dashboard.streamlit.app/)

---

## ✨ Key Highlights

- Upload custom sales datasets (`train.csv`, `test.csv` inside ZIP)  
- Automated preprocessing & handling of missing values  
- Three forecasting models implemented: **Holt-Winters, SARIMA, Prophet**  
- Side-by-side model performance evaluation with **MAE & RMSE**  
- Interactive forecast plots & future predictions (user-defined horizon)  
- Ranked model selection based on lowest RMSE  
- Final forecast displayed in **visual and tabular form**  

---

## 🚀 Features

- 📂 **Data Upload**: Supports ZIP files containing `train.csv` and `test.csv`.
- 🔎 **Data Preprocessing**: Handles missing values, sets frequency to daily, and prepares time series.
- 📊 **Model Implementations**:
  - Holt-Winters Exponential Smoothing
  - SARIMA (Seasonal ARIMA)
  - Prophet (Facebook)
- 📈 **Visualizations**:
  - Forecast plots for each model
  - Comparison of predicted vs actual sales
  - RMSE and MAE evaluation metrics
- 🏆 **Model Comparison**: Ranks models based on RMSE.
- 🔮 **Final Forecast**: Extend predictions into the future (user-defined horizon).
- 📄 **Downloadable Forecast Table**.

---

## 🎯 Use Cases

- Demand forecasting for retail & e-commerce businesses  
- Inventory planning & optimization  
- Financial forecasting & revenue projections  
- Strategy development for marketing & sales campaigns  

---

## 📈 Future Enhancements

- Add advanced hyperparameter tuning for SARIMA & Prophet  
- Include machine learning regressors (XGBoost, LightGBM, RNN/LSTM)  
- Enable multi-store or product-level forecasting  
- Export forecasts directly as CSV or Excel  
- Add real-time API integration for continuous forecasting  

---

## 🛠 Tech Stack

- **Python**: Pandas, NumPy, Matplotlib, scikit-learn  
- **Time Series Models**: SARIMA, Holt-Winters, Prophet  
- **Streamlit**: Interactive web dashboard  
- **Deployment**: Streamlit Cloud  

---
