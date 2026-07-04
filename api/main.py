from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="S&P 500 Forecasting API")

model = joblib.load("models/lgbm_forecast.pkl")

@app.get("/")
def home():
    return {"message": "S&P 500 Forecasting API is running"}

@app.get("/predict")
def predict(
    lag_1: float, lag_2: float, lag_5: float, lag_10: float,
    rolling_mean_5: float, rolling_mean_10: float, rolling_std_5: float,
    month: int, dayofweek: int
):
    features = [[lag_1, lag_2, lag_5, lag_10,
                 rolling_mean_5, rolling_mean_10, rolling_std_5,
                 month, dayofweek]]
    prediction = model.predict(features)[0]
    return {
        "predicted_close_price": round(float(prediction), 2),
        "currency": "USD",
        "model": "LightGBM"
    }