from fastapi import FastAPI
import pandas as pd
import numpy as np
import xgboost as xgb

app = FastAPI(title="S&P 500 Forecasting API")

def train_model():
    df = pd.read_csv('data/raw/yahoo_stock.csv', parse_dates=['Date'])
    df = df.sort_values('Date').set_index('Date')
    df['lag_1']  = df['Close'].shift(1)
    df['lag_2']  = df['Close'].shift(2)
    df['lag_5']  = df['Close'].shift(5)
    df['lag_10'] = df['Close'].shift(10)
    df['rolling_mean_5']  = df['Close'].rolling(5).mean()
    df['rolling_mean_10'] = df['Close'].rolling(10).mean()
    df['rolling_std_5']   = df['Close'].rolling(5).std()
    df['month']     = df.index.month
    df['dayofweek'] = df.index.dayofweek
    df = df.dropna()
    features = ['lag_1','lag_2','lag_5','lag_10',
                'rolling_mean_5','rolling_mean_10','rolling_std_5',
                'month','dayofweek']
    X = df[features][:-60]
    y = df['Close'][:-60]
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05)
    model.fit(X, y)
    return model

model = train_model()

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
    prediction = model.predict(np.array(features))[0]
    return {
        "predicted_close_price": round(float(prediction), 2),
        "currency": "USD",
        "model": "XGBoost"
    }