# S&P 500 Time-Series Forecasting Pipeline

End-to-end ML pipeline forecasting S&P 500 closing prices using 
statistical and machine learning models.

## Results

| Model        | MAPE  | RMSE   |
|--------------|-------|--------|
| ARIMA(1,1,1) | 4.41% | 182.06 |
| LightGBM     | 1.68% | 73.50  |

**LightGBM outperforms ARIMA by 2.6x on MAPE.**

## Methodology
- ADF stationarity test confirmed non-stationarity (p=0.7976)
- First-order differencing applied → stationary (p=0.0000)
- ACF/PACF analysis used to select ARIMA(1,1,1) parameters
- LightGBM trained on lag features and rolling statistics

## How to run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## API Usageye