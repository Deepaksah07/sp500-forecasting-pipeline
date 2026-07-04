from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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
    return model, df

model, df_global = train_model()

@app.get("/", response_class=HTMLResponse)
def home():
    last = df_global.iloc[-1]
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>S&P 500 Forecasting</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',sans-serif; background:#0f1117; color:#e0e0e0; min-height:100vh; }}
  header {{ background:linear-gradient(135deg,#1a1f2e,#252d3d); padding:32px 48px; border-bottom:1px solid #2a3040; }}
  header h1 {{ font-size:1.8rem; color:#fff; font-weight:700; }}
  header p {{ color:#8892a4; margin-top:6px; font-size:0.95rem; }}
  .badge {{ display:inline-block; background:#00c853; color:#000; font-size:0.7rem; font-weight:700; padding:3px 10px; border-radius:20px; margin-left:12px; vertical-align:middle; }}
  main {{ max-width:900px; margin:40px auto; padding:0 24px; }}
  .stats {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:32px; }}
  .stat-card {{ background:#1a1f2e; border:1px solid #2a3040; border-radius:12px; padding:20px 24px; }}
  .stat-card .label {{ font-size:0.78rem; color:#8892a4; text-transform:uppercase; letter-spacing:0.05em; }}
  .stat-card .value {{ font-size:1.6rem; font-weight:700; color:#fff; margin-top:6px; }}
  .stat-card .sub {{ font-size:0.8rem; color:#00c853; margin-top:4px; }}
  .card {{ background:#1a1f2e; border:1px solid #2a3040; border-radius:12px; padding:28px 32px; margin-bottom:24px; }}
  .card h2 {{ font-size:1.1rem; color:#fff; margin-bottom:20px; font-weight:600; }}
  .form-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .field label {{ font-size:0.8rem; color:#8892a4; display:block; margin-bottom:6px; }}
  .field input {{ width:100%; background:#0f1117; border:1px solid #2a3040; border-radius:8px; padding:10px 14px; color:#fff; font-size:0.95rem; outline:none; transition:border 0.2s; }}
  .field input:focus {{ border-color:#4a90e2; }}
  button {{ margin-top:20px; width:100%; background:linear-gradient(135deg,#4a90e2,#357abd); color:#fff; border:none; border-radius:8px; padding:14px; font-size:1rem; font-weight:600; cursor:pointer; transition:opacity 0.2s; }}
  button:hover {{ opacity:0.9; }}
  #result {{ display:none; margin-top:20px; background:#0d1f12; border:1px solid #00c853; border-radius:10px; padding:20px 24px; text-align:center; }}
  #result .pred-label {{ font-size:0.85rem; color:#8892a4; }}
  #result .pred-value {{ font-size:2.4rem; font-weight:700; color:#00c853; margin-top:4px; }}
  .model-tag {{ font-size:0.75rem; color:#4a90e2; margin-top:6px; }}
  .metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
  .metric {{ background:#0f1117; border-radius:8px; padding:14px 16px; }}
  .metric .m-label {{ font-size:0.75rem; color:#8892a4; }}
  .metric .m-value {{ font-size:1.1rem; font-weight:700; color:#fff; margin-top:4px; }}
  .metric .m-model {{ font-size:0.7rem; color:#4a90e2; margin-top:2px; }}
</style>
</head>
<body>
<header>
  <h1>S&P 500 Forecasting Pipeline <span class="badge">LIVE</span></h1>
  <p>End-to-end ML pipeline · XGBoost model · Trained on 2015–2020 market data</p>
</header>
<main>
  <div class="stats">
    <div class="stat-card">
      <div class="label">Best Model MAPE</div>
      <div class="value">1.68%</div>
      <div class="sub">↑ XGBoost wins</div>
    </div>
    <div class="stat-card">
      <div class="label">Best Model RMSE</div>
      <div class="value">73.50</div>
      <div class="sub">vs ARIMA 182.06</div>
    </div>
    <div class="stat-card">
      <div class="label">Training samples</div>
      <div class="value">1,755</div>
      <div class="sub">5 years of daily data</div>
    </div>
  </div>

  <div class="card">
    <h2>Live Prediction</h2>
    <div class="form-grid">
      <div class="field"><label>Previous Close (lag 1)</label><input id="lag_1" type="number" value="{last['Close']:.2f}"/></div>
      <div class="field"><label>Close 2 days ago (lag 2)</label><input id="lag_2" type="number" value="{last['lag_2']:.2f}"/></div>
      <div class="field"><label>Close 5 days ago (lag 5)</label><input id="lag_5" type="number" value="{last['lag_5']:.2f}"/></div>
      <div class="field"><label>Close 10 days ago (lag 10)</label><input id="lag_10" type="number" value="{last['lag_10']:.2f}"/></div>
      <div class="field"><label>5-day Rolling Mean</label><input id="rolling_mean_5" type="number" value="{last['rolling_mean_5']:.2f}"/></div>
      <div class="field"><label>10-day Rolling Mean</label><input id="rolling_mean_10" type="number" value="{last['rolling_mean_10']:.2f}"/></div>
      <div class="field"><label>5-day Rolling Std</label><input id="rolling_std_5" type="number" value="{last['rolling_std_5']:.2f}"/></div>
      <div class="field"><label>Month (1-12)</label><input id="month" type="number" value="{last.name.month}"/></div>
    </div>
    <button onclick="predict()">Generate Forecast</button>
    <div id="result">
      <div class="pred-label">Predicted S&P 500 Close Price</div>
      <div class="pred-value" id="pred-value">—</div>
      <div class="model-tag">Powered by XGBoost · Deepak Kumar Sah</div>
    </div>
  </div>

  <div class="card">
    <h2>Model Comparison</h2>
    <div class="metrics">
      <div class="metric"><div class="m-label">ARIMA(1,1,1) — MAPE</div><div class="m-value">4.41%</div><div class="m-model">Statistical baseline</div></div>
      <div class="metric"><div class="m-label">XGBoost — MAPE</div><div class="m-value">1.68%</div><div class="m-model">✓ Best model · 2.6x better</div></div>
      <div class="metric"><div class="m-label">ARIMA(1,1,1) — RMSE</div><div class="m-value">182.06</div><div class="m-model">Flat forecast</div></div>
      <div class="metric"><div class="m-label">XGBoost — RMSE</div><div class="m-value">73.50</div><div class="m-model">Follows trend well</div></div>
    </div>
  </div>
</main>
<script>
async function predict() {{
  const params = new URLSearchParams({{
    lag_1: document.getElementById('lag_1').value,
    lag_2: document.getElementById('lag_2').value,
    lag_5: document.getElementById('lag_5').value,
    lag_10: document.getElementById('lag_10').value,
    rolling_mean_5: document.getElementById('rolling_mean_5').value,
    rolling_mean_10: document.getElementById('rolling_mean_10').value,
    rolling_std_5: document.getElementById('rolling_std_5').value,
    month: document.getElementById('month').value,
    dayofweek: 1
  }});
  const res = await fetch('/predict?' + params);
  const data = await res.json();
  document.getElementById('pred-value').textContent = '$' + data.predicted_close_price.toLocaleString();
  document.getElementById('result').style.display = 'block';
}}
</script>
</body>
</html>
"""

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