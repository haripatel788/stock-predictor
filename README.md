# Stock Predictor

A complete backend + frontend stock prediction app using Python, FastAPI, pandas, and scikit-learn.

## Features

- Free market data from Stooq's public CSV endpoint
- ML forecasting pipeline built with scikit-learn `RandomForestRegressor`
- REST endpoint for predictions
- Browser UI for symbol input, horizon selection, and forecast table
- Built-in validation and health endpoint

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## API

`POST /api/predict`

Request body:

```json
{
  "symbol": "AAPL",
  "horizon_days": 7
}
```

Response body:

```json
{
  "symbol": "AAPL",
  "data_source": "Stooq",
  "last_close": 235.1,
  "predicted_prices": [236.2, 236.7],
  "predicted_dates": ["2026-02-13", "2026-02-16"],
  "model_mae": 2.0831,
  "model_mae_pct": 0.89,
  "baseline_mae": 2.4512
}
```

`model_mae` is the one-day holdout MAE (in dollars).  
`model_mae_pct` is one-day holdout MAE in percent.  
`baseline_mae` is the one-day MAE from a naive baseline (`tomorrow = today`).
