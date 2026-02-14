import os
from datetime import datetime, timedelta
import re
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

class PredictionRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=24)
    horizon_days: int = Field(default=7, ge=1, le=30)

class PredictionResponse(BaseModel):
    symbol: str
    data_source: str
    last_close: float
    predicted_prices: list[float]
    predicted_dates: list[str]
    model_mae: float

def normalize_symbol(raw_symbol: str) -> str:
    clean = raw_symbol.strip().upper()
    if not clean:
        raise ValueError("Ticker symbol is required")
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,24}", clean):
        raise ValueError("Ticker contains unsupported characters")
    return clean

def try_fetch_yahoo_chart(symbol: str, rows: int = 1200) -> pd.DataFrame | None:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    params = {"range": "10y", "interval": "1d"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        result = data["chart"]["result"][0]
        indicators = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "date": pd.to_datetime(result["timestamp"], unit="s"),
            "open": indicators["open"],
            "high": indicators["high"],
            "low": indicators["low"],
            "close": indicators["close"],
            "volume": indicators["volume"]
        })
        return df.tail(rows)
    except Exception:
        return None

def fetch_history(symbol: str) -> tuple[pd.DataFrame, str]:
    df = try_fetch_yahoo_chart(symbol)
    if df is not None and len(df) > 50:
        return df, "Yahoo Finance"
    
    stooq_url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        df = pd.read_csv(stooq_url)
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            return df, "Stooq"
    except Exception:
        pass
    
    raise HTTPException(status_code=404, detail=f"Could not fetch data for {symbol}")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date")
    df["ma_20"] = df["close"].rolling(window=20).mean()
    df["ma_50"] = df["close"].rolling(window=50).mean()
    df["daily_ret"] = df["close"].pct_change()
    df["target"] = df["close"].shift(-1)
    return df.dropna()

def train_model(df: pd.DataFrame):
    features = ["open", "high", "low", "close", "volume", "ma_20", "ma_50", "daily_ret"]
    X = df[features]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    return model, mae, features

def forecast_next_days(history: pd.DataFrame, model, feature_cols, days: int):
    prices = []
    dates = []
    current_date = history["date"].iloc[-1]
    
    for _ in range(days):
        current_date += timedelta(days=1)
        if current_date.weekday() >= 5:
            current_date += timedelta(days=7 - current_date.weekday())
            
        last_row = build_features(history).iloc[-1:]
        X_next = last_row[feature_cols]
        next_close = model.predict(X_next)[0]
        
        synthetic = {
            "date": current_date,
            "open": last_row["close"].values[0],
            "high": next_close * 1.01,
            "low": next_close * 0.99,
            "close": next_close,
            "volume": last_row["volume"].values[0]
        }
        history = pd.concat([history, pd.DataFrame([synthetic])], ignore_index=True)
        prices.append(round(next_close, 2))
        dates.append(current_date.strftime("%Y-%m-%d"))
    return prices, dates

app = FastAPI(title="Stock Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root() -> FileResponse:
    path = os.path.join(os.getcwd(), "static", "index.html")
    return FileResponse(path)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        raw_df, source = fetch_history(payload.symbol)
        feature_df = build_features(raw_df)
        if len(feature_df) < 80:
            raise ValueError("Not enough clean data after preprocessing")
        model, mae, feature_cols = train_model(feature_df)
        prices, dates = forecast_next_days(raw_df, model, feature_cols, payload.horizon_days)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    
    return PredictionResponse(
        symbol=normalize_symbol(payload.symbol),
        data_source=source,
        last_close=round(float(raw_df["close"].iloc[-1]), 2),
        predicted_prices=prices,
        predicted_dates=dates,
        model_mae=mae
    )