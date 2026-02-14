import os
import re
from datetime import datetime
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

def standardize_dataframe(frame: pd.DataFrame, rows: int) -> pd.DataFrame | None:
    needed = {"date", "open", "high", "low", "close", "volume"}
    if not needed.issubset(frame.columns):
        return None
    work = frame[["date", "open", "high", "low", "close", "volume"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["date", "close", "volume"]).sort_values("date").tail(rows)
    if len(work) < 120:
        return None
    return work.reset_index(drop=True)

def try_fetch_stooq(symbol: str, rows: int = 1200) -> pd.DataFrame | None:
    symbol_variants = [symbol.lower()]
    if "." not in symbol:
        symbol_variants.append(f"{symbol.lower()}.us")
    for variant in symbol_variants:
        url = f"https://stooq.com/q/d/l/?s={variant}&i=d"
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            df = pd.read_csv(pd.io.common.StringIO(response.text))
        except Exception:
            continue
        if "Date" not in df.columns or "Close" not in df.columns or df.empty:
            continue
        normalized = standardize_dataframe(df.rename(columns=str.lower), rows)
        if normalized is not None:
            return normalized
    return None

def try_fetch_yahoo_chart(symbol: str, rows: int = 1200) -> pd.DataFrame | None:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    params = {"range": "10y", "interval": "1d", "events": "div,splits"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None
    chart = payload.get("chart", {})
    result = chart.get("result")
    if not result:
        return None
    item = result[0]
    timestamps = item.get("timestamp")
    quote = item.get("indicators", {}).get("quote", [])
    if not timestamps or not quote:
        return None
    quote0 = quote[0]
    length = min(len(timestamps), len(quote0.get("open", [])), len(quote0.get("high", [])), len(quote0.get("low", [])), len(quote0.get("close", [])), len(quote0.get("volume", [])))
    if length == 0:
        return None
    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps[:length], unit="s", utc=True).tz_localize(None),
        "open": quote0["open"][:length],
        "high": quote0["high"][:length],
        "low": quote0["low"][:length],
        "close": quote0["close"][:length],
        "volume": quote0["volume"][:length],
    })
    return standardize_dataframe(df, rows)

def fetch_history(symbol: str, rows: int = 1200) -> tuple[pd.DataFrame, str]:
    normalized = normalize_symbol(symbol)
    stooq_data = try_fetch_stooq(normalized, rows)
    if stooq_data is not None:
        return stooq_data, "Stooq"
    yahoo_data = try_fetch_yahoo_chart(normalized, rows)
    if yahoo_data is not None:
        return yahoo_data, "Yahoo Finance"
    raise ValueError("Unable to load data for this ticker right now.")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["return_1d"] = work["close"].pct_change()
    work["return_5d"] = work["close"].pct_change(5)
    work["ma_5"] = work["close"].rolling(5).mean()
    work["ma_10"] = work["close"].rolling(10).mean()
    work["ma_20"] = work["close"].rolling(20).mean()
    work["volatility_10"] = work["return_1d"].rolling(10).std()
    work["target"] = work["close"].shift(-1)
    return work.dropna().reset_index(drop=True)

def train_model(feature_df: pd.DataFrame) -> tuple[RandomForestRegressor, float, list[str]]:
    feature_cols = ["close", "volume", "return_1d", "return_5d", "ma_5", "ma_10", "ma_20", "volatility_10"]
    x = feature_df[feature_cols]
    y = feature_df["target"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=350, random_state=42, min_samples_leaf=2)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    mae = float(mean_absolute_error(y_test, preds))
    return model, mae, feature_cols

def forecast_next_days(history_df: pd.DataFrame, model: RandomForestRegressor, feature_cols: list[str], horizon_days: int) -> tuple[list[float], list[str]]:
    history = history_df.copy()
    prices: list[float] = []
    dates: list[str] = []
    current_date = history["date"].iloc[-1]
    for _ in range(horizon_days):
        feature_frame = build_features(history)
        if feature_frame.empty:
            raise ValueError("Insufficient data to build forecasting features")
        latest = feature_frame.iloc[-1]
        next_close = float(model.predict(latest[feature_cols].to_frame().T)[0])
        current_date = current_date + pd.tseries.offsets.BDay(1)
        synthetic = {
            "date": current_date,
            "open": next_close,
            "high": next_close * 1.005,
            "low": next_close * 0.995,
            "close": next_close,
            "volume": float(history["volume"].tail(20).mean()),
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
    return FileResponse("public/index.html")

@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        raw_df, source = fetch_history(payload.symbol)
        feature_df = build_features(raw_df)
        if len(feature_df) < 80:
            raise ValueError("Not enough clean data after preprocessing")
        model, mae, feature_cols = train_model(feature_df)
        prices, dates = forecast_next_days(raw_df, model, feature_cols, payload.horizon_days)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse(
        symbol=normalize_symbol(payload.symbol),
        data_source=source,
        last_close=round(float(raw_df["close"].iloc[-1]), 2),
        predicted_prices=prices,
        predicted_dates=dates,
        model_mae=round(mae, 4),
    )

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "time": datetime.utcnow().isoformat()}