import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay
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
    model_mae_pct: float
    baseline_mae: float

FEATURE_COLS = [
    "close",
    "volume",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "ma_5",
    "ma_10",
    "ma_20",
    "ma_50",
    "price_vs_ma_10",
    "price_vs_ma_20",
    "price_vs_ma_50",
    "volatility_10",
    "volatility_20",
    "close_lag_1",
    "close_lag_2",
    "close_lag_3",
    "close_lag_5",
    "close_lag_10",
    "volume_lag_1",
    "volume_lag_5",
    "volume_lag_10",
    "volume_change_1d",
    "intraday_range",
    "high_close_ratio",
    "low_close_ratio",
]


class NyseHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday, start_date="2022-01-01"),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


NYSE_BDAY = CustomBusinessDay(calendar=NyseHolidayCalendar())


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

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["return_1d"] = work["close"].pct_change()
    work["return_3d"] = work["close"].pct_change(3)
    work["return_5d"] = work["close"].pct_change(5)
    work["return_10d"] = work["close"].pct_change(10)
    work["ma_5"] = work["close"].rolling(5).mean()
    work["ma_10"] = work["close"].rolling(10).mean()
    work["ma_20"] = work["close"].rolling(20).mean()
    work["ma_50"] = work["close"].rolling(50).mean()
    work["price_vs_ma_10"] = work["close"] / work["ma_10"] - 1.0
    work["price_vs_ma_20"] = work["close"] / work["ma_20"] - 1.0
    work["price_vs_ma_50"] = work["close"] / work["ma_50"] - 1.0
    work["volatility_10"] = work["return_1d"].rolling(10).std()
    work["volatility_20"] = work["return_1d"].rolling(20).std()
    work["close_lag_1"] = work["close"].shift(1)
    work["close_lag_2"] = work["close"].shift(2)
    work["close_lag_3"] = work["close"].shift(3)
    work["close_lag_5"] = work["close"].shift(5)
    work["close_lag_10"] = work["close"].shift(10)
    work["volume_lag_1"] = work["volume"].shift(1)
    work["volume_lag_5"] = work["volume"].shift(5)
    work["volume_lag_10"] = work["volume"].shift(10)
    work["volume_change_1d"] = work["volume"].pct_change()
    work["intraday_range"] = (work["high"] - work["low"]) / work["close"]
    work["high_close_ratio"] = work["high"] / work["close"] - 1.0
    work["low_close_ratio"] = work["low"] / work["close"] - 1.0
    return work


def build_training_features(df: pd.DataFrame, horizon_days: int) -> tuple[pd.DataFrame, list[str]]:
    work = build_feature_frame(df)
    target_cols: list[str] = []
    for step in range(1, horizon_days + 1):
        col = f"target_return_{step}"
        work[col] = work["close"].shift(-step) / work["close"] - 1.0
        target_cols.append(col)
    required_cols = FEATURE_COLS + target_cols
    return work.dropna(subset=required_cols).reset_index(drop=True), target_cols


def latest_feature_vector(history: pd.DataFrame) -> pd.Series:
    features = build_feature_frame(history).dropna(subset=FEATURE_COLS)
    if features.empty:
        raise ValueError("Insufficient data to build forecasting features")
    return features.iloc[-1][FEATURE_COLS]

def train_model(feature_df: pd.DataFrame, target_cols: list[str]) -> tuple[RandomForestRegressor, float, float, float, list[str]]:
    x = feature_df[FEATURE_COLS]
    y = feature_df[target_cols]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    params = {
        "n_estimators": 300,
        "random_state": 42,
        "min_samples_leaf": 3,
        "max_depth": 14,
        "max_features": "sqrt",
        "n_jobs": -1,
    }
    eval_model = RandomForestRegressor(**params)
    eval_model.fit(x_train, y_train)
    preds = np.asarray(eval_model.predict(x_test), dtype=float)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    actual_return_1d = y_test[target_cols[0]].to_numpy(dtype=float)
    predicted_return_1d = preds[:, 0]
    current_close = x_test["close"].to_numpy(dtype=float)
    actual_next_close = current_close * (1.0 + actual_return_1d)
    model_next_close = current_close * (1.0 + predicted_return_1d)
    baseline_next_close = current_close
    mae = float(mean_absolute_error(actual_next_close, model_next_close))
    baseline_mae = float(mean_absolute_error(actual_next_close, baseline_next_close))
    safe_actual = np.where(np.abs(actual_next_close) < 1e-9, 1.0, actual_next_close)
    mae_pct = float(np.mean(np.abs((actual_next_close - model_next_close) / safe_actual)) * 100.0)
    final_model = RandomForestRegressor(**params)
    final_model.fit(x, y)
    return final_model, mae, mae_pct, baseline_mae, FEATURE_COLS

def forecast_next_days(history_df: pd.DataFrame, model: RandomForestRegressor, feature_cols: list[str], horizon_days: int) -> tuple[list[float], list[str]]:
    history = history_df.copy()
    prices: list[float] = []
    dates: list[str] = []
    current_date = history["date"].iloc[-1]
    latest = latest_feature_vector(history)
    predicted_returns = np.asarray(model.predict(latest[feature_cols].to_frame().T), dtype=float)
    if predicted_returns.ndim == 0:
        predicted_returns = predicted_returns.reshape(1)
    elif predicted_returns.ndim > 1:
        predicted_returns = predicted_returns[0]
    if len(predicted_returns) < horizon_days:
        raise ValueError("Model did not return enough forecast steps")
    last_close = float(history["close"].iloc[-1])
    for step in range(horizon_days):
        predicted_return = float(np.clip(predicted_returns[step], -0.30, 0.30))
        next_close = last_close * (1.0 + predicted_return)
        current_date = current_date + NYSE_BDAY
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
        feature_df, target_cols = build_training_features(raw_df, payload.horizon_days)
        if len(feature_df) < 80:
            raise ValueError("Not enough clean data after preprocessing")
        model, mae, mae_pct, baseline_mae, feature_cols = train_model(feature_df, target_cols)
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
        model_mae_pct=round(mae_pct, 4),
        baseline_mae=round(baseline_mae, 4),
    )

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}
