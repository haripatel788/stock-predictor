import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_WORKERS = int(os.getenv("GROQ_MAX_WORKERS", "8"))

_cache: pd.DataFrame | None = None


def load_sentiment_cache() -> pd.DataFrame:
    global _cache
    if _cache is not None:
        return _cache

    cache_path = Path(__file__).parent.parent / "data" / "sentiment_cache.csv"

    if not cache_path.exists():
        _cache = pd.DataFrame(
            columns=["stock", "date", "daily_sentiment", "sentiment_3d_avg", "sentiment_momentum"]
        )
        return _cache

    _cache = pd.read_csv(cache_path)
    _cache["date"] = pd.to_datetime(_cache["date"]).dt.date
    logger.info("Sentiment cache loaded: %s rows, %s tickers", f"{len(_cache):,}", _cache["stock"].nunique())
    return _cache


def get_historical_sentiment(ticker: str, date) -> dict:
    cache = load_sentiment_cache()

    if cache.empty:
        return {"daily_sentiment": 0.0, "sentiment_3d_avg": 0.0, "sentiment_momentum": 0.0}

    row = cache[(cache["stock"] == ticker) & (cache["date"] == date)]

    if row.empty:
        return {"daily_sentiment": 0.0, "sentiment_3d_avg": 0.0, "sentiment_momentum": 0.0}

    return {
        "daily_sentiment": float(row["daily_sentiment"].iloc[0]),
        "sentiment_3d_avg": float(row["sentiment_3d_avg"].iloc[0]),
        "sentiment_momentum": float(row["sentiment_momentum"].iloc[0]),
    }


def _score_one_groq(headline: str, ticker: str, api_key: str) -> dict:
    prompt = (
        f'You are a financial analyst. Score this news headline for {ticker} stock.\n\n'
        f'Headline: "{headline}"\n\n'
        f'Return ONLY a valid JSON object with no extra text, no markdown, no backticks:\n'
        f'{{"sentiment": 0.0, "relevance": 0.0, "reasoning": "one sentence explanation"}}\n\n'
        f'Rules:\n'
        f'- sentiment: -1.0 (very bearish) to 1.0 (very bullish), 0.0 = neutral\n'
        f'- relevance: 0.0 (unrelated noise) to 1.0 (directly impacts stock price)\n'
        f'- reasoning: one short sentence explaining your score'
    )
    try:
        with httpx.Client() as client:
            response = client.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "temperature": 0,
                    "max_tokens": 256,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30.0,
            )
            response.raise_for_status()
            payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("no choices in Groq response")
        text = (choices[0].get("message") or {}).get("content", "").strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.lower().startswith("json"):
                text = text[4:]
        text = text.strip()
        parsed = json.loads(text)
        return {
            "headline": headline,
            "sentiment": float(parsed.get("sentiment", 0.0)),
            "relevance": float(parsed.get("relevance", 0.0)),
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception as exc:
        logger.warning("Groq headline score failed: %s", exc)
        return {
            "headline": headline,
            "sentiment": 0.0,
            "relevance": 0.0,
            "reasoning": "Scoring unavailable",
        }


def score_headlines_groq(headlines: list[str], ticker: str) -> list[dict]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return [
            {"headline": h, "sentiment": 0.0, "relevance": 0.0, "reasoning": "No GROQ_API_KEY configured"}
            for h in headlines
        ]

    if not headlines:
        return []

    workers = min(GROQ_MAX_WORKERS, max(1, len(headlines)))
    results: list[dict | None] = [None] * len(headlines)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(_score_one_groq, h, ticker, api_key): idx for idx, h in enumerate(headlines)
        }
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                logger.warning("headline worker failed idx=%s: %s", idx, exc)
                results[idx] = {
                    "headline": headlines[idx],
                    "sentiment": 0.0,
                    "relevance": 0.0,
                    "reasoning": "Scoring unavailable",
                }

    return [r for r in results if r is not None]


def score_headlines_gemini(headlines: list[str], ticker: str) -> list[dict]:
    """Deprecated: use score_headlines_groq (Gemini removed)."""
    logger.warning("score_headlines_gemini is deprecated; calling Groq instead")
    return score_headlines_groq(headlines, ticker)


def compute_weighted_sentiment(scored_headlines: list[dict]) -> float:
    if not scored_headlines:
        return 0.0
    num = 0.0
    den = 0.0
    for h in scored_headlines:
        s = float(h.get("sentiment", 0.0))
        r = float(h.get("relevance", 0.0))
        num += s * r
        den += r
    if den <= 0.0:
        return 0.0
    return round(num / den, 6)
