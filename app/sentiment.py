import os
import json
import pandas as pd
from pathlib import Path

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
    print(f"Sentiment cache loaded: {len(_cache):,} rows, {_cache['stock'].nunique()} tickers")
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


def score_headlines_gemini(headlines: list[str], ticker: str) -> list[dict]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return [
            {"headline": h, "sentiment": 0.0, "relevance": 0.0, "reasoning": "No API key configured"}
            for h in headlines
        ]

    try:
        from google import genai
    except ImportError:
        return [
            {"headline": h, "sentiment": 0.0, "relevance": 0.0, "reasoning": "google-genai not installed"}
            for h in headlines
        ]

    client = genai.Client(api_key=api_key)
    results = []

    for headline in headlines:
        try:
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

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            text = response.text.strip()
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else text
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            parsed = json.loads(text)
            results.append({
                "headline": headline,
                "sentiment": float(parsed.get("sentiment", 0.0)),
                "relevance": float(parsed.get("relevance", 0.0)),
                "reasoning": str(parsed.get("reasoning", "")),
            })

        except Exception:
            results.append({
                "headline": headline,
                "sentiment": 0.0,
                "relevance": 0.0,
                "reasoning": "Scoring unavailable",
            })

    return results


def compute_weighted_sentiment(scored_headlines: list[dict]) -> float:
    if not scored_headlines:
        return 0.0

    weighted_scores = [h["sentiment"] * h["relevance"] for h in scored_headlines]
    return round(sum(weighted_scores) / len(weighted_scores), 6)