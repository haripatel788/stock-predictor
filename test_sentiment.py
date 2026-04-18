"""Manual smoke script for sentiment + news (run with API keys in .env)."""

from dotenv import load_dotenv

load_dotenv()

from app.sentiment import load_sentiment_cache, get_historical_sentiment, score_headlines_groq, compute_weighted_sentiment
from app.news import fetch_live_headlines
import datetime

print("=" * 50)
print("TEST 1: Loading sentiment cache")
cache = load_sentiment_cache()
print(f"Cache rows: {len(cache):,}")
print(f"Cache tickers: {cache['stock'].nunique()}")
print("PASS" if len(cache) > 0 else "NOTE — cache is empty (optional for live path)")

print("\n" + "=" * 50)
print("TEST 2: Historical sentiment lookup")
result = get_historical_sentiment("AA", datetime.date(2020, 1, 15))
print(f"Sentiment for AA on 2020-01-15: {result}")
print("PASS" if isinstance(result, dict) else "FAIL")

result_unknown = get_historical_sentiment("FAKEFAKE", datetime.date(2020, 1, 15))
print(f"Sentiment for unknown ticker: {result_unknown}")
print("PASS — fallback works" if result_unknown["daily_sentiment"] == 0.0 else "FAIL")

print("\n" + "=" * 50)
print("TEST 3: Fetching live headlines from NewsAPI")
headlines = fetch_live_headlines("AAPL")
print(f"Headlines fetched: {len(headlines)}")
for h in headlines:
    print(f"  - {h}")
print("PASS" if isinstance(headlines, list) else "FAIL")

print("\n" + "=" * 50)
print("TEST 4: Groq API scoring")
if headlines:
    scored = score_headlines_groq(headlines[:2], "AAPL")
    for s in scored:
        print(f"  sentiment: {s['sentiment']:+.2f} | relevance: {s['relevance']:.2f} | {s['headline'][:60]}")
        print(f"  reasoning: {s['reasoning']}")
    weighted = compute_weighted_sentiment(scored)
    print(f"  weighted_sentiment: {weighted:+.4f}")
    print("PASS" if scored[0]["reasoning"] != "Scoring unavailable" else "FAIL — check GROQ_API_KEY / quota")
else:
    print("SKIP — no headlines fetched")

print("\n" + "=" * 50)
print("All tests complete.")
