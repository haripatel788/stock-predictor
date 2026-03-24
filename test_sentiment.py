from dotenv import load_dotenv
load_dotenv()

from app.sentiment import load_sentiment_cache, get_historical_sentiment, score_headlines_gemini, compute_weighted_sentiment
from app.news import fetch_live_headlines
import datetime

# ---- TEST 1: Cache loading ----
print("=" * 50)
print("TEST 1: Loading sentiment cache")
cache = load_sentiment_cache()
print(f"Cache rows: {len(cache):,}")
print(f"Cache tickers: {cache['stock'].nunique()}")
print("PASS" if len(cache) > 0 else "FAIL — cache is empty")

# ---- TEST 2: Historical sentiment lookup ----
print("\n" + "=" * 50)
print("TEST 2: Historical sentiment lookup")
result = get_historical_sentiment("AA", datetime.date(2020, 1, 15))
print(f"Sentiment for AA on 2020-01-15: {result}")
print("PASS" if isinstance(result, dict) else "FAIL")

result_unknown = get_historical_sentiment("FAKEFAKE", datetime.date(2020, 1, 15))
print(f"Sentiment for unknown ticker: {result_unknown}")
print("PASS — fallback works" if result_unknown["daily_sentiment"] == 0.0 else "FAIL")

# ---- TEST 3: NewsAPI live headlines ----
print("\n" + "=" * 50)
print("TEST 3: Fetching live headlines from NewsAPI")
headlines = fetch_live_headlines("AAPL")
print(f"Headlines fetched: {len(headlines)}")
for h in headlines:
    print(f"  - {h}")
print("PASS" if isinstance(headlines, list) else "FAIL")

# ---- TEST 4: Gemini scoring ----
print("\n" + "=" * 50)
print("TEST 4: Gemini API scoring")
if headlines:
    scored = score_headlines_gemini(headlines[:2], "AAPL")
    for s in scored:
        print(f"  sentiment: {s['sentiment']:+.2f} | relevance: {s['relevance']:.2f} | {s['headline'][:60]}")
        print(f"  reasoning: {s['reasoning']}")
    weighted = compute_weighted_sentiment(scored)
    print(f"  weighted_sentiment: {weighted:+.4f}")
    print("PASS" if scored[0]['reasoning'] != 'Scoring unavailable' else "FAIL — quota still exhausted")
else:
    print("SKIP — no headlines fetched")

print("\n" + "=" * 50)
print("All tests complete.")