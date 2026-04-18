import logging
import os
import threading
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

TICKER_TO_COMPANY: dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Google",
    "GOOG": "Google",
    "META": "Meta",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "JPM": "JPMorgan",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "INTC": "Intel",
    "UBER": "Uber",
    "LYFT": "Lyft",
    "SNAP": "Snapchat",
    "SPOT": "Spotify",
    "SHOP": "Shopify",
    "SQ": "Block",
    "PYPL": "PayPal",
    "DIS": "Disney",
    "BABA": "Alibaba",
    "V": "Visa",
    "MA": "Mastercard",
    "WMT": "Walmart",
    "COST": "Costco",
    "PG": "Procter Gamble",
    "JNJ": "Johnson Johnson",
    "XOM": "ExxonMobil",
    "CVX": "Chevron",
}

_HEADLINE_CACHE_TTL = int(os.getenv("NEWS_HEADLINE_CACHE_TTL_SEC", "900"))
_headline_cache: dict[str, tuple[float, list[str]]] = {}
_headline_cache_lock = threading.Lock()


def _cache_key(ticker: str) -> str:
    return ticker.upper()


def resolve_news_query(ticker: str) -> str:
    t = ticker.upper()
    if t in TICKER_TO_COMPANY:
        return TICKER_TO_COMPANY[t]
    try:
        r = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": t, "quotesCount": 1, "newsCount": 0},
            headers={"User-Agent": UA},
            timeout=8,
        )
        r.raise_for_status()
        data: dict[str, Any] = r.json()
        quotes = data.get("quotes") or []
        if quotes:
            q0 = quotes[0]
            name = q0.get("shortname") or q0.get("longname") or q0.get("symbol")
            if name:
                return str(name)
    except Exception:
        logger.exception("Yahoo search failed for ticker=%s", t)
    return t


def fetch_live_headlines(ticker: str, max_headlines: int = 5) -> list[str]:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        logger.info("NEWS_API_KEY not set; skipping NewsAPI")
        return []

    key = _cache_key(ticker)
    now = time.time()
    with _headline_cache_lock:
        hit = _headline_cache.get(key)
        if hit is not None and now - hit[0] < _HEADLINE_CACHE_TTL:
            logger.debug("headline cache hit ticker=%s", key)
            return hit[1][:max_headlines]

    query = resolve_news_query(ticker)

    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": f"{query} stock",
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": max_headlines,
                "apiKey": api_key,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        articles = data.get("articles", [])
        headlines = [
            a["title"]
            for a in articles
            if a.get("title") and a["title"] != "[Removed]"
        ]
        out = headlines[:max_headlines]
        with _headline_cache_lock:
            _headline_cache[key] = (time.time(), out)
        logger.info("NewsAPI ticker=%s query=%r headlines=%s", key, query, len(out))
        return out

    except Exception:
        logger.exception("NewsAPI request failed ticker=%s", ticker)
        return []
