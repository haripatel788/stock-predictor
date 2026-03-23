import os
import requests
TICKER_TO_COMPANY = {
    "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon",
    "GOOGL": "Google", "META": "Meta", "TSLA": "Tesla",
    "NVDA": "Nvidia", "JPM": "JPMorgan", "BAC": "Bank of America",
    "GS": "Goldman Sachs", "NFLX": "Netflix", "AMD": "AMD",
    "INTC": "Intel", "UBER": "Uber", "LYFT": "Lyft",
    "SNAP": "Snapchat", "SPOT": "Spotify", "SHOP": "Shopify",
    "SQ": "Block Square", "PYPL": "PayPal",
}

def fetch_live_headlines(ticker: str, max_headlines: int = 5) -> list[str]:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return []

    query = TICKER_TO_COMPANY.get(ticker.upper(), ticker.upper())

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
            a["title"] for a in articles
            if a.get("title") and a["title"] != "[Removed]"
        ]
        return headlines[:max_headlines]

    except Exception:
        return []