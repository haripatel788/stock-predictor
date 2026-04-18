from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import FEATURE_COLS, app


class ConstantReturnModel:
    def predict(self, x):
        n = len(x) if hasattr(x, "__len__") else 1
        return np.full((n, 1), 0.01, dtype=float)


def _history(symbol_end: str = "2025-02-14", n: int = 130) -> pd.DataFrame:
    dates = pd.bdate_range(end=symbol_end, periods=n)
    closes = [100.0 + i * 0.05 for i in range(n)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1_000_000 + i for i in range(n)],
        }
    )


@patch("app.main.fetch_live_headlines", return_value=[])
@patch("app.main.get_or_train_cached_model")
@patch("app.main.fetch_history")
def test_predict_returns_bands(mock_fetch, mock_train, _mock_news):
    mock_fetch.return_value = (_history(), "Test")
    mock_train.return_value = (ConstantReturnModel(), 1.25, 0.5, 2.0, FEATURE_COLS)

    client = TestClient(app)
    res = client.post("/api/predict", json={"symbol": "AAPL", "horizon_days": 3})
    assert res.status_code == 200
    data = res.json()
    assert len(data["predicted_prices"]) == 3
    assert len(data["predicted_price_low"]) == 3
    assert len(data["predicted_price_high"]) == 3
    assert data["predicted_price_low"][0] == round(data["predicted_prices"][0] - data["model_mae"], 2)
