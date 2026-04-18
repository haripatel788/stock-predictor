import pytest

from app.sentiment import compute_weighted_sentiment


def test_weighted_sentiment_uses_relevance_denominator():
    scored = [
        {"sentiment": 1.0, "relevance": 0.1},
        {"sentiment": -1.0, "relevance": 0.9},
    ]
    # (1*0.1 + -1*0.9) / (0.1+0.9) = -0.8
    assert compute_weighted_sentiment(scored) == pytest.approx(-0.8)


def test_weighted_sentiment_zero_when_no_relevance():
    assert compute_weighted_sentiment([{"sentiment": 0.5, "relevance": 0.0}]) == 0.0
