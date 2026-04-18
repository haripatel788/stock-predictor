from fastapi.testclient import TestClient

from app.main import app

UA = {"User-Agent": "pytest"}

client = TestClient(app)

def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'

def test_predict_rejects_empty_symbol():
    response = client.post('/api/predict', json={'symbol': '', 'horizon_days': 5}, headers=UA)
    assert response.status_code == 422


def test_predict_rejects_invalid_ticker():
    response = client.post(
        '/api/predict',
        json={'symbol': '!!!', 'horizon_days': 5},
        headers=UA,
    )
    assert response.status_code == 400


def test_bootstrap_public():
    response = client.get("/api/bootstrap", headers=UA)
    assert response.status_code == 200
    data = response.json()
    assert "auth" in data and "tier_limits" in data
    assert "enabled" in data["auth"]
    assert "public" in data["tier_limits"]