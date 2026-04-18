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