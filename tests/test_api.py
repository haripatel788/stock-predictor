from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'

def test_predict_rejects_empty_symbol():
    response = client.post('/api/predict', json={'symbol': '', 'horizon_days': 5})
    assert response.status_code == 422