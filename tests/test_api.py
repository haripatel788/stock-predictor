from fastapi.testclient import TestClient

from app.auth import get_current_user, require_admin
from app.main import app
from app.routes import admin as admin_routes

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


def test_users_me_requires_auth():
    response = client.get("/api/users/me", headers=UA)
    assert response.status_code == 401


def test_users_me_returns_profile(monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: {
        "id": "u1",
        "email": "test@example.com",
        "email_verified": True,
        "tier": "free",
        "display_name": "Test User",
        "forecasts_today": 2,
        "forecasts_today_reset": "2026-04-28",
    }
    try:
        response = client.get("/api/users/me", headers=UA)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    assert response.status_code == 200
    body = response.json()
    assert body["display_name"] == "Test User"
    assert body["email_verified"] is True
    assert body["limits"]["max_horizon"] >= 1


def test_patch_users_me_validates_display_name(monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: {
        "id": "u1",
        "email": "test@example.com",
        "tier": "free",
        "display_name": None,
        "forecasts_today": 0,
        "forecasts_today_reset": None,
    }
    try:
        response = client.patch("/api/users/me", json={"display_name": "x" * 61}, headers=UA)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    assert response.status_code == 422


def test_patch_users_me_updates_profile(monkeypatch):
    class FakeExecute:
        def execute(self):
            return self

    class FakeTable:
        def upsert(self, _payload):
            return FakeExecute()

    class FakeSb:
        def table(self, _name):
            return FakeTable()

    monkeypatch.setattr("app.routes.users.get_supabase_required", lambda: FakeSb())
    app.dependency_overrides[get_current_user] = lambda: {
        "id": "u1",
        "email": "test@example.com",
        "tier": "free",
        "display_name": None,
        "forecasts_today": 0,
        "forecasts_today_reset": None,
    }
    try:
        response = client.patch("/api/users/me", json={"display_name": "Hari"}, headers=UA)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    assert response.status_code == 200
    body = response.json()
    assert body["display_name"] == "Hari"


def test_admin_users_requires_admin():
    response = client.get("/api/admin/users", headers=UA)
    assert response.status_code == 401


def test_admin_users_list_and_update(monkeypatch):
    class FakeExecute:
        def __init__(self, data=None):
            self.data = data or []

        def execute(self):
            return self

    class FakeTable:
        def __init__(self, name):
            self.name = name
            self._data = []

        def select(self, _query):
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def update(self, _payload):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def execute(self):
            if self.name == "profiles":
                return FakeExecute(
                    [
                        {
                            "id": "u1",
                            "email": "x@example.com",
                            "display_name": "X",
                            "tier": "free",
                            "forecasts_today": 0,
                            "forecasts_today_reset": None,
                            "created_at": "2026-04-28T00:00:00Z",
                        }
                    ]
                )
            return FakeExecute([])

    class FakeSb:
        def table(self, name):
            return FakeTable(name)

    app.dependency_overrides[admin_routes.require_admin] = lambda: {"id": "admin-1", "tier": "admin"}
    monkeypatch.setattr("app.routes.admin.get_supabase_required", lambda: FakeSb())
    try:
        admin_headers = {**UA, "Authorization": "Bearer test-admin-token"}
        list_res = client.get("/api/admin/users", headers=admin_headers)
        patch_res = client.patch("/api/admin/users/u1/tier", json={"tier": "pro"}, headers=admin_headers)
    finally:
        app.dependency_overrides.pop(admin_routes.require_admin, None)
    assert list_res.status_code == 200
    assert list_res.json()["items"][0]["email"] == "x@example.com"
    assert patch_res.status_code == 200
    assert patch_res.json()["tier"] == "pro"