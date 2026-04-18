import hashlib
import time
from collections import defaultdict
from threading import Lock

from fastapi import HTTPException, Request

from app.security.audit import AuditEvent, audit_log


class RateLimiter:
    """Sliding-window in-memory limiter (replace with Redis for multi-instance)."""

    def __init__(self) -> None:
        self._store: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    @staticmethod
    def _client_key(request: Request) -> str:
        ip = request.client.host if request.client else "unknown"
        return hashlib.sha256(ip.encode()).hexdigest()[:16]

    def check(self, request: Request, scope: str, max_calls: int, window_seconds: int) -> None:
        key = f"{scope}:{self._client_key(request)}"
        now = time.time()
        window_start = now - window_seconds

        with self._lock:
            self._store[key] = [t for t in self._store[key] if t > window_start]
            if len(self._store[key]) >= max_calls:
                oldest = self._store[key][0]
                retry_after = int(oldest + window_seconds - now) + 1
                audit_log(AuditEvent.RATE_LIMITED, request, details={"scope": scope, "retry_after": retry_after})
                raise HTTPException(
                    status_code=429,
                    detail={"error": "Rate limit exceeded", "retry_after_seconds": retry_after, "scope": scope},
                    headers={"Retry-After": str(retry_after)},
                )
            self._store[key].append(now)


limiter = RateLimiter()

LIMITS: dict[str, dict[str, int]] = {
    "predict": {"max_calls": 10, "window": 60},
    "predict_daily_public": {"max_calls": 3, "window": 86400},
    "chat": {"max_calls": 30, "window": 60},
    "auth": {"max_calls": 5, "window": 300},
    "admin": {"max_calls": 100, "window": 60},
    "global": {"max_calls": 200, "window": 60},
}


def enforce_predict_limits(request: Request, tier_public: bool) -> None:
    p = LIMITS["predict"]
    limiter.check(request, "predict", p["max_calls"], p["window"])
    if tier_public:
        d = LIMITS["predict_daily_public"]
        limiter.check(request, "predict_daily_public", d["max_calls"], d["window"])
    g = LIMITS["global"]
    limiter.check(request, "global", g["max_calls"], g["window"])


def enforce_chat_limits(request: Request) -> None:
    c = LIMITS["chat"]
    limiter.check(request, "chat", c["max_calls"], c["window"])
    g = LIMITS["global"]
    limiter.check(request, "global", g["max_calls"], g["window"])


def enforce_admin_limits(request: Request) -> None:
    a = LIMITS["admin"]
    limiter.check(request, "admin", a["max_calls"], a["window"])
    g = LIMITS["global"]
    limiter.check(request, "global", g["max_calls"], g["window"])
