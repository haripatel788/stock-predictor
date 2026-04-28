import os
from functools import lru_cache

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import Client, create_client

from app.security.audit import AuditEvent, audit_log

security = HTTPBearer(auto_error=False)


@lru_cache
def get_supabase_client() -> Client | None:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def supabase_enabled() -> bool:
    return get_supabase_client() is not None


def get_supabase_required() -> Client:
    """Service-role client; raises 503 if Supabase env is not configured."""
    sb = get_supabase_client()
    if sb is None:
        raise HTTPException(
            status_code=503,
            detail="Account features are not available (server missing Supabase configuration).",
        )
    return sb


async def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict | None:
    if not credentials:
        return None
    sb = get_supabase_client()
    if sb is None:
        return None
    token = credentials.credentials
    try:
        res = sb.auth.get_user(token)
        if not res or not getattr(res, "user", None):
            audit_log(AuditEvent.INVALID_TOKEN, request)
            return None
        uid = res.user.id
        email = getattr(res.user, "email", None)
        email_confirmed = bool(getattr(res.user, "email_confirmed_at", None))
        prof = sb.table("profiles").select("*").eq("id", uid).limit(1).execute()
        rows = prof.data or []
        data = rows[0] if rows else None
        if not data:
            return {
                "id": uid,
                "email": email,
                "tier": "free",
                "display_name": None,
                "email_verified": email_confirmed,
                "forecasts_today": 0,
                "forecasts_today_reset": None,
            }
        return {**data, "email_verified": email_confirmed}
    except Exception:
        audit_log(AuditEvent.INVALID_TOKEN, request)
        return None


async def get_current_user(
    request: Request,
    user: dict | None = Depends(get_optional_user),
) -> dict:
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("tier") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
