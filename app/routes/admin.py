from fastapi import APIRouter, Depends, Request

from app.auth import require_admin
from app.security.audit import AuditEvent, audit_log
from app.security.rate_limiter import enforce_admin_limits

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/ping")
def admin_ping(
    request: Request,
    _admin: dict = Depends(require_admin),
) -> dict[str, str]:
    enforce_admin_limits(request)
    audit_log(AuditEvent.ADMIN_ACTION, request, user_id=_admin.get("id"), details={"action": "ping"})
    return {"status": "ok", "tier": "admin"}


@router.get("/settings/kill-switch")
def get_kill_switch_stub(
    request: Request,
    _admin: dict = Depends(require_admin),
) -> dict:
    enforce_admin_limits(request)
    return {"enabled": False, "note": "Wire to Supabase admin_settings when database is live"}
