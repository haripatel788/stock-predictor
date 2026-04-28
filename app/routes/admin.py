from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.auth import get_supabase_required, require_admin
from app.security.audit import AuditEvent, audit_log
from app.security.rate_limiter import enforce_admin_limits
from app.tiers import TIER_LIMITS

router = APIRouter(prefix="/admin", tags=["admin"])


class AdminTierUpdate(BaseModel):
    tier: str


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


@router.get("/users")
def admin_list_users(
    request: Request,
    _admin: dict = Depends(require_admin),
) -> dict:
    enforce_admin_limits(request)
    sb = get_supabase_required()
    res = (
        sb.table("profiles")
        .select("id,email,display_name,tier,forecasts_today,forecasts_today_reset,created_at")
        .order("created_at", desc=True)
        .limit(500)
        .execute()
    )
    return {"items": res.data or []}


@router.patch("/users/{user_id}/tier")
def admin_update_user_tier(
    user_id: str,
    body: AdminTierUpdate,
    request: Request,
    admin: dict = Depends(require_admin),
) -> dict:
    enforce_admin_limits(request)
    tier = str(body.tier or "").strip().lower()
    if tier not in TIER_LIMITS or tier == "public":
        raise HTTPException(status_code=400, detail="Invalid tier")
    sb = get_supabase_required()
    sb.table("profiles").update({"tier": tier}).eq("id", user_id).execute()
    audit_log(
        AuditEvent.ADMIN_ACTION,
        request,
        user_id=admin.get("id"),
        details={"action": "update_tier", "target_user": user_id, "tier": tier},
    )
    return {"ok": True, "id": user_id, "tier": tier}
