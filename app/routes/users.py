from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.auth import get_current_user, get_supabase_required
from app.tiers import TIER_LIMITS, effective_tier

router = APIRouter(prefix="/users", tags=["users"])


class ProfileUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=60)


@router.get("/me")
def read_me(user: dict = Depends(get_current_user)) -> dict:
    tier = effective_tier(user)
    if tier not in TIER_LIMITS:
        tier = "free"
    lim = TIER_LIMITS[tier]
    return {
        "id": user.get("id"),
        "email": user.get("email"),
        "email_verified": bool(user.get("email_verified")),
        "tier": tier,
        "display_name": user.get("display_name"),
        "forecasts_today": user.get("forecasts_today"),
        "forecasts_today_reset": user.get("forecasts_today_reset"),
        "limits": {
            "forecasts_per_day": lim["forecasts_per_day"],
            "max_horizon": lim["max_horizon"],
        },
    }


@router.patch("/me")
def update_me(body: ProfileUpdateRequest, user: dict = Depends(get_current_user)) -> dict:
    sb = get_supabase_required()
    display_name = (body.display_name or "").strip() or None
    sb.table("profiles").upsert(
        {
            "id": user["id"],
            "email": user.get("email"),
            "display_name": display_name,
        }
    ).execute()
    next_user = {**user, "display_name": display_name}
    return read_me(next_user)
