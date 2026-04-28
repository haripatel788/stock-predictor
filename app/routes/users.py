from fastapi import APIRouter, Depends

from app.auth import get_current_user
from app.tiers import TIER_LIMITS, effective_tier

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
def read_me(user: dict = Depends(get_current_user)) -> dict:
    tier = effective_tier(user)
    if tier not in TIER_LIMITS:
        tier = "free"
    lim = TIER_LIMITS[tier]
    return {
        "id": user.get("id"),
        "email": user.get("email"),
        "tier": tier,
        "forecasts_today": user.get("forecasts_today"),
        "forecasts_today_reset": user.get("forecasts_today_reset"),
        "limits": {
            "forecasts_per_day": lim["forecasts_per_day"],
            "max_horizon": lim["max_horizon"],
        },
    }
