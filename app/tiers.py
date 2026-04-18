from datetime import date

from fastapi import HTTPException

from app.auth import get_supabase_client

TIER_LIMITS: dict[str, dict[str, int]] = {
    "public": {"forecasts_per_day": 3, "max_horizon": 7},
    "free": {"forecasts_per_day": 15, "max_horizon": 14},
    "pro": {"forecasts_per_day": 9999, "max_horizon": 30},
    "admin": {"forecasts_per_day": 9999, "max_horizon": 30},
}


def effective_tier(user: dict | None) -> str:
    if user is None:
        return "public"
    return str(user.get("tier") or "free")


def assert_horizon_allowed(tier: str, horizon_days: int) -> None:
    max_h = TIER_LIMITS[tier]["max_horizon"]
    if horizon_days > max_h:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum forecast horizon for your access level is {max_h} days",
        )


def enforce_authenticated_daily_forecast(user: dict) -> None:
    tier = effective_tier(user)
    if tier == "public":
        return
    limit = TIER_LIMITS[tier]["forecasts_per_day"]
    today = date.today().isoformat()
    reset = user.get("forecasts_today_reset")
    count = int(user.get("forecasts_today") or 0)
    if reset is None or reset != today:
        return
    if count >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Daily forecast limit reached for {tier} accounts",
        )


def record_forecast_usage(user: dict | None) -> None:
    if not user:
        return
    sb = get_supabase_client()
    if sb is None:
        return
    today = date.today().isoformat()
    uid = user["id"]
    reset = user.get("forecasts_today_reset")
    count = int(user.get("forecasts_today") or 0)
    if reset != today:
        sb.table("profiles").update({"forecasts_today": 1, "forecasts_today_reset": today}).eq("id", uid).execute()
    else:
        sb.table("profiles").update({"forecasts_today": count + 1}).eq("id", uid).execute()


def save_forecast_row(user: dict, row: dict) -> None:
    sb = get_supabase_client()
    if sb is None or not user:
        return
    payload = {"user_id": user["id"], **row}
    sb.table("forecasts").insert(payload).execute()
