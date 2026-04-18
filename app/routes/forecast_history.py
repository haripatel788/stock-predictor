from fastapi import APIRouter, Depends, Query

from app.auth import get_current_user, get_supabase_required

router = APIRouter(prefix="/forecasts", tags=["forecasts"])


@router.get("/history")
def forecast_history(
    user: dict = Depends(get_current_user),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    sb = get_supabase_required()
    res = (
        sb.table("forecasts")
        .select(
            "id,symbol,horizon,last_close,predicted_prices,predicted_dates,model_mae,created_at"
        )
        .eq("user_id", user["id"])
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return {"items": res.data or []}
