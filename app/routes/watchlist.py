from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.auth import get_current_user, get_supabase_required
from app.security.sanitizer import sanitize_ticker

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


class WatchlistAdd(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)


@router.get("")
def list_watchlist(user: dict = Depends(get_current_user)) -> dict:
    sb = get_supabase_required()
    res = (
        sb.table("watchlists")
        .select("id,symbol,added_at")
        .eq("user_id", user["id"])
        .order("added_at", desc=True)
        .execute()
    )
    return {"items": res.data or []}


@router.post("")
def add_watchlist(body: WatchlistAdd, user: dict = Depends(get_current_user)) -> dict:
    sb = get_supabase_required()
    sym = sanitize_ticker(body.symbol)
    existing = (
        sb.table("watchlists")
        .select("id")
        .eq("user_id", user["id"])
        .eq("symbol", sym)
        .limit(1)
        .execute()
    )
    if existing.data:
        return {"symbol": sym, "already": True}
    sb.table("watchlists").insert({"user_id": user["id"], "symbol": sym}).execute()
    return {"symbol": sym, "already": False}


@router.delete("")
def remove_watchlist(symbol: str, user: dict = Depends(get_current_user)) -> dict:
    sb = get_supabase_required()
    sym = sanitize_ticker(symbol)
    sb.table("watchlists").delete().eq("user_id", user["id"]).eq("symbol", sym).execute()
    return {"removed": sym}
