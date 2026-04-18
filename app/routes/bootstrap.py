import os

from fastapi import APIRouter

from app.tiers import TIER_LIMITS

router = APIRouter(prefix="/bootstrap", tags=["bootstrap"])


@router.get("")
def bootstrap() -> dict:
    """Public values for the browser (anon key is safe to expose; never the service_role key)."""
    url = (os.getenv("SUPABASE_URL") or "").strip()
    anon = (os.getenv("SUPABASE_ANON_KEY") or "").strip()
    return {
        "auth": {
            "enabled": bool(url and anon),
            "supabase_url": url,
            "supabase_anon_key": anon,
        },
        "tier_limits": TIER_LIMITS,
    }
