from fastapi import APIRouter, Depends

from app.auth import get_current_user

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
def read_me(user: dict = Depends(get_current_user)) -> dict:
    return {"id": user.get("id"), "email": user.get("email"), "tier": user.get("tier")}
