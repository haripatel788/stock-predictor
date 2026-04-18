import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.auth import get_optional_user
from app.chat_service import run_chat_turn
from app.security.audit import AuditEvent, audit_log
from app.security.rate_limiter import enforce_chat_limits
from app.security.sanitizer import sanitize_chat_message
from app.tiers import effective_tier

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)


@router.post("")
def chat_endpoint(
    request: Request,
    body: ChatRequest,
    user: dict | None = Depends(get_optional_user),
) -> dict[str, str]:
    enforce_chat_limits(request)
    tier = effective_tier(user)
    if tier == "public" and os.getenv("CHAT_ALLOW_PUBLIC", "").lower() not in ("1", "true", "yes"):
        raise HTTPException(
            status_code=402,
            detail="Chat requires a signed-in user, or set CHAT_ALLOW_PUBLIC=1 for local demos.",
        )

    normalized: list[dict[str, Any]] = []
    for m in body.messages:
        if m.role == "user":
            normalized.append({"role": "user", "content": sanitize_chat_message(m.content)})
        else:
            normalized.append({"role": m.role, "content": (m.content or "")[:8000]})

    audit_log(
        AuditEvent.CHAT_MESSAGE,
        request,
        user_id=(user or {}).get("id"),
        details={"tier": tier},
    )
    reply = run_chat_turn(normalized)
    return {"reply": reply}
