import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import Request

logger = logging.getLogger("marketpulse.audit")


class AuditEvent(str, Enum):
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"
    FORECAST_RUN = "forecast.run"
    FORECAST_BLOCKED = "forecast.blocked"
    RATE_LIMITED = "security.rate_limited"
    ADMIN_ACTION = "admin.action"
    TIER_CHANGED = "admin.tier_changed"
    KILL_SWITCH = "admin.kill_switch"
    INVALID_TOKEN = "security.invalid_token"
    SUSPICIOUS_INPUT = "security.suspicious_input"
    PROMPT_INJECTION = "security.prompt_injection"
    CHAT_MESSAGE = "chat.message"


def _ensure_log_dir() -> None:
    path = os.getenv("AUDIT_LOG_PATH", "logs/audit.log")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


_ensure_log_dir()
_audit_file = os.getenv("AUDIT_LOG_PATH", "logs/audit.log")
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "").endswith("audit.log") for h in logger.handlers):
    try:
        fh = logging.FileHandler(_audit_file)
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    except OSError:
        pass


def audit_log(
    event: AuditEvent,
    request: Request,
    *,
    user_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    import hashlib

    ip = request.client.host if request.client else "unknown"
    ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:16]
    rid = getattr(request.state, "request_id", None)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event.value,
        "request_id": rid,
        "ip_hash": ip_hash,
        "user_agent": (request.headers.get("user-agent") or "")[:200],
        "path": str(request.url.path),
        "user_id": user_id,
        "details": details or {},
    }
    logger.info(json.dumps(entry))
