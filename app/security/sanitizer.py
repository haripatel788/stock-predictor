import re

from fastapi import HTTPException

from app.ticker import normalize_symbol

SQLI_PATTERNS = [
    re.compile(r"\b(UNION|SELECT|DROP|INSERT|DELETE|UPDATE)\b", re.I),
    re.compile(r"(--|;|/\*|\*/|xp_)", re.I),
    re.compile(r"\bOR\b\s+\d+\s*=\s*\d+|\bAND\b\s+\d+\s*=\s*\d+", re.I),
]
XSS_PATTERNS = [
    re.compile(r"<script[^>]*>", re.I),
    re.compile(r"javascript:", re.I),
    re.compile(r"on\w+\s*=", re.I),
    re.compile(r"<iframe", re.I),
    re.compile(r"<object", re.I),
]

INJECTION_PHRASES = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard your system prompt",
    "you are now",
    "act as if",
    "pretend you are",
    "forget everything",
    "new instructions:",
    "system prompt:",
]


def _reject_patterns(value: str, patterns: list[re.Pattern], detail: str) -> None:
    for pat in patterns:
        if pat.search(value):
            raise HTTPException(status_code=400, detail=detail)


def sanitize_ticker(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="Invalid ticker input")
    clean = raw.strip().upper()
    _reject_patterns(clean, SQLI_PATTERNS + XSS_PATTERNS, "Invalid ticker input")
    try:
        return normalize_symbol(clean)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def sanitize_chat_message(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(raw) > 2000:
        raise HTTPException(status_code=400, detail="Message too long (max 2000 chars)")
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
    _reject_patterns(clean, SQLI_PATTERNS + XSS_PATTERNS, "Message contains disallowed content")
    lower = clean.lower()
    for phrase in INJECTION_PHRASES:
        if phrase in lower:
            raise HTTPException(status_code=400, detail="Message contains disallowed content")
    return clean.strip()


def sanitize_string(raw: str, max_length: int = 500) -> str:
    if not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="Invalid input")
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
    return clean[:max_length].strip()
