"""Shared ticker normalization (used by API + security sanitizer)."""

import re


def normalize_symbol(raw_symbol: str) -> str:
    clean = raw_symbol.strip().upper()
    if not clean:
        raise ValueError("Ticker symbol is required")
    if clean.startswith("^"):
        if not re.fullmatch(r"\^[A-Z][A-Z0-9.\-]{0,14}", clean):
            raise ValueError("Invalid index symbol format")
        return clean
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", clean):
        raise ValueError("Use 1–10 characters: start with A–Z, then letters, digits, dot, or hyphen")
    return clean
