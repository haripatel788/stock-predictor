"""Indirection so chat routes can call forecast logic without circular imports."""

from typing import Any, Callable

_runner: Callable[[str, int], Any] | None = None


def register_forecast_runner(fn: Callable[[str, int], Any]) -> None:
    global _runner
    _runner = fn


def run_forecast_via_tool(symbol: str, horizon_days: int) -> Any:
    if _runner is None:
        raise RuntimeError("Forecast runner not registered")
    return _runner(symbol, horizon_days)
