import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

ADMIN_PREFIXES = ("/api/admin",)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Request ID, body size cap, optional user-agent gate for /api/*."""

    max_body_bytes = 1_048_576

    async def dispatch(self, request: Request, call_next) -> Response:
        request.state.request_id = str(uuid.uuid4())[:12]
        cl = request.headers.get("content-length")
        if cl:
            try:
                if int(cl) > self.max_body_bytes:
                    return JSONResponse(
                        {"detail": "Request body too large"},
                        status_code=413,
                    )
            except ValueError:
                pass

        path = request.url.path
        if path.startswith("/api/") and not path.startswith("/api/health"):
            ua = request.headers.get("user-agent", "")
            if not ua.strip():
                return JSONResponse({"detail": "Forbidden"}, status_code=403)

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Response-Time"] = f"{duration_ms}ms"
        return response


class AdminPathGuardMiddleware(BaseHTTPMiddleware):
    """Require Authorization header on /api/admin/* (tier checked in route)."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if any(path.startswith(p) for p in ADMIN_PREFIXES):
            auth = request.headers.get("authorization", "")
            if not auth.startswith("Bearer "):
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)
