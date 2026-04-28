import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=()"
        )
        csp = os.getenv(
            "CSP_HEADER",
            (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://fonts.gstatic.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data:; "
                "connect-src 'self' https://api.groq.com https://newsapi.org https://*.supabase.co wss://*.supabase.co; "
                "frame-ancestors 'none'"
            ),
        )
        response.headers["Content-Security-Policy"] = csp
        if os.getenv("ENABLE_HSTS", "").lower() in ("1", "true", "yes"):
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        for h in ("server", "x-powered-by"):
            if h in response.headers:
                del response.headers[h]
        return response
