"""
SignBridge AI - Production CORS Configuration Tests
Verifies that the deployed Vercel frontend origin and local dev origins are allowed,
while arbitrary untrusted origins are rejected.
"""
from fastapi.testclient import TestClient
from app.core.config import settings


def test_cors_origins_configuration():
    """Verifies exact production Vercel origin and local dev origins are configured."""
    assert "https://sign-bridge-ai-ecru.vercel.app" in settings.CORS_ORIGINS
    assert "http://localhost:5173" in settings.CORS_ORIGINS
    assert "*" not in settings.CORS_ORIGINS


def test_cors_preflight_allows_production_vercel_origin(client: TestClient):
    """Verifies OPTIONS preflight succeeds for https://sign-bridge-ai-ecru.vercel.app."""
    res = client.options(
        "/api/auth/login",
        headers={
            "Origin": "https://sign-bridge-ai-ecru.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type,Authorization",
        },
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "https://sign-bridge-ai-ecru.vercel.app"
    assert res.headers.get("access-control-allow-credentials") == "true"


def test_cors_preflight_rejects_untrusted_origin(client: TestClient):
    """Verifies OPTIONS preflight rejects arbitrary unknown origins."""
    res = client.options(
        "/api/auth/login",
        headers={
            "Origin": "https://evil-untrusted-site.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert res.status_code == 400
    assert "access-control-allow-origin" not in res.headers
