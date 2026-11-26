"""
SignBridge AI - Health & Root Endpoint Tests
"""
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Verifies root endpoint returns project identity and documentation URLs."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["project"] == "SIGNBRIDGE AI Backend"
    assert data["status"] == "online"
    assert "docs_url" in data
    assert "health_url" in data


def test_health_check_endpoint(client: TestClient):
    """Verifies health check endpoint returns database and AI model diagnostics."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "database" in data
    assert "ai_engine" in data
    assert data["ai_engine"]["loaded"] is True
    assert data["ai_engine"]["classes_count"] == 49
    assert data["ai_engine"]["sha256_verified"] is True
