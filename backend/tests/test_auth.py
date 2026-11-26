"""
SignBridge AI - Authentication & Authorization Tests
Tests registration, login, JWT issuance, password hashing, and token refresh.
"""
from fastapi.testclient import TestClient
from app.core.security import verify_password, hash_password


def test_password_hashing():
    """Verifies Argon2id password hashing and verification."""
    password = "MyComplexPassword@2026"
    hashed = hash_password(password)
    assert hashed.startswith("$argon2id$")
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_register_user_success(client: TestClient):
    """Verifies successful user registration."""
    res = client.post(
        "/api/auth/register",
        json={
            "name": "Dev Test",
            "email": "dev.test@signbridge.ai",
            "password": "StrongPassword99!"
        }
    )
    assert res.status_code == 201
    data = res.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "dev.test@signbridge.ai"
    assert data["user"]["name"] == "Dev Test"
    assert "hashed_password" not in data["user"]


def test_register_duplicate_email(client: TestClient, test_user):
    """Verifies 409 Conflict on duplicate email registration."""
    res = client.post(
        "/api/auth/register",
        json={
            "name": "Duplicate User",
            "email": test_user.email,
            "password": "ValidPassword123!"
        }
    )
    assert res.status_code == 409
    assert "already exists" in res.json()["detail"]


def test_login_success(client: TestClient, test_user):
    """Verifies successful login with valid credentials."""
    res = client.post(
        "/api/auth/login",
        json={"email": test_user.email, "password": "SecurePassword123!"}
    )
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["email"] == test_user.email


def test_login_invalid_password(client: TestClient, test_user):
    """Verifies 401 Unauthorized on invalid password."""
    res = client.post(
        "/api/auth/login",
        json={"email": test_user.email, "password": "WrongPassword123!"}
    )
    assert res.status_code == 401
    assert "Invalid email or password" in res.json()["detail"]


def test_login_nonexistent_email(client: TestClient):
    """Verifies 401 Unauthorized on unknown email."""
    res = client.post(
        "/api/auth/login",
        json={"email": "nobody@signbridge.ai", "password": "AnyPassword123!"}
    )
    assert res.status_code == 401


def test_auth_me_authenticated(client: TestClient, auth_headers: dict, test_user):
    """Verifies /api/auth/me returns current user profile."""
    res = client.get("/api/auth/me", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["email"] == test_user.email
    assert data["name"] == test_user.name
    assert data["id"] == test_user.id


def test_auth_me_unauthorized(client: TestClient):
    """Verifies 401 Unauthorized when Authorization header is missing."""
    res = client.get("/api/auth/me")
    assert res.status_code == 401


def test_auth_me_invalid_token(client: TestClient):
    """Verifies 401 Unauthorized on invalid JWT token."""
    res = client.get("/api/auth/me", headers={"Authorization": "Bearer invalid.jwt.token"})
    assert res.status_code == 401


def test_token_refresh(client: TestClient, test_user):
    """Verifies token refresh returns new access and refresh tokens."""
    login_res = client.post(
        "/api/auth/login",
        json={"email": test_user.email, "password": "SecurePassword123!"}
    )
    refresh_token = login_res.json()["refresh_token"]

    refresh_res = client.post(
        "/api/auth/refresh",
        json={"refresh_token": refresh_token}
    )
    assert refresh_res.status_code == 200
    data = refresh_res.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["email"] == test_user.email


def test_logout(client: TestClient, auth_headers: dict):
    """Verifies logout endpoint response."""
    res = client.post("/api/auth/logout", headers=auth_headers)
    assert res.status_code == 200
    assert "Successfully logged out" in res.json()["message"]
