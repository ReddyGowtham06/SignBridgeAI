"""
SignBridge AI - Test Suite Fixtures and Configuration
Configures isolated test database, FastAPI TestClient, and authentication fixtures.
"""
import os
import sys
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Ensure backend root is on sys.path
backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from app.main import app
from app.core.config import settings
from app.core.security import hash_password
from app.database.base import Base
from app.database.session import get_db
from app.models.user import User
from app.services.learning_service import LearningService

# In-memory SQLite engine for fast, isolated unit and integration tests
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """Creates fresh in-memory database schema for each test."""
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        # Seed default curriculum lessons
        LearningService.ensure_default_lessons(session)
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def configure_test_env():
    """Configures test settings and short timeouts."""
    settings.DATABASE_CONNECT_TIMEOUT = 1


@pytest.fixture(scope="function")
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """Provides a TestClient with get_db overridden to use test database session."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app, raise_server_exceptions=True)
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db_session: Session) -> User:
    """Creates a standard test user."""
    user = User(
        name="Aarav Sharma",
        email="aarav@signbridge.ai",
        hashed_password=hash_password("SecurePassword123!"),
        status="active"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(client: TestClient, test_user: User) -> dict:
    """Logs in test_user and returns Authorization Bearer header dict."""
    response = client.post(
        "/api/auth/login",
        json={"email": "aarav@signbridge.ai", "password": "SecurePassword123!"}
    )
    assert response.status_code == 200, f"Login failed: {response.text}"
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def second_user(db_session: Session) -> User:
    """Creates a second test user for testing cross-user data isolation."""
    user = User(
        name="Priya Patel",
        email="priya@signbridge.ai",
        hashed_password=hash_password("AnotherSecurePw456!"),
        status="active"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def second_auth_headers(client: TestClient, second_user: User) -> dict:
    """Logs in second_user and returns Authorization Bearer header dict."""
    response = client.post(
        "/api/auth/login",
        json={"email": "priya@signbridge.ai", "password": "AnotherSecurePw456!"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
