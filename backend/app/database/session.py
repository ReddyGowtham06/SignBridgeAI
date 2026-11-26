"""
SignBridge AI - Database Session and Engine Management
Manages PostgreSQL connections via SQLAlchemy 2.0 with configurable dialect support.
"""
import logging
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError

from app.core.config import settings

logger = logging.getLogger("signbridge.database")

# Build engine based on database URL
# If PostgreSQL is not reachable, fallback handling allows the app to start or test gracefully
_connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    _connect_args["check_same_thread"] = False
elif "psycopg" in settings.DATABASE_URL:
    _connect_args["connect_timeout"] = settings.DATABASE_CONNECT_TIMEOUT

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,
    echo=(settings.ENVIRONMENT == "development_debug"),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def check_database_connection() -> dict:
    """
    Checks database connectivity.
    Returns status dict with dialect, reachability, and details.
    """
    dialect_name = engine.dialect.name
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            return {
                "status": "connected",
                "dialect": dialect_name,
                "reachable": True,
                "url_masked": _mask_db_url(settings.DATABASE_URL),
                "error": None
            }
    except OperationalError as exc:
        logger.warning(f"Database connection failed: {exc}")
        return {
            "status": "disconnected",
            "dialect": dialect_name,
            "reachable": False,
            "url_masked": _mask_db_url(settings.DATABASE_URL),
            "error": str(exc.orig) if hasattr(exc, "orig") else str(exc)
        }
    except Exception as exc:
        logger.error(f"Unexpected database error: {exc}")
        return {
            "status": "error",
            "dialect": dialect_name,
            "reachable": False,
            "url_masked": _mask_db_url(settings.DATABASE_URL),
            "error": str(exc)
        }


def _mask_db_url(url: str) -> str:
    """Masks credentials in database URL for safe logging."""
    if "@" in url:
        prefix, rest = url.split("@", 1)
        if "://" in prefix:
            proto, _ = prefix.split("://", 1)
            return f"{proto}://*****:*****@{rest}"
    return url


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
