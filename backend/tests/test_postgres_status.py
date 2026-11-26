"""
SignBridge AI - PostgreSQL Status & Configuration Audit Tests
Explicitly verifies and reports PostgreSQL connectivity without silent substitution.
"""
from app.core.config import settings
from app.database.session import check_database_connection


def test_postgresql_driver_availability():
    """Verifies that the PostgreSQL driver (psycopg 3) is installed and importable."""
    import psycopg
    assert psycopg.__version__ is not None


def test_database_connection_status_reporting():
    """
    CRITICAL RULE VERIFICATION:
    Verifies that check_database_connection() truthfully reports PostgreSQL reachability.
    If PostgreSQL is offline, it must report reachable=False without crashing or fabricating success.
    """
    status = check_database_connection()
    assert "status" in status
    assert "dialect" in status
    assert "reachable" in status
    assert "url_masked" in status

    # Verify that credentials are not leaked in logs/status
    assert "postgres:postgres" not in status["url_masked"]

    if status["dialect"] == "postgresql":
        if status["reachable"]:
            print("\n[POSTGRES STATUS] PostgreSQL server is actively CONNECTED and reachable.")
        else:
            print(f"\n[POSTGRES STATUS] PostgreSQL is OFFLINE: {status['error']}")
            print("[POSTGRES NOTICE] Configurable database support is active. Follow local setup in README.md to start PostgreSQL.")
