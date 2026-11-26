"""
SignBridge AI - Health Check Endpoint
Reports backend service health, database connection state, and AI model readiness.
"""
from fastapi import APIRouter
from app.core.config import settings
from app.database.session import check_database_connection
from app.services.inference_service import get_inference_service_status

router = APIRouter()


@router.get("/health", tags=["Health"])
def health_check():
    """
    Comprehensive system health diagnostic.
    Verifies API server, database connectivity, and M1 AI inference engine.
    """
    db_status = check_database_connection()
    ai_status = get_inference_service_status()

    is_healthy = True
    # If database is configured and reachable, or running in local dev fallback
    if db_status["dialect"] == "postgresql" and not db_status["reachable"]:
        # PostgreSQL is specified in config but connection failed
        is_healthy = False

    return {
        "status": "healthy" if is_healthy else "degraded",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "database": db_status,
        "ai_engine": ai_status
    }
