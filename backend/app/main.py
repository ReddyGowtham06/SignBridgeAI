"""
SignBridge AI - FastAPI Application Factory
Main entry point for the backend server.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import settings
from app.database.session import check_database_connection
from app.services.inference_service import get_inference_service_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("signbridge.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook executed during backend startup and shutdown."""
    logger.info("=" * 60)
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}]")
    
    # Check Database
    db_status = check_database_connection()
    logger.info(f"Database dialect: {db_status['dialect']}, reachable: {db_status['reachable']}")
    if not db_status["reachable"]:
        logger.warning(f"Database warning: {db_status['error']}")

    # Check AI Engine
    ai_status = get_inference_service_status()
    logger.info(
        f"AI Inference Engine loaded: {ai_status['loaded']}, "
        f"classes: {ai_status['classes_count']}, "
        f"SHA-256 verified: {ai_status['sha256_verified']}"
    )
    logger.info("=" * 60)
    yield
    logger.info("SignBridge AI backend shutdown complete.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Real-time Indian Sign Language (ISL) Communication Platform API",
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Returns standardized JSON error response for request validation failures."""
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        errors.append({
            "field": field,
            "message": err.get("msg", "Validation error"),
            "type": err.get("type", "value_error")
        })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": "Invalid request parameters",
            "errors": errors
        }
    )


@app.get("/", tags=["Root"])
def root():
    """Root status greeting endpoint."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs_url": "/api/docs",
        "health_url": "/api/health"
    }
