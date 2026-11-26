"""
SignBridge AI - API Router Assembler
Mounts all REST endpoints under /api.
"""
from fastapi import APIRouter

from app.api.endpoints import health, auth, users, history, learning, inference

api_router = APIRouter()

# 1. Health
api_router.include_router(health.router, prefix="", tags=["Health"])

# 2. Authentication
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])

# 3. Users
api_router.include_router(users.router, prefix="/users", tags=["Users"])

# 4. Translation History
api_router.include_router(history.router, prefix="/history", tags=["Translation History"])

# 5. Learning Studio
api_router.include_router(learning.router, prefix="/learning", tags=["Learning Studio"])

# 6. AI Inference
api_router.include_router(inference.router, prefix="/inference", tags=["AI Inference"])
