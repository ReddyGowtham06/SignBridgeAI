"""
SignBridge AI - Authentication Endpoints
"""
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, get_current_user
from app.middleware.rate_limit import rate_limit_auth
from app.models.user import User
from app.schemas.auth import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    LogoutResponse
)
from app.schemas.user import UserResponse
from app.services.auth_service import AuthService

router = APIRouter()


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit_auth)],
    summary="Register a new user account"
)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    return AuthService.register(db, req)


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limit_auth)],
    summary="Authenticate user and obtain JWT tokens"
)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    return AuthService.login(db, req)


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get authenticated user profile"
)
def get_auth_me(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Refresh access token using refresh token"
)
def refresh_token(req: RefreshRequest, db: Session = Depends(get_db)):
    return AuthService.refresh(db, req)


@router.post(
    "/logout",
    response_model=LogoutResponse,
    status_code=status.HTTP_200_OK,
    summary="Log out and invalidate client session"
)
def logout(current_user: User = Depends(get_current_user)):
    # Standard client token discard pattern; extensible for Redis blacklist in Phase 10B
    return LogoutResponse(message="Successfully logged out")
