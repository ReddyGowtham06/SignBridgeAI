"""
SignBridge AI - Authentication Service
Handles user registration, authentication, token issuance, and refresh.
"""
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token
)
from app.models.user import User
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse, RefreshRequest
from app.schemas.user import UserResponse


class AuthService:
    @staticmethod
    def register(db: Session, req: RegisterRequest) -> TokenResponse:
        """Registers a new user and returns JWT token pair."""
        # 1. Check if email already registered
        existing = db.query(User).filter(User.email == req.email.lower()).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists"
            )

        # 2. Hash password with Argon2
        hashed_pw = hash_password(req.password)

        # 3. Create user record
        user = User(
            name=req.name.strip(),
            email=req.email.lower().strip(),
            hashed_password=hashed_pw,
            status="active"
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        # 4. Issue tokens
        access_token = create_access_token(user.id, {"email": user.email, "name": user.name})
        refresh_token = create_refresh_token(user.id)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            user=UserResponse.model_validate(user)
        )

    @staticmethod
    def login(db: Session, req: LoginRequest) -> TokenResponse:
        """Authenticates user credentials and returns JWT token pair."""
        user = db.query(User).filter(User.email == req.email.lower().strip()).first()
        if not user or not verify_password(req.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if user.status != "active":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Account is {user.status}"
            )

        access_token = create_access_token(user.id, {"email": user.email, "name": user.name})
        refresh_token = create_refresh_token(user.id)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            user=UserResponse.model_validate(user)
        )

    @staticmethod
    def refresh(db: Session, req: RefreshRequest) -> TokenResponse:
        """Issues a new access and refresh token pair using a valid refresh token."""
        payload = decode_token(req.refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user_id = payload.get("sub")
        user = db.query(User).filter(User.id == user_id).first()
        if not user or user.status != "active":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or account inactive",
                headers={"WWW-Authenticate": "Bearer"},
            )

        new_access = create_access_token(user.id, {"email": user.email, "name": user.name})
        new_refresh = create_refresh_token(user.id)

        return TokenResponse(
            access_token=new_access,
            refresh_token=new_refresh,
            token_type="bearer",
            user=UserResponse.model_validate(user)
        )
