"""
SignBridge AI - Users Endpoints
"""
from fastapi import APIRouter, Depends, status
from app.core.dependencies import get_current_user
from app.models.user import User
from app.schemas.user import UserResponse

router = APIRouter()


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve current user account profile"
)
def get_user_me(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)
