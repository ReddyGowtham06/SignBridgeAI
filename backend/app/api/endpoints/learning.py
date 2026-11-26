"""
SignBridge AI - Learning Studio Endpoints
"""
from typing import List
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, get_current_user
from app.models.user import User
from app.schemas.learning import (
    LessonResponse,
    ProgressCreate,
    ProgressResponse,
    LearningProgressSummary
)
from app.services.learning_service import LearningService

router = APIRouter()


@router.get(
    "/lessons",
    response_model=List[LessonResponse],
    status_code=status.HTTP_200_OK,
    summary="Retrieve all available ISL curriculum lessons"
)
def get_lessons(db: Session = Depends(get_db)):
    lessons = LearningService.list_lessons(db)
    return [LessonResponse.model_validate(l) for l in lessons]


@router.get(
    "/progress",
    response_model=LearningProgressSummary,
    status_code=status.HTTP_200_OK,
    summary="Get user learning progress and score summary"
)
def get_progress(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return LearningService.get_user_progress_summary(db, current_user.id)


@router.post(
    "/progress",
    response_model=ProgressResponse,
    status_code=status.HTTP_200_OK,
    summary="Record lesson score and completion status"
)
def record_progress(
    req: ProgressCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    record = LearningService.record_progress(db, current_user.id, req)
    return ProgressResponse.model_validate(record)
