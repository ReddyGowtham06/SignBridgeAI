"""
SignBridge AI - Translation History Endpoints
Enforces strict user ownership on all records.
"""
from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, get_current_user
from app.models.user import User
from app.schemas.history import HistoryCreate, HistoryResponse, HistoryListResponse
from app.services.history_service import HistoryService

router = APIRouter()


@router.get(
    "",
    response_model=HistoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="List user translation history"
)
def list_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    items, total = HistoryService.list_history(db, current_user.id, skip=skip, limit=limit)
    return HistoryListResponse(
        items=[HistoryResponse.model_validate(r) for r in items],
        total=total
    )


@router.post(
    "",
    response_model=HistoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record a new sign translation event"
)
def create_history_record(
    req: HistoryCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    record = HistoryService.create_record(db, current_user.id, req)
    return HistoryResponse.model_validate(record)


@router.delete(
    "/{history_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a translation history record"
)
def delete_history_record(
    history_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    HistoryService.delete_record(db, current_user.id, history_id)
    return None
