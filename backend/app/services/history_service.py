"""
SignBridge AI - Translation History Service
Manages user translation records with strict user isolation and ownership checks.
"""
from typing import List, Tuple
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.history import TranslationHistory
from app.schemas.history import HistoryCreate


class HistoryService:
    @staticmethod
    def list_history(
        db: Session,
        user_id: str,
        skip: int = 0,
        limit: int = 50
    ) -> Tuple[List[TranslationHistory], int]:
        """Retrieves paginated translation history records strictly belonging to user."""
        query = db.query(TranslationHistory).filter(TranslationHistory.user_id == user_id)
        total = query.count()
        records = (
            query.order_by(TranslationHistory.created_at.desc())
            .offset(skip)
            .limit(min(limit, 100))
            .all()
        )
        return records, total

    @staticmethod
    def create_record(
        db: Session,
        user_id: str,
        req: HistoryCreate
    ) -> TranslationHistory:
        """Creates a new translation history record for user."""
        record = TranslationHistory(
            user_id=user_id,
            source_language=req.source_language,
            translation_direction=req.translation_direction,
            input_text=req.input_text,
            recognized_output=req.recognized_output,
            confidence=req.confidence,
            recognition_path=req.recognition_path
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def delete_record(
        db: Session,
        user_id: str,
        history_id: str
    ) -> None:
        """Deletes a history record, verifying user ownership first."""
        record = db.query(TranslationHistory).filter(
            TranslationHistory.id == history_id,
            TranslationHistory.user_id == user_id
        ).first()

        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Translation history record not found or does not belong to you"
            )

        db.delete(record)
        db.commit()
