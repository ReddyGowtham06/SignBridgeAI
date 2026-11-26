"""
SignBridge AI - Learning Studio Service
Curriculum lesson management and user score / progress tracking.
"""
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.learning import LearningLesson, LearningProgress
from app.schemas.learning import ProgressCreate, LearningProgressSummary, ProgressResponse


# Standard default curriculum lessons covering Phase 9A vocabulary & Indian Sign Language fundamentals
DEFAULT_LESSONS = [
    {
        "id": "lesson-01-greetings",
        "title": "ISL Greetings & Courtesies",
        "description": "Learn fundamental everyday Indian Sign Language greetings including Hello, Namaste, and Good Morning.",
        "difficulty": "beginner",
        "category": "Greetings",
        "content_metadata": {
            "vocabulary": ["9.hello", "7.goodmorning", "namaste"],
            "target_motion": "dynamic",
            "estimated_minutes": 15
        }
    },
    {
        "id": "lesson-02-static-foundations",
        "title": "Static Precision & Pose Alignment",
        "description": "Master clean static signing poses for Good, Stop, and Hand using proper posture and spatial stillness.",
        "difficulty": "beginner",
        "category": "Foundations",
        "content_metadata": {
            "vocabulary": ["21.good", "34.stop", "45.hand"],
            "target_motion": "static",
            "estimated_minutes": 20
        }
    },
    {
        "id": "lesson-03-body-parts",
        "title": "Facial & Anatomical Signs",
        "description": "Recognize and practice anatomical ISL signs including Nose, Ear, Eyes, and Teeth.",
        "difficulty": "intermediate",
        "category": "Anatomy",
        "content_metadata": {
            "vocabulary": ["29.nose", "ear", "eyes", "teeth"],
            "target_motion": "dynamic",
            "estimated_minutes": 25
        }
    },
    {
        "id": "lesson-04-emergency",
        "title": "Essential & Emergency Signs",
        "description": "Crucial signs for safety, asking for assistance, and emergency communication.",
        "difficulty": "intermediate",
        "category": "Emergency",
        "content_metadata": {
            "vocabulary": ["help", "doctor", "police", "danger"],
            "target_motion": "mixed",
            "estimated_minutes": 30
        }
    }
]


class LearningService:
    @staticmethod
    def ensure_default_lessons(db: Session) -> None:
        """Seeds default curriculum lessons if table is empty."""
        count = db.query(LearningLesson).count()
        if count == 0:
            for item in DEFAULT_LESSONS:
                lesson = LearningLesson(
                    id=item["id"],
                    title=item["title"],
                    description=item["description"],
                    difficulty=item["difficulty"],
                    category=item["category"],
                    content_metadata=item["content_metadata"]
                )
                db.add(lesson)
            db.commit()

    @staticmethod
    def list_lessons(db: Session) -> List[LearningLesson]:
        """Returns all available curriculum lessons."""
        LearningService.ensure_default_lessons(db)
        return db.query(LearningLesson).order_by(LearningLesson.id.asc()).all()

    @staticmethod
    def get_user_progress_summary(db: Session, user_id: str) -> LearningProgressSummary:
        """Computes curriculum progress summary and score metrics for a user."""
        LearningService.ensure_default_lessons(db)
        lessons = db.query(LearningLesson).all()
        records = db.query(LearningProgress).filter(LearningProgress.user_id == user_id).all()
        
        progress_map = {r.lesson_id: r for r in records}
        completed = sum(1 for r in records if r.completion_status == "completed")
        in_progress = sum(1 for r in records if r.completion_status == "in_progress")
        avg_score = (sum(r.score for r in records) / len(records)) if records else 0.0

        progress_responses = [
            ProgressResponse(
                id=r.id,
                user_id=r.user_id,
                lesson_id=r.lesson_id,
                completion_status=r.completion_status,
                score=r.score,
                last_activity=r.last_activity,
            )
            for r in records
        ]

        return LearningProgressSummary(
            total_lessons=len(lessons),
            completed_lessons=completed,
            in_progress_lessons=in_progress,
            average_score=round(avg_score, 1),
            progress_records=progress_responses
        )

    @staticmethod
    def record_progress(
        db: Session,
        user_id: str,
        req: ProgressCreate
    ) -> LearningProgress:
        """Records or updates user completion progress for a lesson."""
        lesson = db.query(LearningLesson).filter(LearningLesson.id == req.lesson_id).first()
        if not lesson:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lesson '{req.lesson_id}' does not exist"
            )

        existing = db.query(LearningProgress).filter(
            LearningProgress.user_id == user_id,
            LearningProgress.lesson_id == req.lesson_id
        ).first()

        now = datetime.now(timezone.utc)
        if existing:
            existing.completion_status = req.completion_status
            existing.score = max(existing.score, req.score)
            existing.last_activity = now
            db.commit()
            db.refresh(existing)
            return existing
        else:
            progress = LearningProgress(
                user_id=user_id,
                lesson_id=req.lesson_id,
                completion_status=req.completion_status,
                score=req.score,
                last_activity=now
            )
            db.add(progress)
            db.commit()
            db.refresh(progress)
            return progress
