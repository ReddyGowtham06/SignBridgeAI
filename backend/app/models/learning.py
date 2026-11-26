"""
SignBridge AI - Learning Models
Defines curriculum lessons and user progress tracking.
"""
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from sqlalchemy import String, Text, Float, DateTime, ForeignKey, JSON, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base

if TYPE_CHECKING:
    from app.models.user import User


class LearningLesson(Base):
    __tablename__ = "learning_lessons"

    id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
        index=True
    )
    title: Mapped[str] = mapped_column(String(150), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty: Mapped[str] = mapped_column(
        String(20),
        default="beginner",
        nullable=False,
        index=True
    )
    category: Mapped[str] = mapped_column(
        String(50),
        default="General",
        nullable=False,
        index=True
    )
    content_metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships
    progress_records: Mapped[List["LearningProgress"]] = relationship(
        "LearningProgress",
        back_populates="lesson",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<LearningLesson id={self.id} title={self.title} difficulty={self.difficulty}>"


class LearningProgress(Base):
    __tablename__ = "learning_progress"
    __table_args__ = (
        UniqueConstraint("user_id", "lesson_id", name="uq_user_lesson_progress"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False
    )
    lesson_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("learning_lessons.id", ondelete="CASCADE"),
        index=True,
        nullable=False
    )
    completion_status: Mapped[str] = mapped_column(
        String(20),
        default="not_started",
        nullable=False,
        index=True
    )
    score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False
    )
    last_activity: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="progress")
    lesson: Mapped["LearningLesson"] = relationship("LearningLesson", back_populates="progress_records")

    def __repr__(self) -> str:
        return f"<LearningProgress user={self.user_id} lesson={self.lesson_id} status={self.completion_status} score={self.score}>"
