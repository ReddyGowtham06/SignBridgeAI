"""
SignBridge AI - Translation History Model
Stores sign translation records without raw camera or video data.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base

if TYPE_CHECKING:
    from app.models.user import User


class TranslationHistory(Base):
    __tablename__ = "translation_history"

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
    source_language: Mapped[str] = mapped_column(
        String(20),
        default="ISL",
        nullable=False
    )
    translation_direction: Mapped[str] = mapped_column(
        String(30),
        default="sign_to_text",
        nullable=False,
        index=True
    )
    input_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    recognized_output: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True
    )
    recognition_path: Mapped[Optional[str]] = mapped_column(
        String(30),
        nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False
    )

    # Relationship
    user: Mapped["User"] = relationship("User", back_populates="history")

    def __repr__(self) -> str:
        return f"<TranslationHistory id={self.id} user_id={self.user_id} output={self.recognized_output[:20]}>"
