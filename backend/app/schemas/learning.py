"""
SignBridge AI - Learning Studio Schemas
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class LessonCreate(BaseModel):
    id: str = Field(..., max_length=50)
    title: str = Field(..., min_length=1, max_length=150)
    description: str = Field(..., min_length=1)
    difficulty: str = Field(default="beginner", max_length=20)
    category: str = Field(default="General", max_length=50)
    content_metadata: Dict[str, Any] = Field(default_factory=dict)


class LessonResponse(BaseModel):
    id: str
    title: str
    description: str
    difficulty: str
    category: str
    content_metadata: Dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProgressCreate(BaseModel):
    lesson_id: str = Field(..., max_length=50)
    completion_status: str = Field(default="completed", pattern="^(not_started|in_progress|completed)$")
    score: float = Field(default=100.0, ge=0.0, le=100.0)


class ProgressResponse(BaseModel):
    id: str
    user_id: str
    lesson_id: str
    completion_status: str
    score: float
    last_activity: datetime
    lesson: Optional[LessonResponse] = None

    model_config = ConfigDict(from_attributes=True)


class LearningProgressSummary(BaseModel):
    total_lessons: int
    completed_lessons: int
    in_progress_lessons: int
    average_score: float
    progress_records: List[ProgressResponse]
