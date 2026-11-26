"""
SignBridge AI - Translation History Schemas
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class HistoryBase(BaseModel):
    source_language: str = Field(default="ISL", max_length=20)
    translation_direction: str = Field(default="sign_to_text", max_length=30)
    input_text: Optional[str] = Field(default=None)
    recognized_output: str = Field(..., min_length=1)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    recognition_path: Optional[str] = Field(default=None, max_length=30)


class HistoryCreate(HistoryBase):
    pass


class HistoryResponse(HistoryBase):
    id: str
    user_id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class HistoryListResponse(BaseModel):
    items: List[HistoryResponse]
    total: int
