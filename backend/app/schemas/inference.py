"""
SignBridge AI - Inference Schemas
Validates 758-D MediaPipe landmark feature vectors and defines inference response payloads.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class InferenceRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description="758-dimensional MediaPipe landmark and kinematic feature vector"
    )
    motion_energy: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Wrist velocity motion energy metric (v)"
    )
    centroid_y: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.5,
        description="Hand signing-space Y centroid"
    )
    spatial_stability: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Hand spatial stability standard deviation (sigma)"
    )
    hand_presence: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Hand presence confidence ratio"
    )

    @field_validator("features")
    @classmethod
    def validate_features_length(cls, v: List[float]) -> List[float]:
        if len(v) != 758:
            raise ValueError(
                f"Feature vector must have exactly 758 dimensions, received {len(v)}"
            )
        return v


class TopPrediction(BaseModel):
    label: str
    class_index: int
    probability: float


class InferenceResponse(BaseModel):
    status: str = Field(..., description="'accepted' or 'rejected'")
    predicted_label: Optional[str] = Field(default=None)
    class_index: Optional[int] = Field(default=None)
    confidence: float
    margin: float
    recognition_path: str = Field(..., description="'dynamic_path', 'static_path', or 'none'")
    rejection_reason: Optional[str] = Field(default=None)
    top_predictions: List[TopPrediction]
    gating_metrics: Dict[str, Any]
