"""
SignBridge AI - AI Inference API Endpoints
Provides server-side sign classification using the M1 Champion linear model.
"""
from fastapi import APIRouter, HTTPException, status
from app.schemas.inference import InferenceRequest, InferenceResponse
from app.services.inference_service import engine_instance, get_inference_service_status

router = APIRouter()


@router.post(
    "/classify",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a 758-dimensional MediaPipe landmark feature vector"
)
def classify_sign(req: InferenceRequest):
    """
    Performs sign classification on a 758-D normalized landmark feature vector.
    Enforces dual-path adaptive motion gating (Path A Dynamic vs Path B Static).
    """
    if not engine_instance.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI inference engine is not ready or model weights failed to load"
        )

    try:
        result = engine_instance.classify(
            features=req.features,
            motion_energy=req.motion_energy,
            centroid_y=req.centroid_y,
            spatial_stability=req.spatial_stability,
            hand_presence=req.hand_presence
        )
        return InferenceResponse(**result)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}"
        )


@router.get(
    "/model-info",
    status_code=status.HTTP_200_OK,
    summary="Retrieve AI inference engine metadata and verification status"
)
def get_model_info():
    """Returns loaded model architecture, class list, and SHA-256 verification."""
    return get_inference_service_status()
