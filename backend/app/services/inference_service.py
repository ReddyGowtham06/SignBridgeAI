"""
SignBridge AI - M1 AI Inference Service
Implements exact forward pass and dual-path adaptive motion gating
matching the verified TypeScript production engine (signClassifier.ts).
"""
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.core.config import settings

logger = logging.getLogger("signbridge.ai")

TOTAL_FEATURE_DIM = 758
NUM_CLASSES = 49
STATIC_VOCABULARY = ["21.good", "34.stop", "45.hand"]

# Approved M1 Champion SHA-256 hash
APPROVED_M1_HASH = "c8068a3fedd85645d736d16ebde450bd442970784adba40fdacb8970a4535274"


class M1InferenceEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.M1_MODEL_PATH
        self.is_loaded = False
        self.sha256_hash: Optional[str] = None
        self.labels: List[str] = []
        self.W: Optional[np.ndarray] = None  # Shape (758, 49)
        self.b: Optional[np.ndarray] = None  # Shape (49,)
        self.scaler_mean: Optional[np.ndarray] = None  # Shape (758,)
        self.scaler_scale: Optional[np.ndarray] = None  # Shape (758,)
        self.load_model()

    def _locate_model_file(self) -> Optional[Path]:
        """Locates model file across standard project locations."""
        candidates = [
            Path(self.model_path),
            Path("frontend/public/models/sign_classifier.json"),
            Path("../frontend/public/models/sign_classifier.json"),
            Path("d:/SignBridgeAI/frontend/public/models/sign_classifier.json"),
            Path("ai-training/models/phase8b6b/phase8b6b_champion.json"),
            Path("../ai-training/models/phase8b6b/phase8b6b_champion.json"),
            Path("d:/SignBridgeAI/ai-training/models/phase8b6b/phase8b6b_champion.json"),
        ]
        for p in candidates:
            if p.exists() and p.is_file():
                return p.resolve()
        return None

    def load_model(self) -> None:
        """Loads and parses the M1 Champion weights and scaler parameters."""
        model_file = self._locate_model_file()
        if not model_file:
            logger.error(f"M1 model file not found at {self.model_path}")
            self.is_loaded = False
            return

        try:
            with open(model_file, "rb") as f:
                raw_bytes = f.read()
                self.sha256_hash = hashlib.sha256(raw_bytes).hexdigest()

            data = json.loads(raw_bytes.decode("utf-8"))

            self.labels = data["labels"]
            self.scaler_mean = np.array(data["scaler_mean"], dtype=np.float32)
            self.scaler_scale = np.array(data["scaler_scale"], dtype=np.float32)
            self.W = np.array(data["W"], dtype=np.float32)
            self.b = np.array(data["b"], dtype=np.float32)

            # Validations
            assert len(self.labels) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(self.labels)}"
            assert self.scaler_mean.shape == (TOTAL_FEATURE_DIM,), f"Mean shape {self.scaler_mean.shape}"
            assert self.scaler_scale.shape == (TOTAL_FEATURE_DIM,), f"Scale shape {self.scaler_scale.shape}"
            assert self.W.shape == (TOTAL_FEATURE_DIM, NUM_CLASSES), f"W shape {self.W.shape}"
            assert self.b.shape == (NUM_CLASSES,), f"b shape {self.b.shape}"

            self.is_loaded = True
            logger.info(
                f"M1 Model successfully loaded from {model_file} (SHA-256: {self.sha256_hash[:16]}...)"
            )
        except Exception as exc:
            logger.error(f"Failed to load M1 model: {exc}")
            self.is_loaded = False

    def classify(
        self,
        features: List[float],
        motion_energy: Optional[float] = None,
        centroid_y: Optional[float] = None,
        spatial_stability: Optional[float] = None,
        hand_presence: Optional[float] = 1.0,
        presence_threshold: float = 0.20,
        motion_threshold: float = 0.05,
        confidence_threshold: float = 0.45,
        static_min_presence: float = 0.50,
        static_max_pose_std: float = 0.055,
        static_max_y_centroid: float = 0.75,
        static_min_confidence: float = 0.50,
        static_min_margin: float = 0.10,
        enable_adaptive: bool = True
    ) -> Dict[str, Any]:
        """
        Executes forward inference and evaluates Dual-Path Adaptive Motion Gating.
        """
        if not self.is_loaded:
            raise RuntimeError("M1 inference model is not loaded")

        x = np.array(features, dtype=np.float32)
        if x.shape[0] != TOTAL_FEATURE_DIM:
            raise ValueError(f"Feature dimension must be {TOTAL_FEATURE_DIM}, got {x.shape[0]}")

        # 1. Feature Standardization: z = (x - mean) / scale
        x_scaled = (x - self.scaler_mean) / self.scaler_scale

        # 2. Linear Layer
        logits = np.dot(x_scaled, self.W) + self.b

        # 3. Stable Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        # 4. Top Predictions
        sorted_indices = np.argsort(probabilities)[::-1]
        best_idx = int(sorted_indices[0])
        best_score = float(probabilities[best_idx])
        best_label = self.labels[best_idx]

        top2_score = float(probabilities[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = float(best_score - top2_score)

        top_candidates = [
            {
                "class_index": int(idx),
                "label": self.labels[idx],
                "probability": float(probabilities[idx])
            }
            for idx in sorted_indices[:5]
        ]

        # Extract or default kinematic metrics
        # If motion_energy is not provided, estimate from feature velocity slices if available or default to dynamic
        m_energy = float(motion_energy) if motion_energy is not None else 0.10
        c_y = float(centroid_y) if centroid_y is not None else 0.50
        s_std = float(spatial_stability) if spatial_stability is not None else 0.02
        h_pres = float(hand_presence) if hand_presence is not None else 1.0

        gating_metrics = {
            "motion_energy": m_energy,
            "centroid_y": c_y,
            "spatial_stability": s_std,
            "hand_presence": h_pres,
            "margin": margin,
            "confidence": best_score,
            "thresholds": {
                "motion": motion_threshold,
                "confidence_dynamic": confidence_threshold,
                "presence": presence_threshold,
                "static_min_presence": static_min_presence,
                "static_max_pose_std": static_max_pose_std,
                "static_max_y_centroid": static_max_y_centroid,
                "static_min_confidence": static_min_confidence,
                "static_min_margin": static_min_margin,
            }
        }

        # Gate 0: Presence Check
        if h_pres < presence_threshold:
            return {
                "status": "rejected",
                "predicted_label": None,
                "class_index": None,
                "confidence": best_score,
                "margin": margin,
                "recognition_path": "none",
                "rejection_reason": "presence_too_low",
                "top_predictions": top_candidates,
                "gating_metrics": gating_metrics
            }

        # PATH A: Dynamic Signs (Normal Motion Energy >= motion_threshold)
        if m_energy >= motion_threshold:
            path = "dynamic_path"
            if best_score < confidence_threshold:
                return {
                    "status": "rejected",
                    "predicted_label": None,
                    "class_index": None,
                    "confidence": best_score,
                    "margin": margin,
                    "recognition_path": path,
                    "rejection_reason": "confidence_low",
                    "top_predictions": top_candidates,
                    "gating_metrics": gating_metrics
                }
            return {
                "status": "accepted",
                "predicted_label": best_label,
                "class_index": best_idx,
                "confidence": best_score,
                "margin": margin,
                "recognition_path": path,
                "rejection_reason": None,
                "top_predictions": top_candidates,
                "gating_metrics": gating_metrics
            }

        # PATH B: Low Motion (< motion_threshold)
        if not enable_adaptive:
            return {
                "status": "rejected",
                "predicted_label": None,
                "class_index": None,
                "confidence": best_score,
                "margin": margin,
                "recognition_path": "none",
                "rejection_reason": "motion_stationary",
                "top_predictions": top_candidates,
                "gating_metrics": gating_metrics
            }

        path = "static_path"

        # Filter B1: Active Hand Presence
        if h_pres < static_min_presence:
            return {
                "status": "rejected",
                "predicted_label": None,
                "class_index": None,
                "confidence": best_score,
                "margin": margin,
                "recognition_path": path,
                "rejection_reason": "static_presence_low",
                "top_predictions": top_candidates,
                "gating_metrics": gating_metrics
            }

        # Filter B2: Spatial Pose Stability (Anti-Jitter)
        if s_std > static_max_pose_std:
            return {
                "status": "rejected",
                "predicted_label": None,
                "class_index": None,
                "confidence": best_score,
                "margin": margin,
                "recognition_path": path,
                "rejection_reason": "static_pose_jitter",
                "top_predictions": top_candidates,
                "gating_metrics": gating_metrics
            }

        # Filter B3: Active Hand Elevation (Anti-Idle Desk/Lap)
        if c_y > static_max_y_centroid:
            return {
                "status": "rejected",
                "predicted_label": None,
                "class_index": None,
                "confidence": best_score,
                "margin": margin,
                "recognition_path": path,
                "rejection_reason": "idle_hand_resting",
                "top_predictions": top_candidates,
                "gating_metrics": gating_metrics
            }

        # Filter B4: Static Vocabulary Membership
        if best_label not in STATIC_VOCABULARY:
            return {
                "status": "rejected",
                "predicted_label": None,
                "class_index": None,
                "confidence": best_score,
                "margin": margin,
                "recognition_path": path,
                "rejection_reason": "motion_stationary",
                "top_predictions": top_candidates,
                "gating_metrics": gating_metrics
            }

        # Filter B5: Static Confidence & Probability Margin Filter
        if best_score < static_min_confidence or margin < static_min_margin:
            return {
                "status": "rejected",
                "predicted_label": None,
                "class_index": None,
                "confidence": best_score,
                "margin": margin,
                "recognition_path": path,
                "rejection_reason": "static_confidence_ambiguous",
                "top_predictions": top_candidates,
                "gating_metrics": gating_metrics
            }

        # All 5 static filters passed!
        return {
            "status": "accepted",
            "predicted_label": best_label,
            "class_index": best_idx,
            "confidence": best_score,
            "margin": margin,
            "recognition_path": path,
            "rejection_reason": None,
            "top_predictions": top_candidates,
            "gating_metrics": gating_metrics
        }


# Singleton engine instance
engine_instance = M1InferenceEngine()


def get_inference_service_status() -> Dict[str, Any]:
    """Returns status dictionary for health check."""
    return {
        "loaded": engine_instance.is_loaded,
        "classes_count": len(engine_instance.labels),
        "sha256_verified": (engine_instance.sha256_hash == APPROVED_M1_HASH) if engine_instance.sha256_hash else False,
        "sha256_hash": engine_instance.sha256_hash,
        "feature_dim": TOTAL_FEATURE_DIM,
        "static_vocabulary": STATIC_VOCABULARY
    }
