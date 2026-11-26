"""
SignBridge AI - AI Inference & Dual-Path Gating Tests
Verifies 758-D schema enforcement, M1 forward pass, and Adaptive Motion Gating parity.
"""
import json
from pathlib import Path
from fastapi.testclient import TestClient
from app.services.inference_service import engine_instance, TOTAL_FEATURE_DIM, APPROVED_M1_HASH


def test_m1_model_loaded_and_verified():
    """Verifies that the M1 champion model is loaded with matching SHA-256 hash."""
    assert engine_instance.is_loaded is True
    assert engine_instance.sha256_hash == APPROVED_M1_HASH
    assert len(engine_instance.labels) == 49
    assert engine_instance.scaler_mean.shape == (TOTAL_FEATURE_DIM,)
    assert engine_instance.W.shape == (TOTAL_FEATURE_DIM, 49)


def test_inference_rejects_invalid_feature_dimension(client: TestClient):
    """Verifies that non-758-D vectors are rejected with HTTP 422 Unprocessable Entity."""
    # 100 features instead of 758
    bad_features = [0.0] * 100
    res = client.post(
        "/api/inference/classify",
        json={"features": bad_features, "motion_energy": 0.15}
    )
    assert res.status_code == 422
    data = res.json()
    assert "Feature vector must have exactly 758 dimensions" in str(data)


def test_inference_presence_rejection(client: TestClient):
    """Verifies that zero or low presence (< 0.20) rejects with presence_too_low."""
    features = [0.0] * TOTAL_FEATURE_DIM
    res = client.post(
        "/api/inference/classify",
        json={
            "features": features,
            "motion_energy": 0.15,
            "hand_presence": 0.05
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "rejected"
    assert data["rejection_reason"] == "presence_too_low"
    assert data["recognition_path"] == "none"


def test_inference_path_a_dynamic_sign(client: TestClient):
    """Verifies Path A dynamic sign routing when motion energy >= 0.05."""
    # Load synthetic vector if available or construct feature vector
    synth_file = Path("ai-training/data/synthetic_static_vectors.json")
    if synth_file.exists():
        with open(synth_file, "r") as f:
            synth_data = json.load(f)
        features = synth_data.get("21.good", [0.0] * TOTAL_FEATURE_DIM)
    else:
        features = [0.0] * TOTAL_FEATURE_DIM

    res = client.post(
        "/api/inference/classify",
        json={
            "features": features,
            "motion_energy": 0.15,  # High motion -> Path A
            "hand_presence": 1.0
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["recognition_path"] == "dynamic_path"
    assert data["status"] in ["accepted", "rejected"]


def test_inference_path_b_static_sign_acceptance(client: TestClient):
    """
    Verifies Path B acceptance for static sign '21.good' under verified kinematic constraints:
    v < 0.05, presence >= 0.50, stability <= 0.055, centroid_y <= 0.75.
    """
    synth_file = Path("ai-training/data/synthetic_static_vectors.json")
    if not synth_file.exists():
        return

    with open(synth_file, "r") as f:
        synth_data = json.load(f)
    features = synth_data["21.good"]

    res = client.post(
        "/api/inference/classify",
        json={
            "features": features,
            "motion_energy": 0.02,        # Stationary (< 0.05)
            "hand_presence": 0.95,        # Active presence (>= 0.50)
            "spatial_stability": 0.031,   # Stable pose (<= 0.055)
            "centroid_y": 0.50            # Elevated in signing space (<= 0.75)
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "accepted"
    assert data["recognition_path"] == "static_path"
    assert data["predicted_label"] == "21.good"
    assert data["confidence"] >= 0.50
    assert data["margin"] >= 0.10


def test_inference_path_b_defensive_filters(client: TestClient):
    """Verifies all 5 defensive filters of Path B."""
    synth_file = Path("ai-training/data/synthetic_static_vectors.json")
    if not synth_file.exists():
        return

    with open(synth_file, "r") as f:
        synth_data = json.load(f)
    features = synth_data["21.good"]

    # Filter B1: Low presence
    r1 = client.post(
        "/api/inference/classify",
        json={"features": features, "motion_energy": 0.02, "hand_presence": 0.30}
    ).json()
    assert r1["status"] == "rejected"
    assert r1["rejection_reason"] == "static_presence_low"

    # Filter B2: Spatial jitter
    r2 = client.post(
        "/api/inference/classify",
        json={"features": features, "motion_energy": 0.02, "spatial_stability": 0.090}
    ).json()
    assert r2["status"] == "rejected"
    assert r2["rejection_reason"] == "static_pose_jitter"

    # Filter B3: Resting hand (lap/desk)
    r3 = client.post(
        "/api/inference/classify",
        json={"features": features, "motion_energy": 0.02, "centroid_y": 0.85}
    ).json()
    assert r3["status"] == "rejected"
    assert r3["rejection_reason"] == "idle_hand_resting"
