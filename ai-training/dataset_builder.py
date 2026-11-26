r"""
SIGNBRIDGE AI — Dataset Builder Module
========================================
Aggregates individual raw video landmark (.npz) files into consolidated,
consistently shaped feature matrices for model training. Generates label mappings
and feature normalization parameters without data leakage.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required for dataset_builder.py.")
    sys.exit(1)

import config


def compute_video_feature_vector(
    landmarks: np.ndarray,
    presence_mask: np.ndarray,
    timestamps_ms: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes a fixed-size, discriminative feature vector from a dynamic landmark sequence.
    Captures geometry, variance, spatial bounds, motion deltas, and dual-hand presence.
    Enforces temporal adjacency for velocity calculations, avoiding spurious jumps across missing frames.

    Args:
        landmarks (np.ndarray): Shape (N, 2, 21, 3)
        presence_mask (np.ndarray): Shape (N, 2)
        timestamps_ms (Optional[np.ndarray]): Shape (N,) integer millisecond timestamps.

    Returns:
        np.ndarray: 1D feature vector of shape (758,)
    """
    total_frames = landmarks.shape[0]
    features: List[float] = []

    # Process Hand Slot 0 (Left) and Hand Slot 1 (Right) independently
    for slot_idx in range(config.NUM_HANDS):
        slot_mask = presence_mask[:, slot_idx]
        slot_coords = landmarks[:, slot_idx, :, :]  # (N, 21, 3)
        slot_flat = slot_coords.reshape(total_frames, -1)  # (N, 63)

        valid_frames_count = np.sum(slot_mask)
        presence_ratio = valid_frames_count / total_frames if total_frames > 0 else 0.0
        features.append(float(presence_ratio))

        if valid_frames_count > 0:
            active_coords = slot_flat[slot_mask]  # (K, 63)

            # Spatial Statistics: Mean, Std, Min, Max (over active frames)
            mean_vals = np.mean(active_coords, axis=0)
            std_vals = np.std(active_coords, axis=0)
            min_vals = np.min(active_coords, axis=0)
            max_vals = np.max(active_coords, axis=0)

            # Temporal Velocity: Calculate differences ONLY across adjacent valid frame pairs
            # consec_mask is True at index i if hand is detected at both frame i and frame i+1
            if total_frames > 1:
                consec_mask = slot_mask[:-1] & slot_mask[1:]
            else:
                consec_mask = np.zeros(0, dtype=bool)

            num_consecutive = int(np.sum(consec_mask))

            if num_consecutive > 0:
                # Coordinate differences across strictly adjacent frames
                delta_coords = slot_flat[1:][consec_mask] - slot_flat[:-1][consec_mask]  # (M, 63)

                if timestamps_ms is not None and len(timestamps_ms) == total_frames:
                    # Time difference in seconds for each valid consecutive pair
                    dt_raw = (timestamps_ms[1:][consec_mask] - timestamps_ms[:-1][consec_mask]).astype(np.float32)
                    # Guard against non-positive intervals; clamp minimum dt to 1 ms (0.001 s)
                    dt_sec = np.maximum(dt_raw / 1000.0, 0.001)  # (M,)
                    vel_deltas = delta_coords / dt_sec[:, np.newaxis]  # (M, 63) in units/sec
                else:
                    # Fallback to nominal 30 FPS if timestamps not provided (dt = 1/30 s)
                    vel_deltas = delta_coords * 30.0

                vel_mean = np.mean(np.abs(vel_deltas), axis=0)
                vel_std = np.std(vel_deltas, axis=0)
            else:
                # No valid consecutive detections exist (e.g. isolated single-frame detections)
                vel_mean = np.zeros(63, dtype=np.float32)
                vel_std = np.zeros(63, dtype=np.float32)

            features.extend(mean_vals.tolist())
            features.extend(std_vals.tolist())
            features.extend(min_vals.tolist())
            features.extend(max_vals.tolist())
            features.extend(vel_mean.tolist())
            features.extend(vel_std.tolist())
        else:
            # When hand is completely absent, fill slot stats with 0.0
            # 63 coords * (4 stats + 2 velocities) = 378 zeros
            features.extend([0.0] * 378)

    feature_array = np.array(features, dtype=np.float32)
    return feature_array


def build_dataset(
    raw_landmarks_dir: Path = config.RAW_LANDMARKS_DIR,
    output_dir: Path = config.PROCESSED_DATA_DIR,
) -> Dict[str, Any]:
    """
    Scans raw extracted .npz files, constructs tabular feature matrices,
    and encodes sign class labels.

    Returns:
        dict: Summary of built dataset arrays and label mappings.
    """
    if not raw_landmarks_dir.exists():
        raise FileNotFoundError(f"Raw landmarks directory not found at: {raw_landmarks_dir}")

    # Discover all class folders with extracted .npz files
    class_folders = [d for d in raw_landmarks_dir.iterdir() if d.is_dir()]
    class_folders.sort(key=lambda d: d.name)

    if not class_folders:
        raise ValueError(
            f"No extracted class folders found in {raw_landmarks_dir}. "
            "Please run landmark_extractor.py first."
        )

    # Build Label Encoders
    label_to_id = {folder.name: idx for idx, folder in enumerate(class_folders)}
    id_to_label = {idx: folder.name for idx, folder in enumerate(class_folders)}

    feature_list = []
    label_list = []
    video_name_list = []
    class_name_list = []

    total_samples = 0
    print("=" * 72)
    print("              SIGNBRIDGE AI — DATASET FEATURE BUILDER")
    print("=" * 72)
    print(f"Source Directory : {raw_landmarks_dir}")
    print(f"Output Directory : {output_dir}")
    print(f"Classes Found    : {len(class_folders)}")
    print("=" * 72 + "\n")

    for folder in class_folders:
        c_name = folder.name
        class_id = label_to_id[c_name]
        npz_files = list(folder.glob("*.npz"))

        for npz_path in npz_files:
            try:
                with np.load(npz_path) as data:
                    landmarks = data["landmarks"]          # (N, 2, 21, 3)
                    presence_mask = data["presence_mask"]  # (N, 2)
                    timestamps_ms = data["timestamps_ms"] if "timestamps_ms" in data else None
                    video_name = str(data["video_name"]) if "video_name" in data else npz_path.stem

                feat_vec = compute_video_feature_vector(landmarks, presence_mask, timestamps_ms)
                feature_list.append(feat_vec)
                label_list.append(class_id)
                video_name_list.append(video_name)
                class_name_list.append(c_name)
                total_samples += 1
            except Exception as err:
                print(f"  [WARNING] Skipping corrupt file {npz_path.name}: {err}")

    X = np.array(feature_list, dtype=np.float32)
    y = np.array(label_list, dtype=np.int64)
    video_names = np.array(video_name_list, dtype="<U64")
    class_names = np.array(class_name_list, dtype="<U64")

    # Save processed tabular dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    tabular_output_path = output_dir / "features_tabular.npz"
    np.savez_compressed(
        tabular_output_path,
        X=X,
        y=y,
        video_names=video_names,
        class_names=class_names,
    )

    # Save label mapping JSON
    label_mapping_path = output_dir / "label_mapping.json"
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "num_classes": len(class_folders),
                "feature_dim": X.shape[1],
                "total_samples": total_samples,
            },
            f,
            indent=2,
        )

    print("Dataset Building Complete!")
    print(f"  • Feature Matrix (X)  : Shape {X.shape} ({X.dtype})")
    print(f"  • Labels Vector (y)   : Shape {y.shape} ({y.dtype})")
    print(f"  • Total Sign Classes  : {len(class_folders)}")
    print(f"  • Processed File      : {tabular_output_path}")
    print(f"  • Label Mapping File  : {label_mapping_path}")

    return {
        "samples": total_samples,
        "feature_dim": X.shape[1],
        "num_classes": len(class_folders),
        "output_file": str(tabular_output_path),
    }


if __name__ == "__main__":
    build_dataset()

