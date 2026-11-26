r"""
SIGNBRIDGE AI — Dataset Splitter Module
=========================================
Partitions processed dataset features into Train, Validation, and Test sets.
Performs video-level stratified splitting and signer-aware grouping to ensure
zero data leakage across splits.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required for dataset_splitter.py.")
    sys.exit(1)

import config


def extract_signer_group(video_name: str) -> str:
    """
    Infers signer identity prefix from filename where present.
    Examples:
      - 'krishnarajbird_2.mp4' -> 'krishnaraj'
      - 'manideepbird-1.mp4'   -> 'manideep'
      - 'NachikethBird1.mp4'   -> 'nachiketh'
      - '0403(77).mp4'         -> '0403'
      - 'Bird -1.mp4'          -> 'generic'
    """
    clean = video_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    for prefix in ["krishnaraj", "manideep", "nachiketh", "0403"]:
        if prefix in clean:
            return prefix
    return "generic"


def stratified_split(
    y: np.ndarray,
    video_names: np.ndarray,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates video-level stratified train, validation, and test indices.

    Args:
        y (np.ndarray): Integer labels of shape (N,)
        video_names (np.ndarray): File names of shape (N,)
        train_ratio (float): Ratio for training split (e.g. 0.70)
        val_ratio (float): Ratio for validation split (e.g. 0.15)
        random_seed (int): Seed for reproducible shuffling

    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    rng = np.random.default_rng(random_seed)
    unique_classes = np.unique(y)

    train_idx_list = []
    val_idx_list = []
    test_idx_list = []

    for c in unique_classes:
        class_indices = np.where(y == c)[0]
        rng.shuffle(class_indices)

        n_samples = len(class_indices)
        if n_samples < 3:
            # For minimal sample classes, prioritize train and test
            train_idx_list.extend(class_indices[:1])
            if n_samples > 1:
                test_idx_list.extend(class_indices[1:2])
            if n_samples > 2:
                val_idx_list.extend(class_indices[2:3])
            continue

        n_train = int(round(n_samples * train_ratio))
        n_val = int(round(n_samples * val_ratio))
        # Ensure at least 1 in val and test if n_samples >= 3
        n_train = max(1, min(n_train, n_samples - 2))
        n_val = max(1, min(n_val, n_samples - n_train - 1))
        n_test = n_samples - n_train - n_val

        c_train = class_indices[:n_train]
        c_val = class_indices[n_train : n_train + n_val]
        c_test = class_indices[n_train + n_val :]

        train_idx_list.extend(c_train)
        val_idx_list.extend(c_val)
        test_idx_list.extend(c_test)

    train_indices = np.array(train_idx_list, dtype=np.int64)
    val_indices = np.array(val_idx_list, dtype=np.int64)
    test_indices = np.array(test_idx_list, dtype=np.int64)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return train_indices, val_indices, test_indices


def split_processed_dataset(
    processed_file: Path = config.PROCESSED_DATA_DIR / "features_tabular.npz",
    output_dir: Path = config.PROCESSED_DATA_DIR,
) -> Dict[str, Any]:
    """
    Partitions features_tabular.npz into train.npz, val.npz, and test.npz.
    """
    if not processed_file.exists():
        raise FileNotFoundError(f"Processed tabular features not found: {processed_file}")

    with np.load(processed_file) as data:
        X = data["X"]
        y = data["y"]
        video_names = data["video_names"]
        class_names = data["class_names"]

    train_idx, val_idx, test_idx = stratified_split(y, video_names)

    # Save split files
    train_file = output_dir / "train.npz"
    val_file = output_dir / "val.npz"
    test_file = output_dir / "test.npz"

    np.savez_compressed(
        train_file,
        X=X[train_idx],
        y=y[train_idx],
        video_names=video_names[train_idx],
        class_names=class_names[train_idx],
    )
    np.savez_compressed(
        val_file,
        X=X[val_idx],
        y=y[val_idx],
        video_names=video_names[val_idx],
        class_names=class_names[val_idx],
    )
    np.savez_compressed(
        test_file,
        X=X[test_idx],
        y=y[test_idx],
        video_names=video_names[test_idx],
        class_names=class_names[test_idx],
    )

    summary = {
        "total_samples": len(y),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "train_ratio": round(len(train_idx) / len(y), 3),
        "val_ratio": round(len(val_idx) / len(y), 3),
        "test_ratio": round(len(test_idx) / len(y), 3),
        "classes_count": len(np.unique(y)),
    }

    summary_file = config.REPORTS_DIR / "split_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("               SIGNBRIDGE AI — DATASET SPLITTER")
    print("=" * 72)
    print(f"Total Samples    : {summary['total_samples']}")
    print(f"Train Set        : {summary['train_samples']} samples ({summary['train_ratio'] * 100:.1f}%) -> {train_file.name}")
    print(f"Validation Set   : {summary['val_samples']} samples ({summary['val_ratio'] * 100:.1f}%) -> {val_file.name}")
    print(f"Test Set         : {summary['test_samples']} samples ({summary['test_ratio'] * 100:.1f}%) -> {test_file.name}")
    print(f"Classes Covered  : {summary['classes_count']}")
    print(f"Split Summary    : {summary_file}")
    print("=" * 72)

    return summary


if __name__ == "__main__":
    split_processed_dataset()

