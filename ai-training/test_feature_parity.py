r"""
SIGNBRIDGE AI — Feature Builder Parity Test Harness
=====================================================
Validates numerical parity between the Python feature extraction pipeline
(`dataset_builder.py`) and the TypeScript frontend feature builder (`featureBuilder.ts`).

Tests real landmark sequences from the processed dataset across diverse profiles:
1. Two-hand active gestures
2. Left-hand only gestures (missing right hand)
3. Right-hand only gestures (missing left hand)
4. Intermittent occlusion gestures (hands entering/exiting field of view)
5. Multi-signer / additional sign classes
"""

import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Real-time line buffering
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required.")
    sys.exit(1)

import config
import dataset_builder

TS_RUNNER = config.TRAINING_DIR / "test_parity_runner.mjs"
RAW_LANDMARKS_DIR = config.RAW_LANDMARKS_DIR
REPORTS_DIR = config.REPORTS_DIR


def run_typescript_feature_builder(frames: List[Dict[str, Any]]) -> np.ndarray:
    """Invokes the TypeScript feature builder via Node 24 and returns the resulting 758 floats."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as in_f:
        json.dump({"frames": frames}, in_f)
        in_path = in_f.name

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as out_f:
        out_path = out_f.name

    try:
        cmd = [
            "node",
            "--experimental-strip-types",
            str(TS_RUNNER),
            in_path,
            out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

        with open(out_path, "r", encoding="utf-8") as f:
            res_data = json.load(f)

        features = np.array(res_data["features"], dtype=np.float32)
        return features
    finally:
        try:
            Path(in_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass


def find_test_cases() -> List[Dict[str, Any]]:
    """Selects representative videos matching all required profile test categories."""
    all_files = list(RAW_LANDMARKS_DIR.rglob("*.npz"))
    if not all_files:
        raise FileNotFoundError(f"No .npz landmark archives found in {RAW_LANDMARKS_DIR}")

    selected = []

    # 1. Two-hand active (>85% both hands)
    for f in all_files:
        d = np.load(f)
        mask = d["presence_mask"]
        l_r = float(np.mean(mask[:, 0]))
        r_r = float(np.mean(mask[:, 1]))
        if l_r > 0.85 and r_r > 0.85:
            selected.append({
                "category": "Two-Hand Active",
                "file": f,
                "class_name": f.parent.name,
                "left_ratio": l_r,
                "right_ratio": r_r,
            })
            break

    # 2. Left-hand only (Right hand completely 0.0%)
    for f in all_files:
        d = np.load(f)
        mask = d["presence_mask"]
        l_r = float(np.mean(mask[:, 0]))
        r_r = float(np.mean(mask[:, 1]))
        if l_r > 0.85 and r_r == 0.0:
            selected.append({
                "category": "Left-Hand Only (Missing Right)",
                "file": f,
                "class_name": f.parent.name,
                "left_ratio": l_r,
                "right_ratio": r_r,
            })
            break

    # 3. Right-hand only (Left hand completely 0.0%)
    for f in all_files:
        d = np.load(f)
        mask = d["presence_mask"]
        l_r = float(np.mean(mask[:, 0]))
        r_r = float(np.mean(mask[:, 1]))
        if l_r == 0.0 and r_r > 0.50:
            selected.append({
                "category": "Right-Hand Only (Missing Left)",
                "file": f,
                "class_name": f.parent.name,
                "left_ratio": l_r,
                "right_ratio": r_r,
            })
            break

    # 4. Intermittent Occlusion (Hands dropping out and re-entering multiple times)
    for f in all_files:
        d = np.load(f)
        mask = d["presence_mask"]
        l_r = float(np.mean(mask[:, 0]))
        r_r = float(np.mean(mask[:, 1]))
        # Detect dropouts (True -> False) and re-entries (False -> True)
        m0 = mask[:, 0]
        drops = int(np.sum((m0[:-1] == True) & (m0[1:] == False)))
        gains = int(np.sum((m0[:-1] == False) & (m0[1:] == True)))
        if drops >= 2 and gains >= 2 and 0.50 < l_r < 0.95:
            selected.append({
                "category": "Intermittent Occlusion",
                "file": f,
                "class_name": f.parent.name,
                "left_ratio": l_r,
                "right_ratio": r_r,
            })
            break

    # 5. Additional class verification
    for f in all_files:
        if f.parent.name == "19.girl" and f not in [s["file"] for s in selected]:
            d = np.load(f)
            mask = d["presence_mask"]
            selected.append({
                "category": "Additional Dynamic Gesture",
                "file": f,
                "class_name": f.parent.name,
                "left_ratio": float(np.mean(mask[:, 0])),
                "right_ratio": float(np.mean(mask[:, 1])),
            })
            break

    return selected


def evaluate_test_case(test_case: Dict[str, Any], tolerance: float = 1e-4) -> Dict[str, Any]:
    """Runs Python vs TypeScript feature building and computes parity metrics."""
    file_path: Path = test_case["file"]
    with np.load(file_path) as data:
        landmarks = data["landmarks"].astype(np.float32)          # (N, 2, 21, 3)
        presence_mask = data["presence_mask"].astype(bool)        # (N, 2)
        timestamps_ms = data["timestamps_ms"].astype(np.int64)    # (N,)

    n_frames = len(timestamps_ms)

    # 1. Compute reference Python feature vector
    py_features = dataset_builder.compute_video_feature_vector(
        landmarks=landmarks,
        presence_mask=presence_mask,
        timestamps_ms=timestamps_ms,
    )

    # 2. Build input frames JSON structure for TypeScript
    frames = []
    for i in range(n_frames):
        left_pts = (
            [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in landmarks[i, 0]]
            if presence_mask[i, 0]
            else None
        )
        right_pts = (
            [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in landmarks[i, 1]]
            if presence_mask[i, 1]
            else None
        )
        frames.append({
            "timestampMs": int(timestamps_ms[i]),
            "landmarks": [left_pts, right_pts],
        })

    # 3. Compute TypeScript feature vector
    ts_features = run_typescript_feature_builder(frames)

    # 4. Compare feature vectors
    if len(py_features) != len(ts_features):
        raise ValueError(f"Length mismatch: Python {len(py_features)} vs TS {len(ts_features)}")

    abs_diff = np.abs(py_features - ts_features)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    mismatches = int(np.sum(abs_diff > tolerance))

    # Breakdown by feature group
    group_slices = {
        "Left Hand (Slot 0) Presence Ratio": (0, 1),
        "Left Hand Spatial Mean": (1, 64),
        "Left Hand Spatial Std": (64, 127),
        "Left Hand Spatial Min": (127, 190),
        "Left Hand Spatial Max": (190, 253),
        "Left Hand Vel Mean": (253, 316),
        "Left Hand Vel Std": (316, 379),
        "Right Hand (Slot 1) Presence Ratio": (379, 380),
        "Right Hand Spatial Mean": (380, 443),
        "Right Hand Spatial Std": (443, 506),
        "Right Hand Spatial Min": (506, 569),
        "Right Hand Spatial Max": (569, 632),
        "Right Hand Vel Mean": (632, 695),
        "Right Hand Vel Std": (695, 758),
    }

    group_stats = {}
    for g_name, (start, end) in group_slices.items():
        g_diff = abs_diff[start:end]
        group_stats[g_name] = {
            "max_diff": float(np.max(g_diff)),
            "mean_diff": float(np.mean(g_diff)),
            "all_finite": bool(np.all(np.isfinite(ts_features[start:end]))),
        }

    return {
        "category": test_case["category"],
        "class_name": test_case["class_name"],
        "filename": file_path.name,
        "frames_count": n_frames,
        "left_presence_ratio": test_case["left_ratio"],
        "right_presence_ratio": test_case["right_ratio"],
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "mismatches_above_tol": mismatches,
        "tolerance": tolerance,
        "passed": bool(max_diff <= tolerance),
        "groups": group_stats,
    }


def main():
    print("=" * 75)
    print("SIGNBRIDGE AI — PHASE 7B: PYTHON-TO-TYPESCRIPT FEATURE BUILDER PARITY TEST")
    print("=" * 75)

    test_cases = find_test_cases()
    print(f"Discovered {len(test_cases)} representative test cases across required profiles:\n")

    results = []
    all_passed = True

    for tc in test_cases:
        res = evaluate_test_case(tc, tolerance=1e-4)
        results.append(res)
        status = "PASSED" if res["passed"] else "FAILED"
        if not res["passed"]:
            all_passed = False

        print(f"[{status}] {res['category']:<32} | Video: {res['filename']} ({res['class_name']})")
        print(f"       Frames: {res['frames_count']} | Left Ratio: {res['left_presence_ratio']:.3f} | Right Ratio: {res['right_presence_ratio']:.3f}")
        print(f"       Max Abs Diff: {res['max_abs_diff']:.2e} | Mean Abs Diff: {res['mean_abs_diff']:.2e} | Mismatches: {res['mismatches_above_tol']}")
        print("       Group Max Differences:")
        for g_name, g_stat in res["groups"].items():
            print(f"         - {g_name:<36}: max {g_stat['max_diff']:.2e}")
        print("-" * 75)

    # Save detailed report
    report_path = REPORTS_DIR / "feature_builder_parity_report.json"
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_test_cases": len(test_cases),
        "all_passed": all_passed,
        "tolerance": 1e-4,
        "results": results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("\n" + "=" * 75)
    print(f"OVERALL FEATURE BUILDER PARITY: {'100% PASSED' if all_passed else 'FAILED'}")
    print(f"Report saved to: {report_path}")
    print("=" * 75)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
