r"""
SIGNBRIDGE AI — Dataset Scanner Module
========================================
Discovers all sign classes and MP4 videos in the dataset, checks OpenCV readability,
extracts container metadata (dimensions, FPS, duration), and generates a clean manifest.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any

try:
    import cv2
except ImportError:
    print("Error: OpenCV (cv2) not found in the current Python environment.")
    sys.exit(1)

import config


def natural_sort_key(s: str) -> List[Any]:
    """Sort strings with embedded numbers naturally (e.g., '2.black' before '10.youareperfect')."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def scan_dataset(dataset_dir: Path = config.DATASET_DIR) -> Dict[str, Any]:
    """
    Scans the dataset directory, tests video readability, and gathers statistics.

    Returns:
        dict: Manifest containing summary and detailed class video records.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda d: natural_sort_key(d.name))

    manifest = {
        "dataset_path": str(dataset_dir),
        "total_classes": len(class_dirs),
        "total_videos": 0,
        "readable_videos": 0,
        "unreadable_videos": 0,
        "classes": {},
        "unreadable_files": [],
    }

    for class_dir in class_dirs:
        class_name = class_dir.name
        video_files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".mp4"
        ]
        video_files.sort(key=lambda f: natural_sort_key(f.name))

        class_records = []
        for vpath in video_files:
            cap = cv2.VideoCapture(str(vpath))
            is_readable = False
            width = 0
            height = 0
            fps = 0.0
            frame_count = 0
            duration_sec = 0.0
            error_msg = ""

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    is_readable = True
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration_sec = frame_count / fps if fps > 0 else 0.0
                else:
                    error_msg = "Could not decode first frame"
            else:
                error_msg = "VideoCapture failed to open file"

            cap.release()

            record = {
                "file_name": vpath.name,
                "file_path": str(vpath),
                "relative_path": f"{class_name}/{vpath.name}",
                "is_readable": is_readable,
                "width": width,
                "height": height,
                "fps": round(fps, 2),
                "frame_count": frame_count,
                "duration_sec": round(duration_sec, 2),
                "error_message": error_msg,
            }
            class_records.append(record)

            manifest["total_videos"] += 1
            if is_readable:
                manifest["readable_videos"] += 1
            else:
                manifest["unreadable_videos"] += 1
                manifest["unreadable_files"].append(record)

        manifest["classes"][class_name] = {
            "total_count": len(video_files),
            "readable_count": sum(1 for r in class_records if r["is_readable"]),
            "unreadable_count": sum(1 for r in class_records if not r["is_readable"]),
            "videos": class_records,
        }

    # Save manifest report
    manifest_path = config.REPORTS_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def print_scan_report(manifest: Dict[str, Any]):
    """Prints a formatted summary report to stdout."""
    print("=" * 72)
    print("                SIGNBRIDGE AI — DATASET SCAN REPORT")
    print("=" * 72)
    print(f"Dataset Path          : {manifest['dataset_path']}")
    print(f"Total Sign Classes    : {manifest['total_classes']}")
    print(f"Total Video Files     : {manifest['total_videos']}")
    print(f"Readable Videos       : {manifest['readable_videos']}")
    print(f"Unreadable Videos     : {manifest['unreadable_videos']}")
    print("=" * 72)

    print("\nClass Distribution Summary:")
    print("-" * 72)
    print(f"{'Class Name':<35} | {'Videos':>8} | {'Readable':>8} | {'Unreadable':>10}")
    print("-" * 72)
    for c_name, c_info in manifest["classes"].items():
        flag = "(!)" if c_info["unreadable_count"] > 0 else "   "
        print(
            f"{flag} {c_name:<33} | "
            f"{c_info['total_count']:>8} | "
            f"{c_info['readable_count']:>8} | "
            f"{c_info['unreadable_count']:>10}"
        )
    print("-" * 72)

    if manifest["unreadable_files"]:
        print("\nATTENTION: Unreadable Videos:")
        for uf in manifest["unreadable_files"]:
            print(f"  - {uf['file_path']} ({uf['error_message']})")
    else:
        print("\nAll videos are readable and valid.")

    print(f"\nManifest saved to: {config.REPORTS_DIR / 'dataset_manifest.json'}")


if __name__ == "__main__":
    print("Scanning dataset...")
    scan_results = scan_dataset()
    print_scan_report(scan_results)

