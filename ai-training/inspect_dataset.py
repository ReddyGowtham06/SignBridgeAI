r"""
SIGNBRIDGE AI — Dataset Inspection Script
===========================================
This script inspects video datasets for Indian Sign Language (ISL) model training.
It checks video integrity using OpenCV, extracts resolution, frame rate, and duration,
and generates a complete readable/unreadable video validation summary.

Dataset Target: D:\SignLang\sign project
"""

import os
import re
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Error: OpenCV (cv2) is not installed in the current Python environment.")
    print("Please run this script using a Python environment that has OpenCV installed.")
    sys.exit(1)


# Default dataset directory
DEFAULT_DATASET_DIR = Path(r"D:\SignLang\sign project")


def natural_sort_key(s: str):
    """
    Helper function to sort folder names naturally (e.g., '2.black' before '10.youareperfect').
    Splits numeric and non-numeric substrings so integers are compared numerically.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def inspect_video(video_path: Path):
    """
    Opens a video file with OpenCV and verifies if its first frame can be read.

    Returns:
        dict: Inspection metrics (is_readable, width, height, fps, frame_count, duration, error_msg)
    """
    result = {
        "file_name": video_path.name,
        "file_path": str(video_path),
        "is_readable": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_sec": 0.0,
        "error_message": "",
    }

    # Verify file existence and non-zero size
    try:
        file_size = video_path.stat().st_size
        if file_size == 0:
            result["error_message"] = "File is empty (0 bytes)"
            return result
    except Exception as e:
        result["error_message"] = f"File access error: {e}"
        return result

    cap = None
    try:
        # Open video with OpenCV VideoCapture
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            result["error_message"] = "cv2.VideoCapture failed to open file (unsupported codec or corrupt header)"
            return result

        # Read the first frame to confirm decodability
        ret, frame = cap.read()
        if not ret or frame is None:
            result["error_message"] = "Unable to decode first frame (corrupted video stream or 0 valid frames)"
            return result

        # Extract metadata
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        duration_sec = 0.0
        if fps and fps > 0 and frame_count > 0:
            duration_sec = frame_count / fps

        result["is_readable"] = True
        result["width"] = width
        result["height"] = height
        result["fps"] = fps
        result["frame_count"] = frame_count
        result["duration_sec"] = duration_sec

    except Exception as err:
        result["error_message"] = f"Unexpected error during video read: {err}"
    finally:
        if cap is not None:
            cap.release()

    return result


def main():
    # Allow passing custom dataset path via command-line arguments, otherwise use default
    if len(sys.argv) > 1:
        if sys.argv[1] in ("-h", "--help"):
            print("Usage: python inspect_dataset.py [path_to_dataset]")
            print("Default path: D:\\SignLang\\sign project")
            sys.exit(0)
        dataset_dir = Path(sys.argv[1])
    else:
        dataset_dir = DEFAULT_DATASET_DIR

    print("=" * 72)
    print("                 SIGNBRIDGE AI — DATASET INSPECTOR")
    print("=" * 72)
    print(f"Target Directory: {dataset_dir}\n")

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory does not exist at:\n  {dataset_dir}")
        print("Please check the path and try again.")
        sys.exit(1)

    if not dataset_dir.is_dir():
        print(f"ERROR: Specified path is not a directory:\n  {dataset_dir}")
        sys.exit(1)

    # Find all immediate subfolders as sign classes
    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda d: natural_sort_key(d.name))

    if not class_dirs:
        print(f"No subdirectories found in: {dataset_dir}")
        print("Expected sign class folders like '1.bird', '2.black', etc.")
        sys.exit(0)

    total_classes = len(class_dirs)
    total_videos = 0
    total_readable = 0
    total_unreadable = 0
    unreadable_list = []
    class_stats = []

    print(f"Found {total_classes} sign class subfolders. Starting inspection...\n")

    for idx, class_dir in enumerate(class_dirs, 1):
        class_name = class_dir.name

        # Find all MP4 video files (case-insensitive)
        video_files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".mp4"
        ]
        video_files.sort(key=lambda f: natural_sort_key(f.name))

        class_video_count = len(video_files)
        total_videos += class_video_count

        class_readable_count = 0
        class_unreadable_count = 0

        print(f"[{idx}/{total_classes}] Class: '{class_name}' ({class_video_count} videos)")

        if class_video_count == 0:
            print("    (No .mp4 videos found in this folder)")
        else:
            for video_path in video_files:
                metrics = inspect_video(video_path)

                if metrics["is_readable"]:
                    class_readable_count += 1
                    total_readable += 1
                    print(
                        f"    [OK] {metrics['file_name']:<30} | "
                        f"{metrics['width']}x{metrics['height']} | "
                        f"{metrics['fps']:.2f} fps | "
                        f"{metrics['frame_count']} frames | "
                        f"{metrics['duration_sec']:.2f}s"
                    )
                else:
                    class_unreadable_count += 1
                    total_unreadable += 1
                    unreadable_list.append((class_name, metrics['file_name'], metrics['error_message']))
                    print(
                        f"    [FAIL] {metrics['file_name']:<30} | "
                        f"ERROR: {metrics['error_message']}"
                    )

        class_stats.append({
            "class_name": class_name,
            "total": class_video_count,
            "readable": class_readable_count,
            "unreadable": class_unreadable_count,
        })
        print()  # Empty line between classes

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    print("=" * 72)
    print("                      DATASET INSPECTION SUMMARY")
    print("=" * 72)
    print(f"Dataset Location       : {dataset_dir}")
    print(f"Total Sign Classes     : {total_classes}")
    print(f"Total Video Files      : {total_videos}")

    if total_videos > 0:
        readable_pct = (total_readable / total_videos) * 100
        unreadable_pct = (total_unreadable / total_videos) * 100
    else:
        readable_pct = 0.0
        unreadable_pct = 0.0

    print(f"Readable Videos        : {total_readable} ({readable_pct:.1f}%)")
    print(f"Unreadable Videos      : {total_unreadable} ({unreadable_pct:.1f}%)")
    print("=" * 72)

    # Class distribution summary (top classes or all)
    print("\nVideos per Class Summary:")
    print("-" * 72)
    print(f"{'Class Name':<35} | {'Total':>7} | {'Readable':>8} | {'Unreadable':>10}")
    print("-" * 72)
    for stat in class_stats:
        status_marker = "(!)" if stat["unreadable"] > 0 else "   "
        print(
            f"{status_marker} {stat['class_name']:<33} | "
            f"{stat['total']:>7} | "
            f"{stat['readable']:>8} | "
            f"{stat['unreadable']:>10}"
        )
    print("-" * 72)

    # Report unreadable video details if any exist
    if unreadable_list:
        print("\n" + "!" * 72)
        print("ATTENTION: UNREADABLE VIDEOS DETECTED")
        print("!" * 72)
        for c_name, f_name, err in unreadable_list:
            print(f" - [{c_name}] {f_name}")
            print(f"   Reason: {err}")
        print("!" * 72)
    else:
        print("\nAll videos were opened and their first frames decoded successfully!")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
