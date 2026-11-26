r"""
SIGNBRIDGE AI — Video Processor Module
========================================
Handles video frame decoding, RGB conversion, and temporal timestamp calculation
using OpenCV. Strictly read-only operations to ensure raw video files are never modified.
"""

from pathlib import Path
from typing import Generator, Tuple, Dict, Any, Optional

try:
    import cv2
    import numpy as np
except ImportError:
    raise ImportError("OpenCV (cv2) and NumPy are required for video_processor.py.")


def get_video_metadata(video_path: Path) -> Dict[str, Any]:
    """
    Extracts structural metadata from a video file.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        dict: Video resolution, fps, frame count, and duration.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(raw_fps) if raw_fps and raw_fps > 0 and not np.isnan(raw_fps) else 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0.0

    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }


def iter_video_frames(
    video_path: Path,
) -> Generator[Tuple[int, int, np.ndarray], None, None]:
    """
    Generator yielding sequential frames from a video.

    Args:
        video_path (Path): Target video file path.

    Yields:
        tuple: (frame_index, timestamp_ms, rgb_frame_array)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Cannot open video stream for: {video_path}")

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(raw_fps) if raw_fps and raw_fps > 0 and not np.isnan(raw_fps) else 30.0
    frame_idx = 0
    last_timestamp_ms = -1

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Calculate strictly monotonic integer millisecond timestamps from frame index and FPS
            calculated_ms = int(round((frame_idx / fps) * 1000.0))
            if frame_idx == 0:
                timestamp_ms = max(0, calculated_ms)
            else:
                timestamp_ms = max(calculated_ms, last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            # Convert BGR (OpenCV default) to RGB (required by neural models)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield frame_idx, timestamp_ms, rgb_frame
            frame_idx += 1
    finally:
        cap.release()

