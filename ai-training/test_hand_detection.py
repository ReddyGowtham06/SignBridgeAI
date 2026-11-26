r"""
SIGNBRIDGE AI — Single-Video Hand Detection Test
==================================================
This script tests genuine MediaPipe Hand Landmarker detection on an ISL dataset video.
It processes every frame, detects hand landmarks (21 points per hand), and reports
detection metrics, including hand counts and coverage percentage.

Target Video: First .mp4 found in D:\SignLang\sign project\1.bird
Model File  : D:\SignBridgeAI\frontend\public\models\hand_landmarker.task
Environment : D:\SignBridgeAI\ai-training\.venv (Python 3.14 + MediaPipe 1.2.0)
"""

import os
import re
import sys
from pathlib import Path

# Verify OpenCV and MediaPipe availability
try:
    import cv2
except ImportError:
    print("Error: OpenCV (cv2) is not installed in this environment.")
    print("Please use the virtual environment at: D:\\SignBridgeAI\\ai-training\\.venv")
    sys.exit(1)

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("Error: MediaPipe is not installed in this environment.")
    print("Please use the virtual environment at: D:\\SignBridgeAI\\ai-training\\.venv")
    sys.exit(1)


# Paths configuration
MODEL_PATH = Path(r"D:\SignBridgeAI\frontend\public\models\hand_landmarker.task")
CLASS_DIR = Path(r"D:\SignLang\sign project\1.bird")


def natural_sort_key(s: str):
    """Sort filenames with numbers naturally (e.g. 0403(77).mp4)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def find_first_video(folder_path: Path):
    """Finds and returns the first .mp4 file in the given directory."""
    if not folder_path.exists():
        return None
    videos = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() == ".mp4"]
    videos.sort(key=lambda f: natural_sort_key(f.name))
    return videos[0] if videos else None


def main():
    print("=" * 72)
    print("         SIGNBRIDGE AI -- MEDIAPIPE HAND DETECTION TEST")
    print("=" * 72)

    # 1. Validate Model File
    print(f"\n[1/4] Checking MediaPipe model asset...")
    print(f"      Path: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"ERROR: Model file not found at:\n  {MODEL_PATH}")
        print("Please verify that hand_landmarker.task exists in frontend/public/models/.")
        sys.exit(1)
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"      Status: Model file present ({model_size_mb:.2f} MB)")

    # 2. Locate Video File
    print(f"\n[2/4] Locating video file in: {CLASS_DIR}...")
    if not CLASS_DIR.exists():
        print(f"ERROR: Target class folder does not exist at:\n  {CLASS_DIR}")
        sys.exit(1)

    video_path = find_first_video(CLASS_DIR)
    if not video_path:
        print(f"ERROR: No .mp4 video files found inside:\n  {CLASS_DIR}")
        sys.exit(1)

    print(f"      Selected video: {video_path.name}")
    print(f"      Full path: {video_path}")

    # 3. Open Video with OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: OpenCV could not open video file:\n  {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_video_frames / fps if fps > 0 else 0.0

    print(f"      Resolution : {width} x {height}")
    print(f"      Frame Rate : {fps:.2f} FPS")
    print(f"      Frame Count: {total_video_frames} frames ({duration_sec:.2f}s)")

    # 4. Initialize MediaPipe HandLandmarker
    print(f"\n[3/4] Initializing MediaPipe HandLandmarker (Task API)...")
    try:
        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)
        print("      Status: HandLandmarker initialized successfully (Dual-hand tracking active)")
    except Exception as err:
        print(f"ERROR: Failed to initialize HandLandmarker: {err}")
        cap.release()
        sys.exit(1)

    # 5. Process Every Frame
    print(f"\n[4/4] Processing all video frames...")
    total_frames_processed = 0
    frames_with_hands = 0
    hand_counts = {0: 0, 1: 0, 2: 0, "3+": 0}
    first_detected_hand_landmarks = None
    first_detected_frame_index = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Calculate monotonic timestamp in milliseconds
            timestamp_ms = int((total_frames_processed / fps) * 1000)

            # Convert BGR frame (OpenCV format) to RGB format for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Run genuine MediaPipe hand landmark detection
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            num_detected = len(detection_result.hand_landmarks) if detection_result.hand_landmarks else 0

            if num_detected > 0:
                frames_with_hands += 1

                # Capture first detected hand details for inspection
                if first_detected_hand_landmarks is None:
                    first_detected_hand_landmarks = detection_result.hand_landmarks[0]
                    first_detected_frame_index = total_frames_processed

            # Update distribution
            if num_detected in hand_counts:
                hand_counts[num_detected] += 1
            else:
                hand_counts["3+"] += 1

            total_frames_processed += 1

            # Progress log every 20 frames
            if total_frames_processed % 20 == 0 or total_frames_processed == total_video_frames:
                pct = (total_frames_processed / total_video_frames) * 100 if total_video_frames else 0
                print(f"      Progress: Frame {total_frames_processed}/{total_video_frames} ({pct:.1f}%) | "
                      f"Hands detected this frame: {num_detected}")

    except Exception as err:
        print(f"\nERROR during video frame processing: {err}")
    finally:
        cap.release()
        landmarker.close()

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 72)
    print("                     HAND DETECTION TEST RESULTS")
    print("=" * 72)
    print(f"Target Video           : {video_path.name}")
    print(f"Class Folder           : {CLASS_DIR.name}")
    print(f"Video Dimensions       : {width} x {height} @ {fps:.2f} fps")
    print(f"Total Frames Processed : {total_frames_processed}")

    if total_frames_processed > 0:
        detection_pct = (frames_with_hands / total_frames_processed) * 100.0
    else:
        detection_pct = 0.0

    print(f"Frames with Hand(s)    : {frames_with_hands} / {total_frames_processed} ({detection_pct:.2f}%)")

    print("\nHand Count Breakdown:")
    print(f"  - Frames with 0 hands: {hand_counts[0]:>4} ({(hand_counts[0] / total_frames_processed * 100):.1f}%)")
    print(f"  - Frames with 1 hand : {hand_counts[1]:>4} ({(hand_counts[1] / total_frames_processed * 100):.1f}%)")
    print(f"  - Frames with 2 hands: {hand_counts[2]:>4} ({(hand_counts[2] / total_frames_processed * 100):.1f}%)")

    # Landmark Verification
    print("\nLandmark Inspection:")
    if first_detected_hand_landmarks is not None:
        pts_count = len(first_detected_hand_landmarks)
        print(f"  - First detected at  : Frame #{first_detected_frame_index}")
        print(f"  - Landmark points    : {pts_count} points (Expected: 21)")

        wrist = first_detected_hand_landmarks[0]
        thumb_tip = first_detected_hand_landmarks[4]
        index_tip = first_detected_hand_landmarks[8]
        print(f"  - Sample Landmark Coordinates (Normalized 0.0 - 1.0):")
        print(f"      Wrist [Point 0]     : (x={wrist.x:.4f}, y={wrist.y:.4f}, z={wrist.z:.4f})")
        print(f"      Thumb Tip [Point 4] : (x={thumb_tip.x:.4f}, y={thumb_tip.y:.4f}, z={thumb_tip.z:.4f})")
        print(f"      Index Tip [Point 8] : (x={index_tip.x:.4f}, y={index_tip.y:.4f}, z={index_tip.z:.4f})")
    else:
        print("  - No hands were detected in this video.")

    print("\nIntegrity Note:")
    print("  - Detection was executed using the genuine MediaPipe HandLandmarker neural model.")
    print("  - Original video and model files were accessed read-only; no files were modified.")
    print("=" * 72)


if __name__ == "__main__":
    main()
