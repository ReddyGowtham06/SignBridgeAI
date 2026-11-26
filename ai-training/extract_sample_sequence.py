r"""
SIGNBRIDGE AI — Sample Video Landmark Sequence Extractor
==========================================================
This script extracts genuine MediaPipe Hand Landmarker 3D coordinates from a sample ISL video
and stores them in a structured, compressed NumPy format (.npz) suitable for machine learning.

Input:
  - Video: D:\SignLang\sign project\1.bird\0403(77).mp4
  - Model: D:\SignBridgeAI\frontend\public\models\hand_landmarker.task

Output:
  - D:\SignBridgeAI\ai-training\sample_output\bird_sample.npz

Hand Slot Convention:
  - Slot 0: Left Hand
  - Slot 1: Right Hand
  - If a hand is not detected in a frame, its presence mask is set to False (0),
    and coordinates remain zeros. Landmarks are never invented or interpolated.

Important Notice:
  This script extracts raw geometric landmark sequences for dataset preparation.
  It is NOT a trained sign language recognition model.
"""

import sys
from pathlib import Path

# Verify external library availability
try:
    import cv2
except ImportError:
    print("Error: OpenCV (cv2) is not installed in the active environment.")
    print("Please run this script inside: D:\\SignBridgeAI\\ai-training\\.venv")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: NumPy is not installed in the active environment.")
    sys.exit(1)

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("Error: MediaPipe is not installed in the active environment.")
    sys.exit(1)


# Path definitions
INPUT_VIDEO_PATH = Path(r"D:\SignLang\sign project\1.bird\0403(77).mp4")
MODEL_PATH = Path(r"D:\SignBridgeAI\frontend\public\models\hand_landmarker.task")
OUTPUT_DIR = Path(r"D:\SignBridgeAI\ai-training\sample_output")
OUTPUT_FILE = OUTPUT_DIR / "bird_sample.npz"


def main():
    print("=" * 72)
    print("        SIGNBRIDGE AI — LANDMARK SEQUENCE EXTRACTION")
    print("=" * 72)

    # 1. Verify Input Video
    print(f"\n[1/5] Verifying input video file...")
    print(f"      Path: {INPUT_VIDEO_PATH}")
    if not INPUT_VIDEO_PATH.exists():
        print(f"ERROR: Input video not found at:\n  {INPUT_VIDEO_PATH}")
        sys.exit(1)
    if INPUT_VIDEO_PATH.stat().st_size == 0:
        print(f"ERROR: Input video file is empty (0 bytes).")
        sys.exit(1)
    print("      Status: Video file exists and is accessible.")

    # 2. Verify MediaPipe Model
    print(f"\n[2/5] Verifying MediaPipe Hand Landmarker model...")
    print(f"      Path: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"ERROR: Model file not found at:\n  {MODEL_PATH}")
        sys.exit(1)
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"      Status: Model file verified ({model_size_mb:.2f} MB)")

    # 3. Open Video Stream
    print(f"\n[3/5] Opening video stream with OpenCV...")
    cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
    if not cap.isOpened():
        print(f"ERROR: OpenCV could not open video: {INPUT_VIDEO_PATH}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    print(f"      Dimensions  : {width} x {height}")
    print(f"      Frame Rate  : {fps:.2f} FPS")
    print(f"      Total Frames: {total_frames} ({duration_sec:.2f} seconds)")

    # 4. Initialize MediaPipe HandLandmarker in VIDEO mode
    print(f"\n[4/5] Initializing MediaPipe HandLandmarker (Task API)...")
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
        print("      Status: Landmarker ready (VIDEO mode, max 2 hands).")
    except Exception as err:
        print(f"ERROR: Failed to initialize HandLandmarker: {err}")
        cap.release()
        sys.exit(1)

    # 5. Extract Landmarks Frame by Frame
    print(f"\n[5/5] Extracting hand landmarks across all frames...")

    landmarks_list = []
    presence_mask_list = []
    handedness_list = []
    timestamps_list = []

    frames_processed = 0
    frames_no_hands = 0
    frames_one_hand = 0
    frames_two_hands = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            timestamp_ms = int((frames_processed / fps) * 1000)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Slot 0 = Left Hand, Slot 1 = Right Hand
            frame_coords = np.zeros((2, 21, 3), dtype=np.float32)
            frame_presence = np.zeros(2, dtype=bool)
            frame_handedness_labels = ["none", "none"]

            num_detected = (
                len(detection_result.hand_landmarks)
                if detection_result.hand_landmarks
                else 0
            )

            if num_detected == 0:
                frames_no_hands += 1
            elif num_detected == 1:
                frames_one_hand += 1
            else:
                frames_two_hands += 1

            if num_detected > 0:
                for hand_idx, landmarks in enumerate(detection_result.hand_landmarks):
                    label = "Unknown"
                    if (
                        detection_result.handedness
                        and hand_idx < len(detection_result.handedness)
                        and detection_result.handedness[hand_idx]
                    ):
                        label = detection_result.handedness[hand_idx][0].category_name

                    target_slot = 0 if label.lower() == "left" else 1

                    if frame_presence[target_slot]:
                        other_slot = 1 - target_slot
                        if not frame_presence[other_slot]:
                            target_slot = other_slot

                    coords = np.array(
                        [[lm.x, lm.y, lm.z] for lm in landmarks],
                        dtype=np.float32,
                    )

                    frame_coords[target_slot] = coords
                    frame_presence[target_slot] = True
                    frame_handedness_labels[target_slot] = label

            landmarks_list.append(frame_coords)
            presence_mask_list.append(frame_presence)
            handedness_list.append(frame_handedness_labels)
            timestamps_list.append(timestamp_ms)

            frames_processed += 1

    except Exception as e:
        print(f"ERROR during frame extraction: {e}")
        sys.exit(1)
    finally:
        cap.release()
        landmarker.close()

    landmarks_array = np.array(landmarks_list, dtype=np.float32)
    presence_mask_array = np.array(presence_mask_list, dtype=bool)
    handedness_array = np.array(handedness_list, dtype="<U10")
    timestamps_array = np.array(timestamps_list, dtype=np.int64)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE,
        landmarks=landmarks_array,
        presence_mask=presence_mask_array,
        handedness=handedness_array,
        timestamps_ms=timestamps_array,
        video_name=np.array(INPUT_VIDEO_PATH.name),
        class_name=np.array("1.bird"),
        slot_mapping=np.array("Slot 0: Left Hand, Slot 1: Right Hand"),
    )

    print(f"\n      Saved compressed NumPy archive successfully:")
    print(f"      -> {OUTPUT_FILE}")
    file_size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"      File size: {file_size_kb:.2f} KB")

    print("\n" + "=" * 72)
    print("                     SEQUENCE EXTRACTION SUMMARY")
    print("=" * 72)
    print(f"Input Video File       : {INPUT_VIDEO_PATH.name}")
    print(f"Output Archive         : {OUTPUT_FILE}")
    print(f"Total Video Frames     : {frames_processed}")
    print(f"Hand Detection Rates   :")
    print(f"  - Frames with 2 hands: {frames_two_hands:>4} ({(frames_two_hands / frames_processed * 100):.1f}%)")
    print(f"  - Frames with 1 hand : {frames_one_hand:>4} ({(frames_one_hand / frames_processed * 100):.1f}%)")
    print(f"  - Frames with NO hand: {frames_no_hands:>4} ({(frames_no_hands / frames_processed * 100):.1f}%)")

    print("\nNumPy Array Specifications:")
    print(f"  1. 'landmarks'     shape: {landmarks_array.shape:<16} dtype: {landmarks_array.dtype}")
    print(f"  2. 'presence_mask' shape: {presence_mask_array.shape:<16} dtype: {presence_mask_array.dtype}")
    print(f"  3. 'handedness'    shape: {handedness_array.shape:<16} dtype: {handedness_array.dtype}")
    print(f"  4. 'timestamps_ms' shape: {timestamps_array.shape:<16} dtype: {timestamps_array.dtype}")
    print("=" * 72)


if __name__ == "__main__":
    main()

