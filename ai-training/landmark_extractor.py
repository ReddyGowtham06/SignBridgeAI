r"""
SIGNBRIDGE AI — Landmark Extractor Module
===========================================
Batch processes dataset videos through the MediaPipe Hand Landmarker model.
Saves extracted landmark sequences as compressed NumPy (.npz) archives.
Includes resumable execution: skips already extracted video files.
Ensures strictly monotonic per-video timestamps, atomic file saving, and clean resource cleanup.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure real-time line buffering in console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

try:
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError as err:
    print(f"Error importing dependencies for landmark_extractor: {err}")
    sys.exit(1)

import config
from video_processor import iter_video_frames, get_video_metadata


def create_landmarker(
    model_path: Path = config.MODEL_TASK_PATH,
) -> vision.HandLandmarker:
    """
    Initializes a MediaPipe HandLandmarker instance in VIDEO mode.

    Args:
        model_path (Path): Path to hand_landmarker.task model.

    Returns:
        vision.HandLandmarker: Configured landmarker ready for frame inference.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"MediaPipe model task not found at: {model_path}")

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=config.NUM_HANDS,
        min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def is_valid_npz(path: Path) -> bool:
    """
    Checks if an existing .npz archive is uncorrupted and contains all required arrays.

    Args:
        path (Path): File path to verify.

    Returns:
        bool: True if the file exists, has valid structure, matching lengths, and monotonic timestamps.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as data:
            required_keys = {"landmarks", "presence_mask", "handedness", "timestamps_ms"}
            if not required_keys.issubset(data.files):
                return False
            num_frames = len(data["timestamps_ms"])
            if num_frames == 0 or len(data["landmarks"]) != num_frames or len(data["presence_mask"]) != num_frames:
                return False
            if num_frames > 1 and not np.all(np.diff(data["timestamps_ms"]) > 0):
                return False
            return True
    except Exception:
        return False


def extract_video_landmarks(
    video_path: Path,
    output_path: Path,
    landmarker: Optional[vision.HandLandmarker] = None,
    model_path: Path = config.MODEL_TASK_PATH,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Extracts 21 3D hand landmarks per hand for every frame in a single video.

    Args:
        video_path (Path): Video file to process.
        output_path (Path): Destination .npz path.
        landmarker (Optional[vision.HandLandmarker]): Optional external landmarker instance.
            If None (recommended), a fresh landmarker is created and closed for this video,
            ensuring video-specific timestamp isolation.
        model_path (Path): Path to hand_landmarker.task model.
        overwrite (bool): Whether to re-extract if output_path already exists.

    Returns:
        dict: Extraction summary statistics.
    """
    if output_path.exists() and not overwrite:
        if is_valid_npz(output_path):
            return {
                "video_path": str(video_path),
                "output_path": str(output_path),
                "status": "skipped_already_exists",
                "frames": 0,
                "frames_with_hands": 0,
            }
        else:
            print(f"  [WARN] Existing file corrupted/incomplete, re-extracting: {output_path.name}")

    landmarks_list = []
    presence_mask_list = []
    handedness_list = []
    timestamps_list = []

    frames_processed = 0
    hands_detected_frames = 0
    last_timestamp_ms = -1

    def _process_with_landmarker(lm: vision.HandLandmarker):
        nonlocal frames_processed, hands_detected_frames, last_timestamp_ms
        for frame_idx, timestamp_ms, rgb_frame in iter_video_frames(video_path):
            # Enforce strictly monotonic timestamp within this video
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = lm.detect_for_video(mp_image, timestamp_ms)

            # Slot 0 = Left Hand, Slot 1 = Right Hand
            frame_coords = np.zeros(
                (config.NUM_HANDS, config.LANDMARKS_PER_HAND, config.COORDS_PER_LANDMARK),
                dtype=np.float32,
            )
            frame_presence = np.zeros(config.NUM_HANDS, dtype=bool)
            frame_handedness_labels = ["none", "none"]

            num_detected = len(detection_result.hand_landmarks) if detection_result.hand_landmarks else 0
            if num_detected > 0:
                hands_detected_frames += 1

                for hand_idx, landmarks in enumerate(detection_result.hand_landmarks):
                    label = "Unknown"
                    if (
                        detection_result.handedness
                        and hand_idx < len(detection_result.handedness)
                        and detection_result.handedness[hand_idx]
                    ):
                        label = detection_result.handedness[hand_idx][0].category_name

                    # Canonical slot assignment: Left -> Slot 0, Right -> Slot 1
                    target_slot = config.SLOT_LEFT if label.lower() == "left" else config.SLOT_RIGHT

                    # Conflict resolution: if slot already taken, use the available other slot
                    if frame_presence[target_slot]:
                        other_slot = 1 - target_slot
                        if not frame_presence[other_slot]:
                            target_slot = other_slot

                    coords = np.array([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)
                    frame_coords[target_slot] = coords
                    frame_presence[target_slot] = True
                    frame_handedness_labels[target_slot] = label

            landmarks_list.append(frame_coords)
            presence_mask_list.append(frame_presence)
            handedness_list.append(frame_handedness_labels)
            timestamps_list.append(timestamp_ms)
            frames_processed += 1

    if landmarker is not None:
        _process_with_landmarker(landmarker)
    else:
        # Create and close a fresh landmarker specifically for this video
        with create_landmarker(model_path) as fresh_lm:
            _process_with_landmarker(fresh_lm)

    if frames_processed == 0:
        raise ValueError(f"No frames could be read from video: {video_path}")

    # Atomic write to temporary file before replacing final destination
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".tmp_{output_path.stem}_{int(time.time() * 1000)}.npz")
    try:
        np.savez_compressed(
            temp_path,
            landmarks=np.array(landmarks_list, dtype=np.float32),
            presence_mask=np.array(presence_mask_list, dtype=bool),
            handedness=np.array(handedness_list, dtype="<U10"),
            timestamps_ms=np.array(timestamps_list, dtype=np.int64),
            video_name=np.array(video_path.name),
            class_name=np.array(video_path.parent.name),
        )
        # Atomically replace destination file
        temp_path.replace(output_path)
    finally:
        # Clean up temporary file if it still exists (e.g. exception prior to replace)
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return {
        "video_path": str(video_path),
        "output_path": str(output_path),
        "status": "extracted",
        "frames": frames_processed,
        "frames_with_hands": hands_detected_frames,
    }


def extract_all(
    dataset_dir: Path = config.DATASET_DIR,
    output_root: Path = config.RAW_LANDMARKS_DIR,
    limit_per_class: Optional[int] = None,
    limit_total: Optional[int] = None,
    overwrite: bool = False,
    target_class: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Batch extracts landmarks across dataset videos with per-video landmarker lifecycle,
    resumable execution, atomic writes, and clear error reporting.
    """
    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda d: d.name)

    if target_class:
        class_dirs = [d for d in class_dirs if d.name == target_class]
        if not class_dirs:
            raise ValueError(f"Target class not found: {target_class}")

    total_scanned = 0
    processed_videos: List[Dict[str, Any]] = []
    skipped_videos: List[str] = []
    failed_videos: List[Dict[str, str]] = []
    start_time = time.time()

    print("=" * 76)
    print("           SIGNBRIDGE AI — BATCH LANDMARK EXTRACTION")
    print("=" * 76)
    print(f"Dataset Source : {dataset_dir}")
    print(f"Output Root    : {output_root}")
    print(f"Total Classes  : {len(class_dirs)}")
    print(f"Resumable Mode : {'Disabled (Overwriting)' if overwrite else 'Enabled (Skipping valid existing)'}")
    print("=" * 76 + "\n")

    for c_idx, class_dir in enumerate(class_dirs, 1):
        class_name = class_dir.name
        video_files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() == ".mp4"]
        video_files.sort(key=lambda f: f.name)

        if limit_per_class:
            video_files = video_files[:limit_per_class]

        for v_idx, video_path in enumerate(video_files, 1):
            if limit_total and (len(processed_videos) + len(skipped_videos)) >= limit_total:
                print("\nReached global processing limit.")
                break

            total_scanned += 1
            dest_file = output_root / class_name / f"{video_path.stem}.npz"

            # Check if valid file already exists (fast skip before reading frames)
            if dest_file.exists() and not overwrite and is_valid_npz(dest_file):
                skipped_videos.append(video_path.name)
                print(
                    f"[{c_idx}/{len(class_dirs)}] {class_name:<20} | "
                    f"{video_path.name:<24} | "
                    f"[SKIPPED] Already extracted (verified)"
                )
                continue

            try:
                # Fresh VIDEO-mode landmarker created and closed per video
                res = extract_video_landmarks(
                    video_path=video_path,
                    output_path=dest_file,
                    landmarker=None,
                    overwrite=overwrite,
                )

                if res["status"] == "skipped_already_exists":
                    skipped_videos.append(video_path.name)
                    print(
                        f"[{c_idx}/{len(class_dirs)}] {class_name:<20} | "
                        f"{video_path.name:<24} | "
                        f"[SKIPPED] Already extracted (verified)"
                    )
                else:
                    processed_videos.append(res)
                    pct_hands = (res["frames_with_hands"] / res["frames"] * 100) if res["frames"] > 0 else 0
                    print(
                        f"[{c_idx}/{len(class_dirs)}] {class_name:<20} | "
                        f"{video_path.name:<24} | "
                        f"{res['frames']} frames ({pct_hands:.1f}% hands detected) [OK]"
                    )
            except Exception as err:
                failed_videos.append({
                    "video": str(video_path),
                    "video_name": video_path.name,
                    "class": class_name,
                    "error": str(err),
                })
                print(
                    f"[{c_idx}/{len(class_dirs)}] {class_name:<20} | "
                    f"{video_path.name:<24} | "
                    f"[FAILED] Error: {err}"
                )

        if limit_total and (len(processed_videos) + len(skipped_videos)) >= limit_total:
            break

    elapsed = time.time() - start_time
    print("\n" + "=" * 76)
    print("                     EXTRACTION SUMMARY REPORT")
    print("=" * 76)
    print(f"  Total Scanned : {total_scanned} videos")
    print(f"  Processed     : {len(processed_videos)} videos successfully")
    print(f"  Skipped       : {len(skipped_videos)} videos (valid existing .npz)")
    print(f"  Failed        : {len(failed_videos)} videos")
    print(f"  Time Taken    : {elapsed:.2f} seconds")

    if failed_videos:
        print("\n  FAILED VIDEOS DETAIL:")
        for f in failed_videos:
            print(f"    - [{f['class']}] {f['video_name']}: {f['error']}")
    else:
        print("\n  Status: All processed videos completed with zero failures.")
    print("=" * 76 + "\n")

    return {
        "scanned": total_scanned,
        "processed": len(processed_videos),
        "skipped": len(skipped_videos),
        "failed": len(failed_videos),
        "failed_details": failed_videos,
        "elapsed_sec": round(elapsed, 2),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks from dataset videos.")
    parser.add_argument("--limit", type=int, default=None, help="Limit total videos to extract (for testing).")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Limit videos extracted per class.")
    parser.add_argument("--class-name", type=str, default=None, help="Extract only for a specific class folder.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract videos even if output .npz exists.")
    args = parser.parse_args()

    extract_all(
        limit_total=args.limit,
        limit_per_class=args.limit_per_class,
        target_class=args.class_name,
        overwrite=args.overwrite,
    )
