import cv2
from pathlib import Path
import numpy as np

video_dir = Path(r"D:\SignLang\sign project\17.friend")
videos = ["0403(39).mp4", "Friend1.mp4", "KrishnarajFriend1.mp4", "manideepfriend-1.mp4"]

for vname in videos:
    vp = video_dir / vname
    cap = cv2.VideoCapture(str(vp))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Inspect 5 evenly spaced frames
    indices = [int(n * p) for p in [0.2, 0.4, 0.5, 0.6, 0.8]]
    print(f"\nVideo: {vname} (Frames: {n}, FPS: {fps:.1f})")

    for f_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: continue
        h, w, _ = frame.shape
        # Center region brightness & color
        center_crop = frame[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
        print(f"  Frame {f_idx:2d}: shape={w}x{h}, center_mean_bgr={center_crop.mean(axis=(0,1)).round(1).tolist()}")

    cap.release()

