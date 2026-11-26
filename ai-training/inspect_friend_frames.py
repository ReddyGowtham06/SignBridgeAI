import cv2
from pathlib import Path
import numpy as np

video_dir = Path(r"D:\SignLang\sign project\17.friend")

for vname in ["0403(39).mp4", "Friend1.mp4", "KrishnarajFriend1.mp4", "manideepfriend-1.mp4"]:
    vp = video_dir / vname
    cap = cv2.VideoCapture(str(vp))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo {vname} (total frames: {n}):")

    # Sample at 25%, 50%, 75%
    for pct in [0.25, 0.50, 0.75]:
        f_idx = int(n * pct)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            # Analyze brightness and hand region
            h, w, _ = frame.shape
            print(f"  Frame {f_idx} ({int(pct*100)}%): shape={w}x{h}, mean_rgb={frame.mean(axis=(0,1)).round(1).tolist()}")

    cap.release()

