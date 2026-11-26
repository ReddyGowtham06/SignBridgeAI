import numpy as np
from pathlib import Path

def analyze_coords(npz_name):
    p = Path("ai-training/data/raw_landmarks/17.friend") / npz_name
    d = np.load(p)
    lm = d["landmarks"] # (N, 2, 21, 3)
    mask = d["presence_mask"] # (N, 2)
    handedness = d["handedness"]

    print(f"\nAnalyzing {npz_name}:")
    for i in range(len(mask)):
        if mask[i, 0] and mask[i, 1]:
            # Both hands detected
            w0 = lm[i, 0, 0] # wrist slot 0
            w1 = lm[i, 1, 0] # wrist slot 1
            dist = np.linalg.norm(w0 - w1)
            print(f"  Frame {i:2d} (BOTH): Slot0 (Left) wrist=({w0[0]:.2f}, {w0[1]:.2f}) | Slot1 (Right) wrist=({w1[0]:.2f}, {w1[1]:.2f}) | wrist_dist={dist:.2f} | raw={handedness[i]}")
        elif mask[i, 0]:
            w0 = lm[i, 0, 0]
            print(f"  Frame {i:2d} (SLOT 0 ONLY): wrist=({w0[0]:.2f}, {w0[1]:.2f}) | raw={handedness[i]}")
        elif mask[i, 1]:
            w1 = lm[i, 1, 0]
            print(f"  Frame {i:2d} (SLOT 1 ONLY): wrist=({w1[0]:.2f}, {w1[1]:.2f}) | raw={handedness[i]}")

analyze_coords("0403(39).npz")
analyze_coords("Friend1.npz")

