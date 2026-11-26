import numpy as np
from pathlib import Path

def inspect_timeline(npz_name):
    p = Path("ai-training/data/raw_landmarks/17.friend") / npz_name
    d = np.load(p)
    mask = d["presence_mask"]
    handedness = d["handedness"]
    ts = d["timestamps_ms"]
    lm = d["landmarks"]
    n = len(ts)

    print(f"\nTimeline for {npz_name} ({n} frames):")
    print(f"{'Frame':<6} {'Timestamp':<10} {'Slot0 (Left)':<15} {'Slot1 (Right)':<15} {'Raw Labels'}")
    print("-" * 65)

    for i in range(n):
        l_det = mask[i, 0]
        r_det = mask[i, 1]
        l_str = "YES" if l_det else "-"
        r_str = "YES" if r_det else "-"
        raw_lbl = str(handedness[i])
        if i % 3 == 0 or l_det != mask[max(0, i-1), 0] or r_det != mask[max(0, i-1), 1]:
            print(f"{i:<6} {ts[i]:<10} {l_str:<15} {r_str:<15} {raw_lbl}")

inspect_timeline("0403(39).npz")
inspect_timeline("KrishnarajFriend1.npz")

