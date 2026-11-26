import numpy as np
from pathlib import Path

def print_detection_counts(npz_name):
    p = Path("ai-training/data/raw_landmarks/17.friend") / npz_name
    d = np.load(p)
    mask = d["presence_mask"] # (N, 2)
    counts = np.sum(mask, axis=1) # 0, 1, or 2 hands detected per frame
    n = len(counts)
    c0 = np.sum(counts == 0)
    c1 = np.sum(counts == 1)
    c2 = np.sum(counts == 2)
    print(f"{npz_name:<25} n={n:<3} | 0 hands: {c0:2d} ({c0/n*100:.1f}%) | 1 hand: {c1:2d} ({c1/n*100:.1f}%) | 2 hands: {c2:2d} ({c2/n*100:.1f}%)")

for f in ["0403(39).npz", "0403(40).npz", "Friend1.npz", "Friend2.npz", "KrishnarajFriend1.npz", "manideepfriend-1.npz"]:
    print_detection_counts(f)

