import json
from pathlib import Path
import numpy as np

p = Path("ai-training/data/raw_landmarks/17.friend")
files = sorted(list(p.glob("*.npz")))
print(f"Total landmark files for 17.friend: {len(files)}")

print(f"{'Filename':<26} {'Frames':<8} {'LeftPres':<10} {'RightPres':<10} {'BothPres':<10} {'DurationSec':<12}")
print("-" * 80)

stats = []
for f in files:
    d = np.load(f)
    lm = d["landmarks"]
    mask = d["presence_mask"]
    ts = d["timestamps_ms"]
    handedness = d["handedness"]
    n_frames = len(ts)
    l_pres = float(np.mean(mask[:, 0]))
    r_pres = float(np.mean(mask[:, 1]))
    both_pres = float(np.mean(mask[:, 0] & mask[:, 1]))
    dur = float((ts[-1] - ts[0]) / 1000.0) if n_frames > 1 else 0.0
    print(f"{f.name:<26} {n_frames:<8} {l_pres:<10.3f} {r_pres:<10.3f} {both_pres:<10.3f} {dur:<12.2f}")
    stats.append({
        "name": f.name,
        "frames": n_frames,
        "l_pres": l_pres,
        "r_pres": r_pres,
        "both_pres": both_pres,
        "dur": dur,
    })

avg_l = float(np.mean([s["l_pres"] for s in stats]))
avg_r = float(np.mean([s["r_pres"] for s in stats]))
avg_both = float(np.mean([s["both_pres"] for s in stats]))
print("-" * 80)
print(f"Average Left Presence:  {avg_l:.3f}")
print(f"Average Right Presence: {avg_r:.3f}")
print(f"Average Both Present:   {avg_both:.3f}")

