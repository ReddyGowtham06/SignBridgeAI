import numpy as np
import json

train_data = np.load("ai-training/data/processed/train.npz")
X_train = train_data["X"]
y_train = train_data["y"]
video_names = train_data["video_names"]

# Samples for 17.friend (class 8)
friend_idx = np.where(y_train == 8)[0]
print(f"Total training samples for 17.friend: {len(friend_idx)}")

print(f"{'Video':<26} {'LeftPres (feat 0)':<18} {'RightPres (feat 379)':<20} {'LeftMeanX (feat 1)':<18} {'RightMeanX (feat 380)'}")
print("-" * 100)

for idx in friend_idx:
    feat = X_train[idx]
    vname = video_names[idx]
    lp = feat[0]
    rp = feat[379]
    lmx = feat[1]
    rmx = feat[380]
    print(f"{vname:<26} {lp:<18.3f} {rp:<20.3f} {lmx:<18.3f} {rmx:.3f}")

