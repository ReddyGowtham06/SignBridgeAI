import numpy as np
import glob
import os

print("=== INSPECTING LANDMARK DYNAMICS ACROSS VIDEO SAMPLES ===")
# Look at several landmark files from different classes to see start/end frames
sample_files = glob.glob('ai-training/data/staged_clean_landmarks/*/*.npz')[:20]
for sf in sample_files:
    d = np.load(sf, allow_pickle=True)
    lms = d['landmarks'] # (T, 2, 21, 3)
    pmask = d['presence_mask'] # (T, 2)
    cls = d['class_name'].item()
    T = lms.shape[0]
    print(f"File: {os.path.basename(sf):<20} | Class: {cls:<15} | Frames: {T:3d} | Active Frames: {np.sum(pmask[:,1])}")
