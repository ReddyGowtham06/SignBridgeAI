import numpy as np
import glob

files = glob.glob('ai-training/data/staged_clean_landmarks/21.good/*.npz')
print("21.good files:", files[:3])
if files:
    d = np.load(files[0], allow_pickle=True)
    print("Keys in clean landmark file:", list(d.keys()))
    for k in d.keys():
        arr = d[k]
        print(f"  {k}: shape {arr.shape}, dtype {arr.dtype}")
