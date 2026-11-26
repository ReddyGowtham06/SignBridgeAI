import numpy as np
import json

train_data = np.load('ai-training/data/staged_phase8b5b_split/train.npz')
X_train, y_train = train_data['X'], train_data['y']

with open('frontend/public/models/sign_classifier.json', 'r') as f:
    model_json = json.load(f)
classes = model_json['labels']

for s_cls in ['21.good', '34.stop', '45.hand']:
    c_idx = classes.index(s_cls)
    indices = np.where(y_train == c_idx)[0]
    print(f"\n--- Class: {s_cls} (Indices in train: {len(indices)}) ---")
    for i in indices[:4]:
        p_l = X_train[i, 0]
        p_r = X_train[i, 379]
        print(f"Sample {i}: p_left={p_l:.3f}, p_right={p_r:.3f}")
