import numpy as np
import json

train_data = np.load('ai-training/data/staged_phase8b5b_split/train.npz')
X_train, y_train = train_data['X'], train_data['y']

with open('frontend/public/models/sign_classifier.json', 'r') as f:
    model_json = json.load(f)
classes = model_json['labels']
W = np.array(model_json['W'])
b = np.array(model_json['b'])
scaler_mean = np.array(model_json['scaler_mean'])
scaler_scale = np.array(model_json['scaler_scale'])

# Test Sample 164 of 21.good
feat = X_train[164].copy()
print("Original Sample 164:")
print(f"p_left={feat[0]:.3f}, p_right={feat[379]:.3f}")

# Standardize & predict
x_std = (feat - scaler_mean) / scaler_scale
logits = np.dot(x_std, W) + b
probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
print(f"Original Pred: {classes[np.argmax(probs)]} (conf={np.max(probs):.4f})")

# Now set presence to 0.90 and velocity to 0.02 (live hold)
feat_live = feat.copy()
feat_live[379] = 0.90 # High right hand presence
feat_live[632:695] = 0.02 # Low velocity (static hold)
feat_live[443:506] = 0.03 # Low spatial std

x_std_live = (feat_live - scaler_mean) / scaler_scale
logits_live = np.dot(x_std_live, W) + b
probs_live = np.exp(logits_live - np.max(logits_live)) / np.sum(np.exp(logits_live - np.max(logits_live)))
top_idx = np.argmax(probs_live)
print(f"Live Hold Pred: {classes[top_idx]} (conf={probs_live[top_idx]:.4f})")
sorted_p = np.sort(probs_live)[::-1]
print(f"Top 3: {classes[np.argsort(probs_live)[::-1][0]]}:{sorted_p[0]:.3f}, {classes[np.argsort(probs_live)[::-1][1]]}:{sorted_p[1]:.3f}, {classes[np.argsort(probs_live)[::-1][2]]}:{sorted_p[2]:.3f}")
