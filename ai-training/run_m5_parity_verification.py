import json
import subprocess
import numpy as np
import os

OUT_DIR = "ai-training/reports/phase9a1"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load M5 checkpoint
m5 = np.load("ai-training/models/phase8b6b/m5_small_mlp.npz")
W1 = m5["W1"].astype(np.float32)
b1 = m5["b1"].astype(np.float32)
W2 = m5["W2"].astype(np.float32)
b2 = m5["b2"].astype(np.float32)
scaler_mean = m5["scaler_mean"].astype(np.float32)
scaler_scale = m5["scaler_scale"].astype(np.float32)

# 2. Load test set
test = np.load("ai-training/data/staged_phase8b5b_split/test.npz")
X_test = test["X"].astype(np.float32)
y_test = test["y"]

# Python reference forward pass
X_scaled = (X_test - scaler_mean) / scaler_scale
h1 = np.maximum(0, np.dot(X_scaled, W1) + b1)
logits_ref = np.dot(h1, W2) + b2

max_logits = np.max(logits_ref, axis=1, keepdims=True)
exp_logits = np.exp(logits_ref - max_logits)
probs_ref = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
preds_ref = np.argmax(probs_ref, axis=1)

# Export first 10 test samples and reference outputs for Node.js verification
parity_samples = []
for idx in range(min(20, len(X_test))):
    parity_samples.append({
        "sample_index": idx,
        "features": [float(x) for x in X_test[idx]],
        "py_logits": [float(x) for x in logits_ref[idx]],
        "py_probs": [float(x) for x in probs_ref[idx]],
        "py_pred": int(preds_ref[idx]),
        "gt_class": int(y_test[idx])
    })

ref_path = os.path.join(OUT_DIR, "test_samples_ref.json")
with open(ref_path, "w", encoding="utf-8") as f:
    json.dump(parity_samples, f)
print(f"Saved parity reference samples to {ref_path}")
