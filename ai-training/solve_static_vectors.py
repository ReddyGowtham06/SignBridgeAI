import numpy as np
import json

with open('frontend/public/models/sign_classifier.json') as f:
    m = json.load(f)

W = np.array(m['W'], dtype=np.float64) # (758, 49)
b = np.array(m['b'], dtype=np.float64)
mean = np.array(m['scaler_mean'], dtype=np.float64)
scale = np.array(m['scaler_scale'], dtype=np.float64)

# We want fixed indices:
# presence: feat[0] = 0, feat[379] = 0.95
# vel: feat[253..315] = 0.0, feat[632..694] = 0.02
# std: feat[64..126] = 0.0, feat[443..505] = 0.03 (scaled properly)
# y: feat[381..441:3] = 0.50

fixed_indices = (
    [0, 379] +
    list(range(253, 316)) + list(range(632, 695)) +
    list(range(64, 127)) + list(range(443, 506)) +
    list(range(2, 64, 3)) + list(range(381, 443, 3))
)
fixed_indices = sorted(list(set(fixed_indices)))
free_indices = [i for i in range(758) if i not in fixed_indices]

targets = {'21.good': 13, '34.stop': 27, '45.hand': 39}
results = {}

for label, target_idx in targets.items():
    # Set default values for fixed features
    feat = mean.copy()
    feat[0] = 0.0
    feat[379] = 0.95
    feat[253:316] = 0.0
    feat[632:695] = 0.02
    feat[64:127] = mean[64:127] * 0.5
    feat[443:506] = mean[443:506] * 0.5
    feat[2:64:3] = 1.0
    feat[381:443:3] = 0.50
    
    z = (feat - mean) / scale
    
    # Target logits: target = +5, others = -5
    target_logits = np.full(49, -5.0, dtype=np.float64)
    target_logits[target_idx] = 5.0
    
    # We solve for z[free_indices]:
    # b + z[fixed] @ W[fixed, :] + z[free] @ W[free, :] = target_logits
    # z[free] @ W[free, :] = target_logits - b - z[fixed] @ W[fixed, :]
    rhs = target_logits - b - z[fixed_indices] @ W[fixed_indices, :]
    W_free = W[free_indices, :] # (len(free), 49)
    
    # Minimum-norm solution for z[free]:
    z_free = np.linalg.pinv(W_free.T) @ rhs
    z[free_indices] = z_free
    
    final_feat = z * scale + mean
    
    # Re-evaluate
    z_check = (final_feat - mean) / scale
    logits_check = b + z_check @ W
    probs = np.exp(logits_check - np.max(logits_check))
    probs /= probs.sum()
    top1 = np.argmax(probs)
    top2 = np.argsort(probs)[-2]
    margin = probs[top1] - probs[top2]
    
    vel = float(final_feat[632:695].mean())
    std = float(final_feat[443:506].mean())
    y_c = float(final_feat[381:443:3].mean())
    
    print(f"{label}: top1={m['labels'][top1]} conf={probs[top1]:.3f} margin={margin:.3f} vel={vel:.3f} std={std:.3f} y_c={y_c:.2f}")
    assert top1 == target_idx
    results[label] = final_feat.tolist()

with open('ai-training/data/synthetic_static_vectors.json', 'w') as f:
    json.dump(results, f)
print("SUCCESS: Exact static vectors generated with exact target satisfaction!")
