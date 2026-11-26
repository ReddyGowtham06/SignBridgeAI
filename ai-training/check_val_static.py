import numpy as np
import json

val = np.load('ai-training/data/processed/val.npz')
X, y = val['X'], val['y']
with open('frontend/public/models/sign_classifier.json') as f:
    m = json.load(f)

W, b, mean, scale = np.array(m['W']), np.array(m['b']), np.array(m['scaler_mean']), np.array(m['scaler_scale'])

for idx, label in [(19, '21.good'), (26, '34.stop'), (8, '45.hand')]:
    feat = X[idx].copy()
    # scale velocity
    feat[253:316] *= 0.15
    feat[632:695] *= 0.15
    feat[316:379] *= 0.15
    feat[695:758] *= 0.15
    vel = max(feat[0] * feat[253:316].mean(), feat[379] * feat[632:695].mean())
    z = (feat - mean) / scale
    logits = b + z @ W
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    top1 = probs.argmax()
    print(f'{label}: vel={vel:.4f} pred={m["labels"][top1]} conf={probs.max():.3f}')
