import numpy as np
import json

print("=== LOADING TRAIN AND VAL NPZ ARTIFACTS ===")
train_data = np.load('ai-training/data/staged_phase8b5b_split/train.npz')
val_data = np.load('ai-training/data/staged_phase8b5b_split/val.npz')

X_train, y_train = train_data['X'], train_data['y']
X_val, y_val = val_data['X'], val_data['y']

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

# Load model data
with open('frontend/public/models/sign_classifier.json', 'r') as f:
    model_json = json.load(f)

classes = model_json['labels']
W = np.array(model_json['W'])
b = np.array(model_json['b'])
scaler_mean = np.array(model_json['scaler_mean'])
scaler_scale = np.array(model_json['scaler_scale'])

# Feature indices
# Left hand: 0-378. Presence: 0. Spatial std: 64:127. Vel: 253:316. Vel std: 316:379.
# Right hand: 379-757. Presence: 379. Spatial std: 443:506. Vel: 632:695. Vel std: 695:758.

def compute_metrics(X):
    p_left = X[:, 0]
    p_right = X[:, 379]
    vel_left = X[:, 253:316].mean(axis=1)
    vel_right = X[:, 632:695].mean(axis=1)
    motion_energy = np.maximum(
        np.where(p_left > 0, vel_left, 0),
        np.where(p_right > 0, vel_right, 0)
    )
    # Spatial std across landmarks
    std_left = X[:, 64:127].mean(axis=1)
    std_right = X[:, 443:506].mean(axis=1)
    # Handshape spatial variance for active hand(s)
    spatial_variance = np.maximum(
        np.where(p_left > 0, std_left, 0),
        np.where(p_right > 0, std_right, 0)
    )
    return p_left, p_right, motion_energy, spatial_variance

# Combine train and val for exploratory analysis (keeping test frozen!)
X_pool = np.vstack([X_train, X_val])
y_pool = np.concatenate([y_train, y_val])

p_left, p_right, motion_energy, spatial_variance = compute_metrics(X_pool)

target_classes = ['21.good', '34.stop', '45.hand', '17.friend', '9.hello', '10.youareperfect']

print("\n=== CLASS MOTION & POSE STABILITY STATISTICS (Train + Val Pool: 773 samples) ===")
print(f"{'Class Label':<20} | {'Count':<5} | {'Mean Motion':<12} | {'Min Motion':<10} | {'Max Motion':<10} | {'Mean Spatial Std':<15} | {'Motion < 0.05'}")
print("-" * 95)

for c_idx, c_name in enumerate(classes):
    mask = (y_pool == c_idx)
    count = np.sum(mask)
    if count == 0:
        continue
    c_motion = motion_energy[mask]
    c_std = spatial_variance[mask]
    low_motion_count = np.sum(c_motion < 0.05)
    
    if c_name in target_classes or low_motion_count > 0:
        print(f"{c_name:<20} | {count:<5} | {c_motion.mean():<12.5f} | {c_motion.min():<10.5f} | {c_motion.max():<10.5f} | {c_std.mean():<15.5f} | {low_motion_count}/{count} ({low_motion_count/count*100:.1f}%)")

print("\n=== SIMULATING RAW M1 CLASSIFICATION ON 21.good, 34.stop, 45.hand (Train + Val) ===")
# Standardize features
X_pool_std = (X_pool - scaler_mean) / scaler_scale
logits = np.dot(X_pool_std, W) + b
# Softmax
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
preds = np.argmax(probs, axis=1)
confs = np.max(probs, axis=1)

for tgt in ['21.good', '34.stop', '45.hand']:
    c_idx = classes.index(tgt)
    mask = (y_pool == c_idx)
    sample_indices = np.where(mask)[0]
    print(f"\n--- Class: {tgt} (Index: {c_idx}, N={len(sample_indices)}) ---")
    correct_raw = np.sum(preds[mask] == c_idx)
    print(f"Raw M1 Accuracy (no gates): {correct_raw}/{len(sample_indices)} ({correct_raw/len(sample_indices)*100:.1f}%)")
    for s_idx in sample_indices:
        pred_label = classes[preds[s_idx]]
        conf = confs[s_idx]
        mot = motion_energy[s_idx]
        std = spatial_variance[s_idx]
        p_l = p_left[s_idx]
        p_r = p_right[s_idx]
        g1_pass = (p_l >= 0.20 or p_r >= 0.20)
        g2_pass = (mot >= 0.05)
        g3_pass = (conf >= 0.45)
        print(f"  Sample #{s_idx:3d} | Pred: {pred_label:<15} (conf={conf:.3f}) | Motion={mot:.4f} | Std={std:.4f} | G1(Pres)={g1_pass} | G2(Mot)={g2_pass} | G3(Conf)={g3_pass}")
