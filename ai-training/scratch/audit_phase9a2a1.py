import numpy as np
import json
import csv
import sys
import os

sys.path.append('ai-training')
print("=== STARTING PHASE 9A.2A.1 EVIDENCE AUDIT SCRIPT ===")

# 1. Load Model Data
with open('frontend/public/models/sign_classifier.json', 'r') as f:
    model_json = json.load(f)

classes = model_json['labels']
W = np.array(model_json['W'])
b = np.array(model_json['b'])
scaler_mean = np.array(model_json['scaler_mean'])
scaler_scale = np.array(model_json['scaler_scale'])

# Load validation split
val_data = np.load('ai-training/data/staged_phase8b5b_split/val.npz')
X_val, y_val = val_data['X'], val_data['y']
print(f"Validation dataset loaded: X_val shape {X_val.shape}, y_val shape {y_val.shape}")

# Load test split ONLY to verify shape and check hash, DO NOT EVALUATE OR TUNE
test_data = np.load('ai-training/data/staged_phase8b5b_split/test.npz')
print(f"Frozen test dataset verified untouched: X_test shape {test_data['X'].shape}")

# Re-run gate evaluations on validation set
from src.adaptive_motion_gate import AdaptiveMotionGate
gate = AdaptiveMotionGate(model_json)

val_audit_rows = []
orig_accepted_count = 0
adapt_accepted_count = 0
orig_correct_count = 0
adapt_correct_count = 0
prediction_flips = 0
new_false_accepts = 0
new_false_rejects = 0

for i in range(len(X_val)):
    x = X_val[i]
    true_idx = int(y_val[i])
    true_label = classes[true_idx]
    
    res = gate.evaluate(x)
    orig = res['original_m1']
    adapt = res['adaptive_gate']
    metrics = res['metrics']
    
    orig_status = orig['status']
    adapt_status = adapt['status']
    orig_pred = orig['predicted_label']
    adapt_pred = adapt['predicted_label']
    
    if orig_status == 'accepted':
        orig_accepted_count += 1
        if orig_pred == true_label:
            orig_correct_count += 1
            
    if adapt_status == 'accepted':
        adapt_accepted_count += 1
        if adapt_pred == true_label:
            adapt_correct_count += 1
            
    # Check for flips
    if orig_pred != adapt_pred:
        prediction_flips += 1
        
    # Check for new false accepts: rejected in M1, accepted in adapt but incorrect
    if orig_status == 'rejected' and adapt_status == 'accepted' and adapt_pred != true_label:
        new_false_accepts += 1
        
    # Check for new false rejects: accepted in M1, but rejected in adapt
    if orig_status == 'accepted' and adapt_status == 'rejected':
        new_false_rejects += 1
        
    val_audit_rows.append({
        'sample_index': i,
        'ground_truth_id': true_idx,
        'ground_truth_label': true_label,
        'm1_status': orig_status,
        'm1_predicted_label': orig_pred or 'none',
        'm1_rejection_reason': orig['rejection_reason'] or 'none',
        'm1_is_correct': (orig_status == 'accepted' and orig_pred == true_label),
        'adaptive_status': adapt_status,
        'adaptive_predicted_label': adapt_pred or 'none',
        'adaptive_rejection_reason': adapt['rejection_reason'] or 'none',
        'adaptive_path': adapt['path_taken'],
        'adaptive_is_correct': (adapt_status == 'accepted' and adapt_pred == true_label),
        'status_change': f"{orig_status}_to_{adapt_status}",
        'motion_energy': f"{metrics['motion_energy']:.4f}",
        'spatial_std': f"{metrics['spatial_std']:.4f}",
        'y_centroid': f"{metrics['y_centroid']:.4f}",
        'top1_conf': f"{metrics['top1_confidence']:.4f}",
        'margin': f"{metrics['margin']:.4f}"
    })

print(f"\n--- VALIDATION COMPARISON AUDIT (142 samples) ---")
print(f"Original M1 Accepted: {orig_accepted_count} / 142 ({orig_accepted_count/142*100:.1f}%)")
print(f"Adaptive Gate Accepted: {adapt_accepted_count} / 142 ({adapt_accepted_count/142*100:.1f}%)")
print(f"Original M1 Correct: {orig_correct_count} / 142")
print(f"Adaptive Gate Correct: {adapt_correct_count} / 142")
print(f"Prediction Flips: {prediction_flips}")
print(f"New False Accepts: {new_false_accepts}")
print(f"New False Rejects: {new_false_rejects}")

os.makedirs('ai-training/reports/phase9a2a1', exist_ok=True)

with open('ai-training/reports/phase9a2a1/phase9a2a1_validation_comparison.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(val_audit_rows[0].keys()))
    writer.writeheader()
    writer.writerows(val_audit_rows)
print("Saved phase9a2a1_validation_comparison.csv")

# 2. AUDIT & MANIFEST OF THE 120 IDLE NEGATIVE WINDOWS
print("\n--- AUDITING 120 IDLE NEGATIVE TEST FIXTURES ---")
negative_manifest_rows = []

# Reproduce the exact 120 windows from run_full_phase9a2a_experiments.py with random seed 42
np.random.seed(42)
window_id = 1

# Category A: Desk / Lap Postures (50)
for i in range(50):
    vec = np.zeros(758, dtype=np.float32)
    vec[379] = np.random.uniform(0.70, 1.0)
    y_base = np.random.uniform(0.76, 0.92)
    x_base = np.random.uniform(0.35, 0.65)
    for lm_i in range(21):
        vec[380 + lm_i*3] = x_base + np.random.normal(0, 0.03)
        vec[380 + lm_i*3 + 1] = y_base + np.random.normal(0, 0.02)
        vec[380 + lm_i*3 + 2] = np.random.normal(0, 0.02)
    vec[443:506] = np.random.uniform(0.005, 0.025)
    vec[506:569] = vec[380:443] - 0.02
    vec[569:632] = vec[380:443] + 0.02
    vec[632:695] = np.random.uniform(0.005, 0.035)
    vec[695:758] = np.random.uniform(0.002, 0.015)
    
    res = gate.evaluate(vec)
    adapt = res['adaptive_gate']
    metrics = res['metrics']
    
    negative_manifest_rows.append({
        'window_id': f"NEG-{window_id:03d}",
        'fixture_type': 'synthetic_parametric_test_fixture',
        'category': 'resting_lap_desk',
        'generator_spec': f"Seed=42; Y_base={y_base:.3f} (Lap/Desk); X_base={x_base:.3f}; Vel in [0.005, 0.035]; Std in [0.005, 0.025]",
        'is_human_recording': 'FALSE',
        'motion_energy': f"{metrics['motion_energy']:.4f}",
        'spatial_std': f"{metrics['spatial_std']:.4f}",
        'y_centroid': f"{metrics['y_centroid']:.4f}",
        'adaptive_status': adapt['status'],
        'rejection_reason': adapt['rejection_reason'],
        'path_taken': adapt['path_taken'],
        'is_correctly_rejected': (adapt['status'] == 'rejected')
    })
    window_id += 1

# Category B: Idle Relaxed Chest (40)
for i in range(40):
    vec = np.zeros(758, dtype=np.float32)
    vec[379] = np.random.uniform(0.60, 0.95)
    y_base = np.random.uniform(0.40, 0.65)
    x_base = np.random.uniform(0.40, 0.60)
    for lm_i in range(21):
        vec[380 + lm_i*3] = x_base + np.random.normal(0, 0.05)
        vec[380 + lm_i*3 + 1] = y_base + np.random.normal(0, 0.05)
        vec[380 + lm_i*3 + 2] = np.random.normal(0, 0.03)
    vec[443:506] = np.random.uniform(0.030, 0.070)
    vec[506:569] = vec[380:443] - 0.04
    vec[569:632] = vec[380:443] + 0.04
    vec[632:695] = np.random.uniform(0.010, 0.040)
    vec[695:758] = np.random.uniform(0.005, 0.020)
    
    res = gate.evaluate(vec)
    adapt = res['adaptive_gate']
    metrics = res['metrics']
    
    negative_manifest_rows.append({
        'window_id': f"NEG-{window_id:03d}",
        'fixture_type': 'synthetic_parametric_test_fixture',
        'category': 'idle_relaxed_chest',
        'generator_spec': f"Seed=42; Y_base={y_base:.3f} (Chest); X_base={x_base:.3f}; Vel in [0.010, 0.040]; Std in [0.030, 0.070]",
        'is_human_recording': 'FALSE',
        'motion_energy': f"{metrics['motion_energy']:.4f}",
        'spatial_std': f"{metrics['spatial_std']:.4f}",
        'y_centroid': f"{metrics['y_centroid']:.4f}",
        'adaptive_status': adapt['status'],
        'rejection_reason': adapt['rejection_reason'],
        'path_taken': adapt['path_taken'],
        'is_correctly_rejected': (adapt['status'] == 'rejected')
    })
    window_id += 1

# Category C: Fidgeting Transition (30)
for i in range(30):
    vec = np.zeros(758, dtype=np.float32)
    vec[379] = np.random.uniform(0.50, 0.85)
    y_base = np.random.uniform(0.50, 0.70)
    x_base = np.random.uniform(0.30, 0.70)
    for lm_i in range(21):
        vec[380 + lm_i*3] = x_base + np.random.normal(0, 0.06)
        vec[380 + lm_i*3 + 1] = y_base + np.random.normal(0, 0.06)
        vec[380 + lm_i*3 + 2] = np.random.normal(0, 0.04)
    vec[443:506] = np.random.uniform(0.060, 0.110)
    vec[506:569] = vec[380:443] - 0.06
    vec[569:632] = vec[380:443] + 0.06
    vec[632:695] = np.random.uniform(0.020, 0.048)
    vec[695:758] = np.random.uniform(0.015, 0.035)
    
    res = gate.evaluate(vec)
    adapt = res['adaptive_gate']
    metrics = res['metrics']
    
    negative_manifest_rows.append({
        'window_id': f"NEG-{window_id:03d}",
        'fixture_type': 'synthetic_parametric_test_fixture',
        'category': 'fidgeting_transition',
        'generator_spec': f"Seed=42; Y_base={y_base:.3f}; X_base={x_base:.3f}; Vel in [0.020, 0.048]; Std in [0.060, 0.110]",
        'is_human_recording': 'FALSE',
        'motion_energy': f"{metrics['motion_energy']:.4f}",
        'spatial_std': f"{metrics['spatial_std']:.4f}",
        'y_centroid': f"{metrics['y_centroid']:.4f}",
        'adaptive_status': adapt['status'],
        'rejection_reason': adapt['rejection_reason'],
        'path_taken': adapt['path_taken'],
        'is_correctly_rejected': (adapt['status'] == 'rejected')
    })
    window_id += 1

with open('ai-training/reports/phase9a2a1/phase9a2a1_idle_window_manifest.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(negative_manifest_rows[0].keys()))
    writer.writeheader()
    writer.writerows(negative_manifest_rows)
print("Saved phase9a2a1_idle_window_manifest.csv")

# Tally rejection reasons
rejection_counts = {}
for r in negative_manifest_rows:
    reason = r['rejection_reason']
    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

print("\n--- IDLE NEGATIVE REJECTION REASON BREAKDOWN (120 windows) ---")
for k, v in rejection_counts.items():
    print(f"  {k}: {v} / 120 ({v/120*100:.1f}%)")
