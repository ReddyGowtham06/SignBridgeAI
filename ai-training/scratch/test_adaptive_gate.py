import numpy as np
import json
import glob
import os

print("=== IMPLEMENTING & EVALUATING ADAPTIVE MOTION GATE IN PYTHON ===")

# Load models and split datasets
with open('frontend/public/models/sign_classifier.json', 'r') as f:
    model_json = json.load(f)

classes = model_json['labels']
W = np.array(model_json['W'])
b = np.array(model_json['b'])
scaler_mean = np.array(model_json['scaler_mean'])
scaler_scale = np.array(model_json['scaler_scale'])

STATIC_VOCABULARY = ['21.good', '34.stop', '45.hand']
STATIC_CLASS_INDICES = [classes.index(c) for c in STATIC_VOCABULARY]

# Threshold configurations
DEFAULT_PRESENCE_THRESHOLD = 0.20
DEFAULT_MOTION_THRESHOLD = 0.05
DEFAULT_CONFIDENCE_THRESHOLD = 0.45

STATIC_MIN_PRESENCE = 0.50
STATIC_MAX_POSE_STD = 0.055
STATIC_MIN_CONFIDENCE = 0.50
STATIC_MIN_MARGIN = 0.10
STATIC_MAX_Y_CENTROID = 0.75  # Must be above bottom 25% of frame (signing zone)

def evaluate_sample(features, apply_adaptive=True):
    """
    Evaluates a single 758-D feature vector.
    Returns:
        pred_label: str or None
        conf: float
        status: 'accepted' or 'rejected'
        reason: str
        path_taken: 'dynamic_path' or 'static_path' or 'none'
    """
    p_left = features[0]
    p_right = features[379]
    
    # Check Gate 1: Presence
    if p_left < DEFAULT_PRESENCE_THRESHOLD and p_right < DEFAULT_PRESENCE_THRESHOLD:
        return None, 0.0, 'rejected', 'presence_too_low', 'none'
        
    # Velocity calculation
    vel_left = features[253:316].mean()
    vel_right = features[632:695].mean()
    motion_energy = max(vel_left if p_left > 0 else 0.0, vel_right if p_right > 0 else 0.0)
    
    # Handshape spatial standard deviation (ddof=0)
    std_left = features[64:127].mean()
    std_right = features[443:506].mean()
    spatial_std = max(std_left if p_left > 0 else 0.0, std_right if p_right > 0 else 0.0)
    
    # Active hand vertical centroid (y coordinate)
    # Left wrist y is index 2, right wrist y is index 381
    # Mean of all y coords for left hand: indices 2, 5, 8 ... 62
    y_mean_left = features[2:64:3].mean() if p_left > 0 else 1.0
    y_mean_right = features[381:443:3].mean() if p_right > 0 else 1.0
    y_centroid = min(y_mean_left if p_left > 0 else 1.0, y_mean_right if p_right > 0 else 1.0)
    
    # Standardize & compute logits
    x_std = (features - scaler_mean) / scaler_scale
    logits = np.dot(x_std, W) + b
    # Stable softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    top_indices = np.argsort(probs)[::-1]
    top1_idx = top_indices[0]
    top2_idx = top_indices[1]
    top1_conf = float(probs[top1_idx])
    top2_conf = float(probs[top2_idx])
    margin = top1_conf - top2_conf
    top1_label = classes[top1_idx]
    
    # PATH A: Standard dynamic motion threshold satisfied
    if motion_energy >= DEFAULT_MOTION_THRESHOLD:
        if top1_conf < DEFAULT_CONFIDENCE_THRESHOLD:
            return None, top1_conf, 'rejected', 'confidence_low', 'dynamic_path'
        return top1_label, top1_conf, 'accepted', 'none', 'dynamic_path'
        
    # PATH B: Motion is low (motion_energy < 0.05)
    if not apply_adaptive:
        # Original M1 Behavior: immediately reject low motion
        return None, top1_conf, 'rejected', 'motion_stationary', 'original_m1_gate'
        
    # --- ADAPTIVE MOTION GATE (PATH B) ---
    # Condition B1: Adequate presence of active hand
    active_presence = max(p_left, p_right)
    if active_presence < STATIC_MIN_PRESENCE:
        return None, top1_conf, 'rejected', 'static_presence_low', 'static_path'
        
    # Condition B2: Stable handshape evidence (low spatial standard deviation)
    if spatial_std > STATIC_MAX_POSE_STD:
        return None, top1_conf, 'rejected', 'static_pose_jitter', 'static_path'
        
    # Condition B3: Hand is inside the active signing space (not resting at bottom)
    if y_centroid > STATIC_MAX_Y_CENTROID:
        return None, top1_conf, 'rejected', 'idle_hand_resting', 'static_path'
        
    # Condition B4: Candidate class MUST belong to static vocabulary
    if top1_idx not in STATIC_CLASS_INDICES:
        # Stationary hand predicting a dynamic sign is an invalid state -> reject
        return None, top1_conf, 'rejected', 'motion_stationary', 'static_path'
        
    # Condition B5: High confidence and distinct classification margin
    if top1_conf < STATIC_MIN_CONFIDENCE or margin < STATIC_MIN_MARGIN:
        return None, top1_conf, 'rejected', 'static_confidence_ambiguous', 'static_path'
        
    # All static-sign criteria passed!
    return top1_label, top1_conf, 'accepted', 'none', 'static_path'

print("Adaptive gate evaluation function compiled successfully.")
