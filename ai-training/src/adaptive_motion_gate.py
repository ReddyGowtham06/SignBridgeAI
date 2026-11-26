r"""
SIGNBRIDGE AI — Phase 9A.2A Adaptive Motion Gate Module (Isolated Development Code)
==================================================================================
Implements the dual-path gesture rejection gate for static vs dynamic signs:
- Path A (Dynamic-Sign Path): Preserves original M1 motion threshold (> 0.05).
- Path B (Static-Sign Candidate Path): Evaluates intentional static handshapes
  (21.good, 34.stop, 45.hand) under low motion, gated by hand presence, pose stability,
  active signing space centroid (anti-idle), and classification confidence/margin.

STRICT PRESERVATION:
- Does NOT replace production classifier or weights.
- Does NOT alter the 758-D feature schema.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np

# Designated static sign vocabulary
STATIC_VOCABULARY = ['21.good', '34.stop', '45.hand']

# Empirical Thresholds
DEFAULT_PRESENCE_THRESHOLD = 0.20
DEFAULT_MOTION_THRESHOLD = 0.05
DEFAULT_CONFIDENCE_THRESHOLD = 0.45

# Static candidate path thresholds (empirically derived on Train + Val splits)
STATIC_MIN_PRESENCE = 0.50
STATIC_MAX_POSE_STD = 0.055
STATIC_MIN_CONFIDENCE = 0.50
STATIC_MIN_MARGIN = 0.10
STATIC_MAX_Y_CENTROID = 0.75  # Must be in upper 75% of frame (chest/chin signing space)


class AdaptiveMotionGate:
    def __init__(self, model_dict: Dict[str, Any]):
        self.labels = model_dict['labels']
        self.W = np.array(model_dict['W'], dtype=np.float32)
        self.b = np.array(model_dict['b'], dtype=np.float32)
        self.scaler_mean = np.array(model_dict['scaler_mean'], dtype=np.float32)
        self.scaler_scale = np.array(model_dict['scaler_scale'], dtype=np.float32)
        
        self.static_indices = set(self.labels.index(c) for c in STATIC_VOCABULARY if c in self.labels)

    def evaluate(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates a single 758-D feature vector through both original M1 gate and adaptive gate.
        Returns dual telemetry comparing original vs adaptive decisions.
        """
        p_left = float(features[0])
        p_right = float(features[379])
        
        vel_left = float(np.mean(features[253:316]))
        vel_right = float(np.mean(features[632:695]))
        motion_energy = max(vel_left if p_left > 0 else 0.0, vel_right if p_right > 0 else 0.0)
        
        std_left = float(np.mean(features[64:127]))
        std_right = float(np.mean(features[443:506]))
        spatial_std = max(std_left if p_left > 0 else 0.0, std_right if p_right > 0 else 0.0)
        
        y_mean_left = float(np.mean(features[2:64:3])) if p_left > 0 else 1.0
        y_mean_right = float(np.mean(features[381:443:3])) if p_right > 0 else 1.0
        y_centroid = min(y_mean_left if p_left > 0 else 1.0, y_mean_right if p_right > 0 else 1.0)
        
        # M1 Standardization & Softmax Forward Pass
        x_std = (features - self.scaler_mean) / self.scaler_scale
        logits = np.dot(x_std, self.W) + self.b
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1])
        top1_conf = float(probs[top1_idx])
        margin = float(top1_conf - probs[top2_idx])
        top1_label = self.labels[top1_idx]
        
        # --- ORIGINAL M1 GATE DECISION ---
        if p_left < DEFAULT_PRESENCE_THRESHOLD and p_right < DEFAULT_PRESENCE_THRESHOLD:
            orig_status = 'rejected'
            orig_reason = 'presence_too_low'
            orig_pred = None
        elif motion_energy < DEFAULT_MOTION_THRESHOLD:
            orig_status = 'rejected'
            orig_reason = 'motion_stationary'
            orig_pred = None
        elif top1_conf < DEFAULT_CONFIDENCE_THRESHOLD:
            orig_status = 'rejected'
            orig_reason = 'confidence_low'
            orig_pred = None
        else:
            orig_status = 'accepted'
            orig_reason = None
            orig_pred = top1_label
            
        # --- ADAPTIVE MOTION GATE DECISION ---
        if p_left < DEFAULT_PRESENCE_THRESHOLD and p_right < DEFAULT_PRESENCE_THRESHOLD:
            adapt_status = 'rejected'
            adapt_reason = 'presence_too_low'
            adapt_pred = None
            adapt_path = 'none'
        elif motion_energy >= DEFAULT_MOTION_THRESHOLD:
            # Dynamic Path (Path A)
            adapt_path = 'dynamic_path'
            if top1_conf < DEFAULT_CONFIDENCE_THRESHOLD:
                adapt_status = 'rejected'
                adapt_reason = 'confidence_low'
                adapt_pred = None
            else:
                adapt_status = 'accepted'
                adapt_reason = None
                adapt_pred = top1_label
        else:
            # Static Path (Path B)
            adapt_path = 'static_path'
            active_presence = max(p_left, p_right)
            
            if active_presence < STATIC_MIN_PRESENCE:
                adapt_status = 'rejected'
                adapt_reason = 'static_presence_low'
                adapt_pred = None
            elif spatial_std > STATIC_MAX_POSE_STD:
                adapt_status = 'rejected'
                adapt_reason = 'static_pose_jitter'
                adapt_pred = None
            elif y_centroid > STATIC_MAX_Y_CENTROID:
                adapt_status = 'rejected'
                adapt_reason = 'idle_hand_resting'
                adapt_pred = None
            elif top1_idx not in self.static_indices:
                adapt_status = 'rejected'
                adapt_reason = 'motion_stationary'
                adapt_pred = None
            elif top1_conf < STATIC_MIN_CONFIDENCE or margin < STATIC_MIN_MARGIN:
                adapt_status = 'rejected'
                adapt_reason = 'static_confidence_ambiguous'
                adapt_pred = None
            else:
                adapt_status = 'accepted'
                adapt_reason = None
                adapt_pred = top1_label
                
        return {
            'metrics': {
                'presence_left': p_left,
                'presence_right': p_right,
                'motion_energy': motion_energy,
                'spatial_std': spatial_std,
                'y_centroid': y_centroid,
                'top1_label': top1_label,
                'top1_confidence': top1_conf,
                'margin': margin,
            },
            'original_m1': {
                'status': orig_status,
                'predicted_label': orig_pred,
                'rejection_reason': orig_reason
            },
            'adaptive_gate': {
                'status': adapt_status,
                'predicted_label': adapt_pred,
                'rejection_reason': adapt_reason,
                'path_taken': adapt_path
            }
        }
