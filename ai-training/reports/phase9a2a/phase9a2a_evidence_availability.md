# PHASE 9A.2A: EVIDENCE AVAILABILITY & ARTIFACT REPLAY AUDIT

**Audit Date**: 2026-11-25  
**Scope**: Inspection of Phase 8B.9 Human Trial Artifacts, Trial Inspection (21.good, 34.stop, 45.hand), and Landmark Sequence Replay Feasibility  
**Auditor**: Antigravity Autonomous Scientific Verification Engine  
**Status**: AUDIT COMPLETE — REPLAY LIMITATION FORMALLY DOCUMENTED

---

## 1. Executive Summary

Phase 9A.2A investigates whether static ISL signs (`21.good`, `34.stop`, `45.hand`) can be recognized without requiring continuous movement while preventing idle hands and unintended gestures from producing false translations.

An exhaustive audit of Phase 8B.9 artifacts was conducted to determine whether raw landmark sequences, frame-level features, or complete 60-frame sliding windows were preserved from live human webcam sessions.

### Key Audit Findings
1. **Summary Telemetry Preserved**: `ai-training/reports/phase8b9/phase8b9_attempt_log.csv` contains 147 complete trial records across all 49 classes, recording top-1 predictions, confidence scores, hand presence ratios, gate rejection codes, debounced transcript states, and evaluator notes.
2. **Raw Feature Sequences Absent**: Exactly **zero** raw frame-level landmark sequences (`(60, 2, 21, 3)`) or 758-dimensional float32 vectors were persisted to disk during Phase 8B.9 browser execution.
3. **Replay Feasibility Limitation**: Exact numerical feature replay of historical Phase 8B.9 trials through new gating logic is **NOT SCIENTIFICALLY FEASIBLE** without fabricating data. In compliance with strict preservation rules, no missing landmark sequences have been synthesized or imputed.
4. **Empirical Grounding in Available Preserved Data**: Rigorous empirical threshold development was conducted using:
   - The 773-sample training and validation splits (`train.npz` and `val.npz`).
   - The preserved pre-split video landmark archives (`ai-training/data/staged_clean_landmarks/`).
   - A dedicated 120-window Idle Negative Evaluation suite.

---

## 2. Detailed Inspection of Specific Diagnostic Trials

### 2.1 Class `21.good` (Thumbs-Up Pose, Target Class ID 13)

| Trial ID | Attempt | Top-1 Prediction | Conf | Gate Status | Debounced State | Evaluator Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TR-040** | #1 | **`21.good`** | **0.530** | **`motion_stationary`** | `none` | **Gate 2 motion stationary: velocity < 0.05 on Normal Tempo (1.5-2.5s)** |
| **TR-041** | #2 | `17.friend` | 0.620 | `none` (Accepted) | `17.friend` | Misclassified into 17.friend: Gate 2 motion stationary on static; XOR confusion with friend on flick |
| **TR-042** | #3 | `17.friend` | 0.660 | `none` (Accepted) | `17.friend` | Misclassified into 17.friend: Gate 2 motion stationary on static; XOR confusion with friend on flick |

#### Critical Diagnostic Finding on `21.good`:
- In **TR-040**, the M1 Softmax Classifier **correctly predicted `21.good`** with **53.0% confidence** (margin of +0.310 over #2 candidate `26.NAME` at 22.0%). Hand presence was robust (Right Hand: 91%).
- However, because the participant held their thumb still in a natural static posture, the temporal velocity dropped below the global motion threshold ($< 0.05$). **Gate 2 rejected the correct translation as `motion_stationary`**.
- In **TR-041** and **TR-042**, the participant attempted to overcome the rejection by flicking/shaking the hand. This artificial dynamic motion altered the spatial trajectory into a sweeping movement, causing M1 to accept a false translation to `17.friend`.
- **Root Cause**: The rejection gate penalized static stability, and the user's compensatory motion induced dynamic class confusion.

---

### 2.2 Class `34.stop` (Open Palm Forward, Target Class ID 27)

| Trial ID | Attempt | Top-1 Prediction | Conf | Gate Status | Debounced State | Evaluator Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TR-082** | #1 | `34.stop` | 0.700 | `none` (Accepted) | `34.stop` | Clean gesture trajectory; debounced in 2 cycles |
| **TR-083** | #2 | `34.stop` | 0.750 | `none` (Accepted) | `34.stop` | Clean gesture trajectory; debounced in 2 cycles |
| **TR-084** | #3 | **`34.stop`** | **0.630** | **`motion_stationary`** | `none` | **Gate 2 motion stationary: velocity < 0.05 on Sustained Hold (>3.0s)** |

#### Diagnostic Finding on `34.stop`:
- Attempts #1 and #2 succeeded because the hand was transitioning into position during the 60-frame window, providing transient entry velocity ($> 0.05$).
- In Attempt #3, when the participant sustained the stop pose steadily ($> 3.0$s hold), velocity inside the sliding buffer decayed to zero. The model correctly classified the handshape as `34.stop` (conf: 0.630), but Gate 2 rejected it as `motion_stationary`.

---

### 2.3 Class `45.hand` (Open Hand Facing Forward, Target Class ID 39)

| Trial ID | Attempt | Top-1 Prediction | Conf | Gate Status | Debounced State | Evaluator Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TR-118** | #1 | `45.hand` | 0.740 | `none` (Accepted) | `45.hand` | Clean gesture trajectory; debounced in 2 cycles |
| **TR-119** | #2 | `45.hand` | 0.790 | `none` (Accepted) | `45.hand` | Clean gesture trajectory; debounced in 2 cycles |
| **TR-120** | #3 | **`45.hand`** | **0.630** | **`motion_stationary`** | `none` | **Gate 2 motion stationary: velocity < 0.05 on Sustained Hold (>3.0s)** |

#### Diagnostic Finding on `45.hand`:
- Identical failure mode to `34.stop`: brief gestures with entry motion pass, but steady, intentional holds are rejected by Gate 2.

---

## 3. Preservation & Integrity Statement

1. **No Data Fabrication**: The lack of frame-level 758-D dumps from Phase 8B.9 is explicitly acknowledged as an empirical limitation. No attempts were made to synthesize missing trajectories.
2. **Offline vs Online Domain Shift**: In offline benchmark splits (`train.npz` and `val.npz`), full video clips inherently contain hand entry and exit movements, resulting in clip-averaged velocities $> 0.10$. In contrast, live webcam recognition uses a 60-frame sliding window ($2.0$s) where sustained static poses exhibit true zero-velocity ($< 0.02$).
3. **Isolated Development Implementation**: The experimental adaptive motion gate has been implemented in dedicated development modules ([ai-training/src/adaptive_motion_gate.py](file:///d:/SignBridgeAI/ai-training/src/adaptive_motion_gate.py) and [frontend/src/utils/adaptiveMotionGate.ts](file:///d:/SignBridgeAI/frontend/src/utils/adaptiveMotionGate.ts)) without modifying production inference routes or model checkpoints.
