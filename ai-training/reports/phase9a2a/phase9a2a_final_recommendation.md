# PHASE 9A.2A: FINAL RECOMMENDATION & EXPERIMENT VERDICT

**Audit Date**: 2026-11-25  
**Experiment**: Phase 9A.2A — Adaptive Motion Gate Experiment  
**Auditor**: Antigravity Autonomous Scientific Verification Engine  
**Final Verdict**: **CONDITIONAL PASS — EXPERIMENT SUCCESSFUL, ADVANCE TO PHASE 9A.2B HUMAN VALIDATION**

---

## 1. Executive Summary & Verdict

Phase 9A.2A investigated whether static ISL signs (`21.good`, `34.stop`, `45.hand`) can be recognized without requiring continuous movement, while preventing idle hands and unintended gestures from producing false translations.

An isolated, dual-path Adaptive Motion Gate was designed and implemented in development-only modules ([ai-training/src/adaptive_motion_gate.py](file:///d:/SignBridgeAI/ai-training/src/adaptive_motion_gate.py) and [frontend/src/utils/adaptiveMotionGate.ts](file:///d:/SignBridgeAI/frontend/src/utils/adaptiveMotionGate.ts)).

### The Decision: **CONDITIONAL PASS**
- **Offline & Simulation Criteria**: **ALL 6 CRITERIA MET**.
- **Static Sign Recovery**: Successfully resolves the exact failure mode observed in TR-040 (`21.good`), TR-084 (`34.stop`), and TR-120 (`45.hand`), where correct classifications were discarded by Gate 2 as `motion_stationary`.
- **Idle Hand Rejection**: Achieved **100.00% rejection (120 / 120 windows)** across resting lap/desk postures, relaxed idle postures, and non-sign fidgeting, far exceeding the 95% requirement.
- **Zero Dynamic Regression**: Verified across all 49 classes on the 142-sample validation split (**0 / 49 classes regressed**).
- **Latency Overhead**: Negligible $+0.0003$ ms median increase (total execution $0.044$ ms, consuming $< 0.15\%$ of frame budget).
- **Condition for Production Merge**: Under strict preservation requirements, experimental code **MUST NOT be merged into production** until confirmed by live human webcam trials in Phase 9A.2B.

---

## 2. Evaluation Scorecard vs Success Criteria

| Evaluation Criterion | Requirement | Observed Result | Status |
| :--- | :--- | :--- | :--- |
| **1. Static Sign Recovery** | Reduced false rejection of intentional static signs | Resolves TR-040 (`21.good`), TR-084 (`34.stop`), TR-120 (`45.hand`) from `motion_stationary` rejection | **PASS** |
| **2. Correct `21.good` Recognition** | Recognized without introducing false `17.friend` predictions | Static hold accepted directly (TR-040); removes compensatory hand flick that induced `17.friend` confusion | **PASS** |
| **3. Handling of `34.stop` and `45.hand`** | Correct recognition without motion rejection | Sustained holds accepted under Path B; dynamic executions pass on Path A | **PASS** |
| **4. Idle Negative Rejection** | $\ge 95\%$ rejection of idle negative windows ($N \ge 100$) | **100.00% Rejection (120 / 120 windows rejected)** | **PASS** (Exceeds Goal) |
| **5. Non-Degradation of Dynamic Signs** | Zero material degradation of dynamic vocabulary | **0 / 49 classes regressed** on validation set; 100% parity across dynamic signs | **PASS** |
| **6. Preservation Compliance** | No changes to production M1 or 758-D schema | `sign_classifier.json` hash unchanged; 758-D schema untouched; isolated code | **PASS** |

---

## 3. Detailed Results Across Test Dimensions

### 3.1 Idle Negative Evaluation Summary ($N = 120$)

From [phase9a2a_idle_negative_evaluation.csv](file:///d:/SignBridgeAI/ai-training/reports/phase9a2a/phase9a2a_idle_negative_evaluation.csv):

| Negative Category | Sample Count | Primary Rejection Filter | Correctly Rejected | False Translations |
| :--- | :--- | :--- | :--- | :--- |
| **Resting Lap / Desk Hand** ($y > 0.75$) | 50 windows | `idle_hand_resting` (Y-centroid filter) | **50 / 50 (100.0%)** | 0 |
| **Idle Relaxed Hand in Chest Space** | 40 windows | `motion_stationary` (Non-static vocab) | **40 / 40 (100.0%)** | 0 |
| **Fidgeting / Jittering Hand Adjustments** | 30 windows | `static_pose_jitter` ($\sigma > 0.055$) | **30 / 30 (100.0%)** | 0 |
| **Total Idle Negative Suite** | **120 windows** | — | **120 / 120 (100.00%)** | **0** |

### 3.2 Validation Split Paired Gating Results ($N = 142$)

From [phase9a2a_gate_comparison.csv](file:///d:/SignBridgeAI/ai-training/reports/phase9a2a/phase9a2a_gate_comparison.csv) and [phase9a2a_per_class_regression.csv](file:///d:/SignBridgeAI/ai-training/reports/phase9a2a/phase9a2a_per_class_regression.csv):
- **Total Samples Evaluated**: 142
- **Original M1 Gate Accepted**: 95 / 142 (66.9%)
- **Adaptive Gate Accepted**: 95 / 142 (66.9%)
- **Dynamic Signs Retaining Path A**: 95 / 95 (100.0% parity)
- **Regressed Classes**: **0 of 49 classes**
- **Explanation**: In offline video clips, entry/exit motion exceeds $0.05$, so all valid dynamic gestures follow Path A without interference. Path B selectively activates when motion falls below $0.05$, providing the necessary safety net for live webcam holds.

### 3.3 Latency Overhead Summary (1,000 Iterations)

From [phase9a2a_latency_benchmark.json](file:///d:/SignBridgeAI/ai-training/reports/phase9a2a/phase9a2a_latency_benchmark.json):
- **M1 Original Gate Median**: 0.0437 ms
- **M1 Adaptive Gate Median**: 0.0440 ms
- **Delta**: $+0.0003$ ms (+0.69%)
- **Budget Compliance**: Consumes $< 0.15\%$ of the 33.3 ms (30 FPS) budget. Latency impact is completely negligible.

---

## 4. Why the Adaptive Motion Gate Succeeds

The Adaptive Motion Gate succeeds because it replaces an oversimplified assumption ("all signs must move") with a physiologically grounded state machine:

1. **Velocity Alone Does Not Identify Signs**: Moving hands can be idle fidgets, and motionless hands can be intentional signs.
2. **Multi-Constraint Gating**: Path B requires unanimous agreement across five independent physical metrics:
   - High active presence ($\ge 0.50$)
   - Rigid pose stability ($\sigma \le 0.055$)
   - Elevated signing space location ($y \le 0.75$)
   - Exact static vocabulary membership (`21.good`, `34.stop`, `45.hand`)
   - High classification confidence ($\ge 0.50$) with decisive margin ($\ge 0.10$)
3. **No Dynamic Class Leakage**: A motionless hand that predicts `9.hello` or `17.friend` is rejected as `motion_stationary`, ensuring dynamic signs cannot be falsely triggered by frozen postures.

---

## 5. Phase 9A.2B Roadmap & Prerequisites

To proceed to **Phase 9A.2B (Controlled Human Validation)**:

1. **Keep Production Frozen**:
   - `frontend/public/models/sign_classifier.json` remains untouched.
   - Production routes in `SignToTextPage.tsx` continue using standard production inference.
2. **Prepare Evaluation Studio for Phase 9A.2B**:
   - Integrate `adaptiveMotionGate.ts` into `EvaluationStudioPage.tsx` under a toggleable experimental test mode.
   - Record paired live webcam trials across participants for `21.good`, `34.stop`, `45.hand`, and negative idle postures.
3. **Formal Sign-off Required**:
   - Await user review and confirmation before beginning Phase 9A.2B.
