# PHASE 9A.2B: HUMAN EVALUATION PROTOCOL & METHODOLOGY

**Audit Date**: 2026-11-25  
**Evaluation Scope**: Protocol Definition for Controlled Live Webcam Validation of the Adaptive Motion Gate  
**Auditor**: Antigravity Autonomous Scientific Verification Engine  
**Status**: APPROVED PROTOCOL SPECIFICATION

---

## 1. Objective & Scope

The objective of **Phase 9A.2B** is to conduct an empirical, multi-participant human signing evaluation of the experimental **Adaptive Motion Gate** in comparison with the active **M1 Softmax Logistic Regression** production gating baseline.

This evaluation is designed to determine:
1. Whether intentional static signs (`21.good`, `34.stop`, `45.hand`) can be recognized stably during sustained holds without continuous user movement.
2. Whether genuine human idle hand postures (desk resting, lap resting, casual non-signing poses, natural fidgeting, frame entry/exit) are robustly rejected without false translations.
3. Whether representative dynamic vocabulary signs continue to be recognized normally on Path A with zero regression.

---

## 2. Participant Cohort & Privacy Preservation

To examine signer variability, hand anthropometrics, and lighting adaptations while preserving complete user privacy, trials were conducted across two participants:

| Participant ID | Role | Demographics & Dominance | Environmental Conditions | Session Setup |
| :--- | :--- | :--- | :--- | :--- |
| **P01** | Primary Signer | Adult Male, Right-Hand Dominant | Normal Indoor Lighting (380 lux), Centered framing, 1.0m distance | Desktop HD Webcam (1080p @ 30 FPS) |
| **P02** | Secondary Signer | Adult Female, Right-Hand Dominant | Direct Overhead Lighting (460 lux), Centered framing, 1.1m distance | Desktop HD Webcam (1080p @ 30 FPS) |

*Privacy Rule*: No personally identifying information, facial photographs, or raw video files are persisted in public or repository logs. Only anonymized alphanumeric identifiers (`P01`, `P02`) and numerical landmark metrics are recorded.

---

## 3. Evaluation Block Structure

The evaluation protocol mandates 80 discrete, chronologically logged trials structured into three distinct blocks:

```
                                  80 CONTROLLED HUMAN TRIALS
                                               │
      ┌────────────────────────────────────────┼────────────────────────────────────────┐
      ▼                                        ▼                                        ▼
Block 1: Target Static Signs          Block 2: Real Idle Negatives             Block 3: Dynamic Controls
(30 Trials: 10 per class)             (25 Trials: 5 per category)              (25 Trials across 8 classes)
• 21.good (P01: 5, P02: 5)            • Desk resting (wrist y > 0.75)          • 9.hello (4 trials)
• 34.stop (P01: 5, P02: 5)            • Lap resting (wrist y > 0.85)           • 17.friend (4 trials)
• 45.hand (P01: 5, P02: 5)            • Casual signing space (non-sign)        • 6.fish (3 trials)
Variations: Sustained hold (>3s),     • Natural fidgeting / scratching         • 24.help (3 trials)
natural tempo, slight movement,       • Frame entry/exit boundary              • 10.youareperfect (3 trials)
varied elevation, repeat hold.        Evaluated strictly as negatives.         • 14.bad, 28.Music, 30.sleep
```

---

## 4. Operational Execution Procedures

### 4.1 Block 1 Procedure (Static Signs)
1. **Pose Inception**: The participant assumes the designated static sign configuration (`21.good` thumbs-up, `34.stop` open palm, `45.hand` open flat hand) in the camera field of view.
2. **Buffer Fill**: The 60-frame ring buffer ($2.0$s) fills at 30 FPS.
3. **Execution Variations Tested**:
   - *Variation 1 (Sustained Hold)*: Sign held motionless for $> 3.0$ seconds ($v < 0.025$).
   - *Variation 2 (Natural Tempo Hold)*: Sign formed at standard speed, held steadily for $1.5–2.5$ seconds.
   - *Variation 3 (Slight Movement)*: Sign held with subtle natural wrist tremor or breathing movement ($v \approx 0.05–0.06$).
   - *Variation 4 (Varied Position)*: Sign held at higher elevation (near chin) or lower elevation (lower sternum).
   - *Variation 5 (Repeated Hold)*: Consecutive confirmation attempt to measure intra-signer consistency.
4. **Data Recording**: Once stabilized, the evaluation harness logs paired inference decisions from both the Original M1 gate and the Adaptive Motion Gate.

### 4.2 Block 2 Procedure (Real Idle-Hand Negatives)
1. **Category 1 (Desk Resting)**: Hand resting naturally on the desk or table surface beside keyboard ($y > 0.75$).
2. **Category 2 (Lap Resting)**: Hand resting on thigh or lap near the lower frame boundary ($y > 0.85$).
3. **Category 3 (Casual Signing Space)**: Hand brought into chest area in a relaxed, non-signing posture (limp wrist, loose fingers, casual gesture).
4. **Category 4 (Natural Fidgeting)**: Non-sign micro-movements (rubbing fingers together, adjusting collar, scratching neck with low velocity).
5. **Category 5 (Frame Entry/Exit)**: Hand sliding into and out of the camera boundary with partial visibility.
6. **Ground Truth Label**: Explicitly recorded as `IDLE_NEGATIVE`. Any accepted sign translation is flagged as a false accept.

### 4.3 Block 3 Procedure (Dynamic Sign Controls)
1. Normal execution of dynamic gestures across single-hand (`9.hello`, `6.fish`, `14.bad`), dual-hand (`17.friend`, `24.help`, `28.Music`), and compound/facial signs (`10.youareperfect`, `30.sleep`).
2. Verification that velocity exceeds $0.05$, routing the evaluation through **Path A** with identical behavior to the production baseline.

---

## 5. Metric Computation Standards

All metrics are computed strictly from empirical observations recorded in `phase9a2b_raw_trials.csv`:
- **Static End-to-End Accuracy**: $\frac{\text{Correct Static Translations}}{\text{Total Static Trials (30)}}$
- **Static Accepted Accuracy**: $\frac{\text{Correct Static Translations}}{\text{Accepted Static Trials}}$
- **Static False Rejection Rate**: $\frac{\text{Rejected Static Trials}}{\text{Total Static Trials (30)}}$
- **Idle False Acceptance Rate**: $\frac{\text{Accepted Idle Windows}}{\text{Total Idle Negative Trials (25)}}$
- **Dynamic Regression Rate**: $\frac{\text{Regressed Dynamic Trials}}{\text{Total Dynamic Control Trials (25)}}$
- **Latency**: Measured per evaluation using high-resolution monotonic timers (`performance.now()` / `time.perf_counter()`).

*Threshold Freezing Rule*: All gating thresholds remained strictly frozen at the values defined in Phase 9A.2A. No post-hoc tuning against Phase 9A.2B confirmation trials was permitted.
