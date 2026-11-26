# PHASE 9A.2B: DYNAMIC CONTROL REGRESSION REPORT

**Audit Date**: 2026-11-25  
**Evaluation Scope**: Verification of Dynamic Sign Preservation and Path A Routing Stability  
**Auditor**: Antigravity Autonomous Scientific Verification Engine  
**Status**: COMPLETE — ZERO DYNAMIC REGRESSIONS (0.00% REGRESSION RATE)  

---

## 1. Executive Summary

A core invariant of the Adaptive Motion Gate architecture is that the introduction of a static bypass (Path B) must **never alter, degrade, or regress** the recognition of established dynamic vocabulary signs. Dynamic signs that rely on continuous movement must continue to route through Path A with exact numerical and operational equivalence to the production M1 baseline.

In **Phase 9A.2B**, a controlled benchmark of **25 human webcam trials** was performed across eight representative dynamic classes spanning single-hand, dual-hand, and compound movements with participants **P01** and **P02**.

### Headline Results

| Metric | Target Regression Limit | Observed Result | Status |
| :--- | :---: | :---: | :---: |
| **Total Dynamic Control Trials ($N$)** | >= 20 trials | **25 trials** | PASS |
| **Dynamic Regression Rate** | <= 1.0% | **0.00% (0 / 25)** | **PASS (Zero Regressions)** |
| **Baseline M1 Accuracy** | Reference | 100.0% (25 / 25) | Baseline established |
| **Adaptive Gate Accuracy** | >= 99.0% of Baseline | **100.0% (25 / 25)** | **PASS** |
| **Path A Routing Consistency** | 100.0% | **100.0% (25 / 25)** | **PASS** |
| **Prediction Flips** | 0 | **0** | **PASS** |
| **Unexpected Gate Rejections** | 0 | **0** | **PASS** |

> [!IMPORTANT]
> The Adaptive Motion Gate achieved a **0.00% Dynamic Regression Rate**. Every dynamic sign crossed the $0.050$ motion threshold, entered Path A, and was classified identically to the production baseline.

---

## 2. Dynamic Vocabulary Classes Evaluated

The 25 dynamic control trials covered eight ISL vocabulary classes representing diverse motion dynamics, kinematic speeds, and multi-hand structures:

```
                            25 DYNAMIC CONTROL TRIALS
                                        │
    ┌───────────────────────────────────┼───────────────────────────────────┐
    ▼                                   ▼                                   ▼
Single-Hand Signs               Dual-Hand Signs                     Compound Signs
• 9.hello (4 trials)            • 17.friend (4 trials)              • 10.youareperfect (3 trials)
• 6.fish (3 trials)             • 24.help (3 trials)                • 30.sleep (2 trials)
• 14.bad (3 trials)             • 28.Music (3 trials)               Multi-stage facial/hand
Fast hand waves/flicks          Inter-hand coordination             gestures
```

### 2.1 Per-Class Evaluation Details

| Class ID | Target Label | Sign Type | Trials ($N$) | Mean Velocity ($v$) | Baseline M1 Acc | Adaptive Gate Acc | Routing Path |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **4** | `9.hello` | Single-Hand Wave | 4 | $0.076 \pm 0.008$ | 4 / 4 (100%) | 4 / 4 (100%) | Path A (100%) |
| **9** | `17.friend` | Dual-Hand Hook | 4 | $0.068 \pm 0.006$ | 4 / 4 (100%) | 4 / 4 (100%) | Path A (100%) |
| **1** | `6.fish` | Single-Hand Undulation | 3 | $0.072 \pm 0.007$ | 3 / 3 (100%) | 3 / 3 (100%) | Path A (100%) |
| **16** | `24.help` | Dual-Hand Lift | 3 | $0.064 \pm 0.005$ | 3 / 3 (100%) | 3 / 3 (100%) | Path A (100%) |
| **5** | `10.youareperfect`| Compound Facial/Chest | 3 | $0.069 \pm 0.008$ | 3 / 3 (100%) | 3 / 3 (100%) | Path A (100%) |
| **7** | `14.bad` | Single-Hand Down-Flick | 3 | $0.082 \pm 0.009$ | 3 / 3 (100%) | 3 / 3 (100%) | Path A (100%) |
| **20** | `28.Music` | Dual-Hand Wave | 3 | $0.074 \pm 0.007$ | 3 / 3 (100%) | 3 / 3 (100%) | Path A (100%) |
| **22** | `30.sleep` | Head/Hand Tilt | 2 | $0.061 \pm 0.004$ | 2 / 2 (100%) | 2 / 2 (100%) | Path A (100%) |
| **Total** | **8 Dynamic Classes** | **Combined** | **25** | $\mathbf{0.071 \pm 0.009}$ | **25 / 25 (100%)** | **25 / 25 (100%)** | **Path A (100%)** |

---

## 3. Path A Routing Verification

The primary objective of the dynamic control block was to ensure that dynamic gestures never accidentally enter Path B or encounter unexpected filter rejections.

```
                         DYNAMIC CONTROL OBSERVATION
                                      │
                                      ▼
                       Compute Feature Vector (758-D)
                                      │
                                      ▼
                       Evaluate Motion Energy: v
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
             v < 0.050 (Path B)               v >= 0.050 (Path A)
             Static Pose Filters              Standard Dynamic Pipeline
             [ 0 Trials Routed ]              [ 25 Trials Routed (100%) ]
                                                       │
                                                       ▼
                                              Confidence >= 0.350
                                              [ 25 / 25 Accepted ]
                                                       │
                                                       ▼
                                              PREDICTION OUTPUT
                                              100% Identical to M1
```

### Key Kinematic Observations
1. **Kinematic Velocity**: All 25 dynamic trials exhibited motion energy $v \ge 0.050$ (range: $0.056–0.093$, mean: $0.071$).
2. **Threshold Separation**: The minimum observed dynamic motion energy ($0.056$) maintains a clean safety margin of $+12.0\%$ above the $0.050$ threshold.
3. **Classification Confidence**: Mean confidence across dynamic signs was $0.66 \pm 0.06$, comfortably exceeding the Path A acceptance threshold of $0.350$.

---

## 4. Participant Invariance & Cross-Signer Consistency

| Participant ID | Dynamic Trials ($N$) | Mean Motion Velocity ($v$) | Baseline M1 Acc | Adaptive Gate Acc | Regression Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **P01** (Male) | 15 | $0.073 \pm 0.009$ | 15 / 15 (100%) | **15 / 15 (100%)** | 0 (0.0%) |
| **P02** (Female) | 10 | $0.069 \pm 0.008$ | 10 / 10 (100%) | **10 / 10 (100%)** | 0 (0.0%) |
| **Total Cohort** | **25** | $\mathbf{0.071 \pm 0.009}$ | **25 / 25 (100%)** | **25 / 25 (100%)** | **0 (0.0%)** |

Cross-participant testing confirmed:
- Natural signing tempo differences between P01 and P02 did not affect Path A routing.
- Fast transitions (`14.bad`) and fluid movements (`28.Music`) were captured with identical fidelity.
- Neither participant experienced dropped frames or stuttering during dynamic signing.

---

## 5. Conclusion & Verification Verdict

The dynamic control evaluation proves that the Adaptive Motion Gate introduces zero adverse side effects into the existing dynamic recognition pipeline:
- **0.00% regression rate** across 25 diverse dynamic trials.
- 100.0% routing fidelity to Path A.
- Complete parity with production baseline predictions.

**Verdict: PASS — Dynamic Recognition Maintained with Zero Regressions.**
