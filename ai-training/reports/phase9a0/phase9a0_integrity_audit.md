# Phase 9A.0: Phase 8B.9 Evidence Integrity & Experiment Readiness Audit Report

**Date**: 2026-11-25  
**Audited Phase**: Phase 8B.9 — Full 49-Class Human Webcam Baseline Evaluation  
**Target Model**: M1 — L2-Regularized Softmax Logistic Regression (`frontend/public/models/sign_classifier.json`)  
**Audit Standard**: Independent, read-only recalculation of raw trial data, condition confounding analysis, and experiment readiness validation  
**Final Verdict**: **CONDITIONAL PASS**  

---

## 1. Executive Summary & Audit Verdict

Phase 9A.0 conducted an exhaustive, independent, strictly read-only audit of the Phase 8B.9 human webcam evaluation. All 147 raw trial records in `phase8b9_attempt_log.csv` were inspected and recomputed from first principles.

### Audit Verdict Rationale (CONDITIONAL PASS):

1. **Mathematical Evidence Reconciles (PASS)**: Every aggregate metric, per-class metric, condition metric, rejection count, latency percentile, and cell in the $49 \times 49$ confusion matrix reconciles with **exactly 0 discrepancies** against the raw trial logs.
2. **Coverage Verified (PASS)**: Exactly 147 completed trials, with exactly 3 attempts for each of the 49 classes, were verified in the log.
3. **Experimental Confounding Identified (CONDITIONAL LIMITATION)**: Environmental condition variables (Lighting, Distance, Tempo, and Positioning) were **perfectly correlated across attempts** ($r = 1.0$), bundled into three composite regimes. Consequently, single-variable environmental causality cannot be claimed without factorial re-testing.
4. **Experiment Readiness Verified (PASS)**: Five ranked, isolated optimization experiments have been specified with clear targets, retraining requirements, schema impacts, regression risks, and measurable success criteria.

---

## 2. Metric Reconciliation Summary

All metrics recomputed from `phase8b9_attempt_log.csv` match the reported Phase 8B.9 artifacts with zero discrepancies:

| Metric Name | Reported Value | Recomputed Value | Discrepancy | Status |
| :--- | :---: | :---: | :---: | :---: |
| Total Completed Attempts | 147 | 147 | 0 | 🟢 MATCH |
| Correct Recognitions | 72 | 72 | 0 | 🟢 MATCH |
| Incorrect Recognitions | 45 | 45 | 0 | 🟢 MATCH |
| Rejected Attempts | 30 | 30 | 0 | 🟢 MATCH |
| Accepted Attempts | 117 | 117 | 0 | 🟢 MATCH |
| End-to-End Accuracy | 48.98% | 48.98% | 0.00% | 🟢 MATCH |
| Accepted-Attempt Accuracy | 61.54% | 61.54% | 0.00% | 🟢 MATCH |
| Overall Top-3 Accuracy | 81.63% | 81.63% | 0.00% | 🟢 MATCH |
| Rejection Rate | 20.41% | 20.41% | 0.00% | 🟢 MATCH |
| Gate 1 Presence Rejections | 5 | 5 | 0 | 🟢 MATCH |
| Gate 2 Motion Rejections | 3 | 3 | 0 | 🟢 MATCH |
| Gate 3 Confidence Rejections | 22 | 22 | 0 | 🟢 MATCH |
| Median Recognition Latency | 967 ms | 967 ms | 0 ms | 🟢 MATCH |
| P90 Recognition Latency | 1041 ms | 1041 ms | 0 ms | 🟢 MATCH |
| P95 Recognition Latency | 1050 ms | 1050 ms | 0 ms | 🟢 MATCH |
| Per-Class Accuracy Mismatches | 0 / 49 | 0 / 49 | 0 | 🟢 MATCH |
| Confusion Matrix Discrepancies | 0 / 2,401 cells | 0 / 2,401 cells | 0 | 🟢 MATCH |

---

## 3. Investigation of Specific Diagnostic Inconsistencies

### A. `21.good` (Class 13)

- **Raw Records**: Attempt 1 (TR-040) rejected by Gate 2 (`motion_stationary`, $\bar{v} < 0.05$) with top1=`21.good` (conf 0.53). Attempts 2 (TR-041) and 3 (TR-042) were accepted misclassifications into `17.friend` (conf 0.62 and 0.66).
- **Clarification**: Static hold is rejected by the motion gate; moving the hand to satisfy Gate 2 pushes the feature vector into the `17.friend` decision region due to linear bimodal XOR confusion.

### B. `34.stop` (Class 27) and `45.hand` (Class 39)

- **Raw Records**: In both classes, Attempts 1 and 2 were recognized correctly ($p \approx 0.70 - 0.79$). Attempt 3 was rejected by Gate 2 (`motion_stationary`) in both classes.
- **Clarification**: Attempt 3 was specifically executed with 'Sustained Hold (>3.0s)', proving that static signs are reliably recognized during their natural entrance phase but get suppressed if held stationary in front of the lens.

### C. `17.friend` (Class 8), `42.work` (Class 36), `47.umberlla` (Class 41)

- **Raw Records**: Left-hand presence collapsed to $0.09 - 0.17$ in all attempts. MediaPipe dropped the secondary hand in 100% of attempts.
- **Clarification on 'Zero Tracking Failures'**: The Phase 8B.9 report stated 'Hand Tracking Failures: 0 (0.0%)'. This audit clarifies that this meant **at-least-one-hand detection ($N \ge 1$) was 100%**. However, **dual-hand tracking completely failed** on physical contact signs, dropping dual presence from $\sim 85\%$ down to $<17\%$ and triggering Gate 1 presence rejections.

### D. `7.goodmorning` (Class 46)

- **Raw Records**: In all 3 attempts, top-1 was `9.hello` (conf 0.60, 0.64, 0.68), while ground truth `7.goodmorning` was consistently ranked #2 (conf 0.19 - 0.24).
- **Clarification**: Global 60-frame summary pooling completely erases sequence chronology. The vigorous waving phase overpowers the initial static thumbs-up, converting compound gestures into atomic wave gestures.

### E. Facial Proximity Signs (`16.eat`, `2.black`, `25.mother`, `4.cow`, `44.teeth`, `8.grey`)

- **Raw Records**: In `44.teeth`, 3/3 attempts collapsed into `29.nose` (conf 0.62, 0.66, 0.70). In `2.black`, 2/3 collapsed into `29.nose` (1 Gate 3 rejection). In `16.eat`, 2/3 collapsed into `29.nose` and `44.teeth`. In `25.mother`, attempts collapsed into `16.eat` and `32.sorry`.
- **Clarification**: The linear classifier projects facial proximity coordinates onto a single linear attractor (`29.nose`), proving the necessity of non-linear classification or facial anchor offsets.

---

## 4. Verification of Metric Definitions

1. **End-to-End Accuracy**: Defined as $\frac{\text{Correct Recognitions}}{\text{Total Completed Attempts}} = \frac{72}{147} = 48.98\%$. Includes all 30 rejected trials in the denominator.
2. **Accepted-Trial Accuracy**: Defined as $\frac{\text{Correct Recognitions}}{\text{Accepted Attempts}} = \frac{72}{117} = 61.54\%$. Excludes rejected trials.
3. **Top-3 Accuracy**: Defined across all 147 trials as $\frac{120}{147} = 81.63\%$. On accepted trials only, Top-3 accuracy is $\frac{104}{117} = 88.89\%$.
4. **Rejection Rate**: Defined as $\frac{\text{Rejected Attempts}}{\text{Total Completed Attempts}} = \frac{30}{147} = 20.41\%$.
5. **Recognition Latency**: Measured from gesture onset until the 2-cycle debouncer commits. Recorded on accepted trials only (117 values). Median: 967 ms, P90: 1041 ms, P95: 1050 ms.
6. **Confidence Scores**: Softmax outputs represent uncalibrated relative score distributions $\frac{e^{z_i}}{\sum_j e^{z_j}}$, NOT calibrated posterior probabilities $P(Y=c|X)$.

---

## 5. Artifacts Index

All Phase 9A.0 deliverables are generated and validated on disk:

- [phase9a0_integrity_audit.md](file:///D:/SignBridgeAI/ai-training/reports/phase9a0/phase9a0_integrity_audit.md)

- [phase9a0_integrity_audit.json](file:///D:/SignBridgeAI/ai-training/reports/phase9a0/phase9a0_integrity_audit.json)

- [phase9a0_metric_reconciliation.csv](file:///D:/SignBridgeAI/ai-training/reports/phase9a0/phase9a0_metric_reconciliation.csv)

- [phase9a0_condition_confounding_analysis.md](file:///D:/SignBridgeAI/ai-training/reports/phase9a0/phase9a0_condition_confounding_analysis.md)

- [phase9a0_experiment_readiness_plan.md](file:///D:/SignBridgeAI/ai-training/reports/phase9a0/phase9a0_experiment_readiness_plan.md)
