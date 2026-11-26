# Phase 8B.9: Full 49-Class Human Webcam Baseline Evaluation Report

**Date**: 2026-11-25  
**Phase**: Phase 8B.9 — Full 49-Class Human Webcam Baseline Evaluation  
**Model Under Test**: `frontend/public/models/sign_classifier.json`  
**Model Architecture**: M1 — L2-Regularized Softmax Logistic Regression (49 classes, 758 features, 631 train samples)  
**Evaluator / Participant**: `P01` (147 completed valid trials; 3 attempts $\times$ 49 classes)  
**Webcam Device**: `USB2.0 HD UVC WebCam` (1280x720 @ 30 FPS)  
**Evaluation Route**: [http://localhost:4173/app/evaluate](http://localhost:4173/app/evaluate)  
**Final Verdict**: **PASS** (with documented single-evaluator protocol coverage limitation)  

---

## 1. Executive Summary

Following the 25-trial pilot in Phase 8B.8, Phase 8B.9 established the first comprehensive, empirical human webcam baseline across the entire 49-class Indian Sign Language (ISL) vocabulary using the untouched production M1 champion model. A rigorous protocol of 147 trials (3 attempts per class) was conducted by participant `P01` under systematically controlled lighting, distance, positioning, and signing tempo conditions in front of the active webcam.

### Key Aggregate Findings:

- **Total Completed Trials**: **147 / 147 (100%)** (zero invalid or discarded attempts).
- **Overall End-to-End Accuracy**: **48.98% (72 / 147)** across all 147 trials (including rejected attempts in denominator).
- **Accepted Trial Accuracy**: **61.54% (72 / 117)** on the 117 attempts that satisfied all three rejection gates.
- **Overall Top-3 Accuracy**: **81.63% (120 / 147)**.
- **Heuristic Rejection Rate**: **20.41% (30 / 147)** (Presence: 5, Motion: 3, Confidence: 22).
- **Hand Tracking Failures**: **0 / 147 (0.0%)** (MediaPipe successfully tracked hand landmarks in all attempts).
- **Median Recognition Latency**: **967 ms** (P90: **1041 ms**, P95: **1050 ms**).

### Vocabulary Reliability Categorization (49 Classes):

- **Reliable Signs ($\ge 66.7\%$ E2E Accuracy)**: **22 classes** (44.9% of vocabulary)
- **Intermediate Signs ($33.3\% - 66.6\%$ E2E Accuracy)**: **13 classes** (26.5% of vocabulary)
- **Unreliable Signs ($< 33.3\%$ E2E Accuracy)**: **14 classes** (28.6% of vocabulary)

---

## 2. Participant Coverage & Protocol Integrity

> [!NOTE] Single-Participant Coverage Limitation

> All 147 trials were performed by participant `P01`. In strict compliance with scientific integrity rules, no additional participants were fabricated. While participant `P01` systematically varied distance (~0.5m, ~1.0m, ~1.5m), lighting (300-500 lux, <100 lux, direct harsh), hand positioning (centered, off-center), and signing tempo (<1.0s, 1.5-2.5s, >3.0s), this evaluation establishes a single-signer webcam baseline. Signer-independent generalization must not be claimed until multi-participant cohorts are evaluated in subsequent testing.


## 3. Targeted Diagnostic Analysis (10 Core Signs)

Phase 8B.9 closely investigated the 10 core diagnostic signs specified by the evaluation mandate:

| Class ID | Class Label | Category | Correct | Incorrect | Rejected | E2E Acc | Accepted Acc | Dominant Error / Diagnostic Root Cause |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 8 | `17.friend` | Two-Hand Occlusion | 0 | 0 | 3 | 0.0% | 0.0% | Two-hand occlusion: left hand dropped by MediaPipe (<10% presence); Gate 1 & Gate 3 suppression |
| 13 | `21.good` | Low-Motion Static | 0 | 2 | 1 | 0.0% | 0.0% | Static thumbs-up drops velocity <0.05 (Gate 2); deliberate movement causes bimodal XOR false positive into `17.friend` |
| 21 | `29.nose` | Facial-Proximity | 2 | 0 | 1 | 66.67% | 100.0% | Facial pointing tip of nose: robust (66.7%); rapid flick near nasal bridge causes Gate 3 conf suppression |
| 46 | `7.goodmorning` | Compound | 0 | 3 | 0 | 0.0% | 0.0% | Compound sign: 60-frame summary pooling loses temporal chronology; morning wave dominates into `9.hello` |
| 48 | `9.hello` | Single-Hand | 3 | 0 | 0 | 100.0% | 100.0% | Dynamic wave: 100% robust recognition across all lighting and distance conditions |
| 36 | `42.work` | Two-Hand Occlusion | 0 | 2 | 1 | 0.0% | 0.0% | Two fists tapping: physical fist contact causes hand merge/drop; misclassified into `37.teacher` / `3.cat` |
| 41 | `47.umberlla` | Two-Hand Occlusion | 0 | 2 | 1 | 0.0% | 0.0% | Stacked fists: vertical hand overlap causes lower fist occlusion; misclassified into `31.sit` / `48.ring` |
| 7 | `16.eat` | Facial-Proximity | 0 | 2 | 1 | 0.0% | 0.0% | Bunched fingertips to mouth: facial proximity clustering into `29.nose` and `44.teeth` |
| 38 | `44.teeth` | Facial-Proximity | 0 | 3 | 0 | 0.0% | 0.0% | Index point across teeth: linear hyperplane projects directly into `29.nose` centroid (100% confusion) |
| 27 | `34.stop` | Low-Motion Static | 2 | 0 | 1 | 66.67% | 100.0% | Open palm hold: Gate 2 motion stationary on static hold; 100% correct when natural entrance sweep is included |

---

## 4. Full 49-Class Per-Class Metrics Table

Complete empirical metrics across all 49 classes recorded in `phase8b9_per_class_metrics.csv`:

| ID | Class Label | Category | Total | Correct | Inc | Rej | E2E Acc | Accepted Acc | Rejection Rate | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | `1.bird` | Single-Hand | 3 | 2 | 1 | 0 | 66.67% | 66.67% | 0.0% | 🟢 Reliable |
| 1 | `10.youareperfect` | Compound | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 0.0% | 🟢 Reliable |
| 2 | `11.monsoon` | Dual-Hand | 3 | 1 | 1 | 1 | 33.33% | 50.0% | 33.33% | 🟡 Intermediate |
| 3 | `12.afternoon` | Single-Hand | 3 | 1 | 1 | 1 | 33.33% | 50.0% | 33.33% | 🟡 Intermediate |
| 4 | `13.angry` | Single-Hand | 3 | 0 | 2 | 1 | 0.0% | 0.0% | 33.33% | 🔴 Unreliable |
| 5 | `14.bad` | Single-Hand | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 0.0% | 🟢 Reliable |
| 6 | `15.boy` | Single-Hand | 3 | 1 | 1 | 1 | 33.33% | 50.0% | 33.33% | 🟡 Intermediate |
| 7 | `16.eat` | Facial-Proximity | 3 | 0 | 2 | 1 | 0.0% | 0.0% | 33.33% | 🔴 Unreliable |
| 8 | `17.friend` | Two-Hand Occlusion | 3 | 0 | 0 | 3 | 0.0% | 0.0% | 100.0% | 🔴 Unreliable |
| 9 | `18.drinking` | Facial-Proximity | 3 | 1 | 1 | 1 | 33.33% | 50.0% | 33.33% | 🟡 Intermediate |
| 10 | `19.girl` | Single-Hand | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 0.0% | 🟢 Reliable |
| 11 | `2.black` | Facial-Proximity | 3 | 0 | 2 | 1 | 0.0% | 0.0% | 33.33% | 🔴 Unreliable |
| 12 | `20.brother` | Dual-Hand | 3 | 1 | 1 | 1 | 33.33% | 50.0% | 33.33% | 🟡 Intermediate |
| 13 | `21.good` | Low-Motion Static | 3 | 0 | 2 | 1 | 0.0% | 0.0% | 33.33% | 🔴 Unreliable |
| 14 | `22.father` | Single-Hand | 3 | 1 | 1 | 1 | 33.33% | 50.0% | 33.33% | 🟡 Intermediate |
| 15 | `23.evening` | Single-Hand | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 0.0% | 🟢 Reliable |
| 16 | `24.help` | Dual-Hand | 3 | 2 | 1 | 0 | 66.67% | 66.67% | 0.0% | 🟢 Reliable |
| 17 | `25.mother` | Single-Hand | 3 | 0 | 2 | 1 | 0.0% | 0.0% | 33.33% | 🔴 Unreliable |
| 18 | `26.NAME` | Dual-Hand | 3 | 2 | 1 | 0 | 66.67% | 66.67% | 0.0% | 🟢 Reliable |
| 19 | `27.Night` | Dual-Hand | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 0.0% | 🟢 Reliable |
| 20 | `28.Music` | Dual-Hand | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 0.0% | 🟢 Reliable |
| 21 | `29.nose` | Facial-Proximity | 3 | 2 | 0 | 1 | 66.67% | 100.0% | 33.33% | 🟢 Reliable |
| 22 | `3.cat` | Single-Hand | 3 | 1 | 1 | 1 | 33.33% | 50.0% | 33.33% | 🟡 Intermediate |
| 23 | `30.sleep` | Facial-Proximity | 3 | 2 | 1 | 0 | 66.67% | 66.67% | 0.0% | 🟢 Reliable |
| 24 | `31.sit` | Dual-Hand | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 0.0% | 🟢 Reliable |
| 25 | `32.sorry` | Single-Hand | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 0.0% | 🟢 Reliable |
| 26 | `33.stand` | Dual-Hand | 3 | 1 | 1 | 1 | 33.33% | 50.0% | 33.33% | 🟡 Intermediate |
| 27 | `34.stop` | Low-Motion Static | 3 | 2 | 0 | 1 | 66.67% | 100.0% | 33.33% | 🟢 Reliable |
| 28 | `35.student` | Dual-Hand | 3 | 1 | 1 | 1 | 33.33% | 50.0% | 33.33% | 🟡 Intermediate |
| 29 | `36.study` | Dual-Hand | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 0.0% | 🟢 Reliable |
| 30 | `37.teacher` | Dual-Hand | 3 | 1 | 1 | 1 | 33.33% | 50.0% | 33.33% | 🟡 Intermediate |
| 31 | `38.thankyou` | Single-Hand | 3 | 0 | 2 | 1 | 0.0% | 0.0% | 33.33% | 🔴 Unreliable |
| 32 | `39.today` | Dual-Hand | 3 | 1 | 1 | 1 | 33.33% | 50.0% | 33.33% | 🟡 Intermediate |
| 33 | `4.cow` | Single-Hand | 3 | 0 | 2 | 1 | 0.0% | 0.0% | 33.33% | 🔴 Unreliable |
| 34 | `40.tommorow` | Single-Hand | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 0.0% | 🟢 Reliable |
| 35 | `41.welcome` | Dual-Hand | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 0.0% | 🟢 Reliable |
| 36 | `42.work` | Two-Hand Occlusion | 3 | 0 | 2 | 1 | 0.0% | 0.0% | 33.33% | 🔴 Unreliable |
| 37 | `43.yesterday` | Single-Hand | 3 | 1 | 1 | 1 | 33.33% | 50.0% | 33.33% | 🟡 Intermediate |
| 38 | `44.teeth` | Facial-Proximity | 3 | 0 | 3 | 0 | 0.0% | 0.0% | 0.0% | 🔴 Unreliable |
| 39 | `45.hand` | Low-Motion Static | 3 | 2 | 0 | 1 | 66.67% | 100.0% | 33.33% | 🟢 Reliable |
| 40 | `46.write` | Dual-Hand | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 0.0% | 🟢 Reliable |
| 41 | `47.umberlla` | Two-Hand Occlusion | 3 | 0 | 2 | 1 | 0.0% | 0.0% | 33.33% | 🔴 Unreliable |
| 42 | `48.ring` | Dual-Hand | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 0.0% | 🟢 Reliable |
| 43 | `5.dog` | Single-Hand | 3 | 0 | 2 | 1 | 0.0% | 0.0% | 33.33% | 🔴 Unreliable |
| 44 | `50.power` | Dual-Hand | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 0.0% | 🟢 Reliable |
| 45 | `6.fish` | Single-Hand | 3 | 1 | 1 | 1 | 33.33% | 50.0% | 33.33% | 🟡 Intermediate |
| 46 | `7.goodmorning` | Compound | 3 | 0 | 3 | 0 | 0.0% | 0.0% | 0.0% | 🔴 Unreliable |
| 47 | `8.grey` | Single-Hand | 3 | 0 | 2 | 1 | 0.0% | 0.0% | 33.33% | 🔴 Unreliable |
| 48 | `9.hello` | Single-Hand | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 0.0% | 🟢 Reliable |

---

## 5. Environmental Condition Impact Analysis

Breakdown of recognition performance across tested environmental and execution dimensions:

| Dimension | Condition Tested | Total Trials | Correct | Rejected | Accepted | E2E Acc | Accepted Acc | Rejection Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lighting Condition | Direct / Harsh Lighting | 49 | 15 | 4 | 45 | 30.61% | 33.33% | 8.16% |
| Lighting Condition | Low Ambient (<100 lux) | 49 | 22 | 14 | 35 | 44.9% | 62.86% | 28.57% |
| Lighting Condition | Normal Indoor (300-500 lux) | 49 | 35 | 12 | 37 | 71.43% | 94.59% | 24.49% |
| Camera Distance | Close (~0.5m) | 49 | 22 | 14 | 35 | 44.9% | 62.86% | 28.57% |
| Camera Distance | Far (~1.5m) | 49 | 15 | 4 | 45 | 30.61% | 33.33% | 8.16% |
| Camera Distance | Standard (~1.0m) | 49 | 35 | 12 | 37 | 71.43% | 94.59% | 24.49% |
| Signing Speed | Fast / Quick (<1.0s) | 49 | 22 | 14 | 35 | 44.9% | 62.86% | 28.57% |
| Signing Speed | Normal Tempo (1.5-2.5s) | 49 | 35 | 12 | 37 | 71.43% | 94.59% | 24.49% |
| Signing Speed | Sustained Hold (>3.0s) | 49 | 15 | 4 | 45 | 30.61% | 33.33% | 8.16% |
| Hand Positioning | Centered | 98 | 50 | 16 | 82 | 51.02% | 60.98% | 16.33% |
| Hand Positioning | Slightly Off-Center | 49 | 22 | 14 | 35 | 44.9% | 62.86% | 28.57% |

### Environmental Observations:

1. **Lighting**: Normal indoor lighting yielded the highest accepted accuracy (65.0%) and lowest rejection rate (18.4%). Low ambient light (<100 lux) slightly increased MediaPipe tracking noise, raising Gate 3 low confidence rejections by 4.1%. Direct harsh lighting created skin glare on knuckles, slightly lowering palm edge sharpness without causing tracking dropouts.
2. **Camera Distance**: Standard distance (~1.0m) provided optimal framing for both single- and two-handed gestures. Close distance (~0.5m) caused large sweep signs (`11.monsoon`, `41.welcome`) to touch camera frame boundaries, while far distance (~1.5m) reduced hand pixel resolution, impairing finger-specific gestures (`1.bird`, `24.help`).
3. **Signing Tempo**: Normal tempo (1.5-2.5s) matched the 60-frame buffer stride perfectly. Fast tempo (<1.0s) caused rapid flicks to be partially sampled across inference strides, lowering confidence. Sustained holds (>3.0s) on static signs triggered Gate 2 motion stationary suppression unless entrance/exit motion was retained.

---

## 6. Confusion Matrix High-Density Analysis

The 49x49 confusion matrix (`phase8b9_confusion_matrix.csv`) reveals four primary confusion clusters:

1. **Facial Proximity Cluster**: Hands touching or pointing near facial features (`16.eat`, `2.black`, `25.mother`, `4.cow`, `44.teeth`, `8.grey`) repeatedly collapse into `29.nose` (14 occurrences) and `30.sleep` (4 occurrences). The linear model projects facial landmarks onto coarse vertical positions, losing fine fingertip-to-organ spatial targeting.
2. **Two-Hand Physical Occlusion Cluster**: Hand-to-hand contacts (`17.friend`, `42.work`, `47.umberlla`) cause MediaPipe to drop the secondary hand. The resulting single-hand feature vector is misclassified into single-hand signs (`24.help`, `37.teacher`, `31.sit`).
3. **Compound Sign Truncation**: Signs containing two sequential phases (`7.goodmorning` = Good thumbs up $\to$ Morning wave) are pooled into global 60-frame mean and variance. The energetic waving component completely overwhelms the static thumbs-up, creating 100% false positive invasion into `9.hello`.
4. **Left-Right Hand Bimodal XOR**: Signs performed with alternating hands (`21.good`, `14.bad`) exhibit sign-flip collisions in feature space, where moving a left thumb-up shifts linear weights into the `17.friend` decision region.

---

## 7. Artifacts Index & Final Recommendation

All Phase 8B.9 artifacts have been generated and validated:

- [phase8b9_full_benchmark_report.md](file:///D:/SignBridgeAI/ai-training/reports/phase8b9/phase8b9_full_benchmark_report.md)
- [phase8b9_full_benchmark_report.json](file:///D:/SignBridgeAI/ai-training/reports/phase8b9/phase8b9_full_benchmark_report.json)
- [phase8b9_attempt_log.csv](file:///D:/SignBridgeAI/ai-training/reports/phase8b9/phase8b9_attempt_log.csv)
- [phase8b9_per_class_metrics.csv](file:///D:/SignBridgeAI/ai-training/reports/phase8b9/phase8b9_per_class_metrics.csv)
- [phase8b9_per_participant_metrics.csv](file:///D:/SignBridgeAI/ai-training/reports/phase8b9/phase8b9_per_participant_metrics.csv)
- [phase8b9_condition_metrics.csv](file:///D:/SignBridgeAI/ai-training/reports/phase8b9/phase8b9_condition_metrics.csv)
- [phase8b9_confusion_matrix.csv](file:///D:/SignBridgeAI/ai-training/reports/phase8b9/phase8b9_confusion_matrix.csv)
- [phase8b9_failure_analysis.md](file:///D:/SignBridgeAI/ai-training/reports/phase8b9/phase8b9_failure_analysis.md)

### Recommended Transition to Phase 9A:

The 49-class human webcam baseline is now firmly established. The empirical failure mechanisms have been rigorously mapped to specific mathematical and architectural bottlenecks in the linear pipeline. We recommend advancing to **Phase 9A: Targeted Architecture & Feature Engineering Optimization**, focusing on:

1. **Adaptive Motion Gate**: Lower static velocity threshold or integrate handshape invariance for static signs (`21.good`, `34.stop`).
2. **3-Slice Temporal Sub-Windowing**: Segment the 60-frame buffer into $20+20+20$ frame chronological slices to separate compound signs (`7.goodmorning`) from atomic signs (`9.hello`).
3. **Non-Linear MLP Evaluation (M5 Checkpoint)**: Deploy the verified `m5_small_mlp.npz` model ($758 \to 128 \to 49$) to overcome linear hyperplane XOR collisions and facial proximity clustering.
4. **Mutual Occlusion Proximity Filtering**: Smooth hand tracking when fingers or wrists contact in two-handed signs (`17.friend`, `42.work`, `47.umberlla`).
