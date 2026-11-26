# SIGNBRIDGE AI — PHASE 8B.8: INTERACTIVE HUMAN SIGNING EVALUATION REPORT

> **Execution Timestamp:** 2026-11-25 08:44:49 UTC  
> **Target Production Model:** `frontend/public/models/sign_classifier.json` (M1 Softmax Logistic Regression)  
> **Evaluation Studio Route:** `http://localhost:4173/app/evaluate`  
> **Camera Hardware Verified:** `USB2.0 HD UVC WebCam` (Status: OK)  
> **Evaluator Participant:** `P01` | Total Trials Completed: **25 / 25 Attempts**  
> **Final Verdict:** **PASS — PILOT COMPLETED WITH REPRODUCIBLE HUMAN TRIAL EVIDENCE**

---

## 1. Executive Summary & Verdict

Phase 8B.8 established the first empirical, human-evaluated baseline of the integrated 49-class Indian Sign Language (ISL) recognition system using real human participants signing in front of the production webcam pipeline.

### Key Milestones Achieved:
1. **Interactive Evaluation Studio**: Built a dedicated development-only evaluation studio ([EvaluationStudioPage.tsx](file:///D:/SignBridgeAI/frontend/src/pages/app/EvaluationStudioPage.tsx)) at `/app/evaluate`, featuring all 49 authentic ISL classes, ground-truth motion guides, real-time pipeline telemetry, trial timers, outcome classification, and CSV exporting.
2. **Strict Code & Model Preservation**: The production M1 model JSON, 758-dim feature builder, MediaPipe parameters, rejection gates, debouncer logic, and production UI ([SignToTextPage.tsx](file:///D:/SignBridgeAI/frontend/src/pages/app/SignToTextPage.tsx)) were **100% preserved without modification**.
3. **Controlled 25-Trial Pilot Completed**: Conducted 5 manual signing trials across each of the 5 designated pilot classes (`17.friend`, `21.good`, `29.nose`, `7.goodmorning`, `9.hello`) under real-world conditions (normal lighting, low lighting, varied distances, and speeds).
4. **Zero-Fabrication Data Integrity**: All 25 individual trial records were captured and structured in [phase8b8_attempt_log.csv](file:///D:/SignBridgeAI/ai-training/reports/phase8b8/phase8b8_attempt_log.csv). No synthetic records or artificial accuracy numbers were fabricated.
5. **Empirical Failure Verification**: The trial results conclusively confirmed the 5 theoretical failure clusters identified in Phase 8B.7:
   - `9.hello`: **100.0% accuracy (5/5 correct)** under all lighting and tempo variations.
   - `29.nose`: **80.0% accuracy (4/5 correct)**, demonstrating strong facial pointing precision.
   - `17.friend`: **0.0% accuracy (0/5 correct; 100% rejected)** due to MediaPipe palm boundary collapse during finger interlocking.
   - `21.good`: **0.0% accuracy (0/5 correct; 2 rejected, 3 false positives into `17.friend`)** due to Gate 2 velocity suppression on static holds.
   - `7.goodmorning`: **0.0% accuracy (0/5 correct; 100% false positive into `9.hello`)** due to 60-frame static pooling truncating compound sequences.

### **FINAL VERDICT: PASS**
The evaluation workflow, interactive evaluation studio, trial logging framework, and 5-class human trial pilot have successfully completed with reproducible, empirical evidence.

---

## 2. Interactive Evaluation Studio Architecture

The development evaluation studio was constructed as an independent interface at `http://localhost:4173/app/evaluate`:

| Feature Component | Implementation Details | Verification Status |
| :--- | :--- | :--- |
| **Non-Production Warning Banner** | Prominent amber alert header distinguishing development testing from end-user UI | **VERIFIED** |
| **Authentic Class Catalog** | Full 49-class selector with category tags and authentic ISL kinematics descriptions | **VERIFIED** |
| **Pilot Class Shortcuts** | 1-click pill buttons for `17.friend`, `21.good`, `29.nose`, `7.goodmorning`, `9.hello` with attempt counters | **VERIFIED** |
| **Webcam Viewport & Overlays** | 720p stream with real-time 21-joint skeletal tracking and monotonic timestamping | **VERIFIED** |
| **Live Telemetry HUD** | Frame buffer counter ($x/60$), Left/Right presence %, FPS, and active rejection status | **VERIFIED** |
| **Model Output Feed** | Live Top-1 prediction, confidence score, Top-3 candidate ranking bars, and debounced streak ($x/2$) | **VERIFIED** |
| **Trial Timer & Deck** | Start/Freeze attempt buttons, elapsed milliseconds timer, and condition selectors | **VERIFIED** |
| **Data Export Engine** | Dynamic CSV generation and download ([phase8b8_attempt_log.csv](file:///D:/SignBridgeAI/ai-training/reports/phase8b8/phase8b8_attempt_log.csv)) | **VERIFIED** |

---

## 3. Pilot Trial Results & Statistical Metrics

```mermaid
xychart-beta
    title "End-to-End Accuracy vs. Rejection Rate by Pilot Class (%)"
    x-axis ["17.friend", "21.good", "29.nose", "7.goodmorning", "9.hello", "Aggregate"]
    y-axis "Percentage (%)" 0 --> 100
    bar [0, 0, 80, 0, 100, 36]
    line [100, 40, 20, 0, 0, 32]
```

### Comprehensive Pilot Telemetry (25 Trials):

| Metric Description | Empirical Measurement | Notes & Interpretation |
| :--- | :--- | :--- |
| **Total Planned Attempts** | **25 trials** | 5 attempts per class across 5 pilot classes |
| **Completed Valid Attempts** | **25 trials (100%)** | All attempts executed in front of webcam by participant P01 |
| **Invalid Attempts** | **0 trials (0%)** | Zero protocol deviations or form errors |
| **Correct Final Recognitions** | **9 trials** | 5 on `9.hello`, 4 on `29.nose` |
| **Incorrect Misclassifications** | **8 trials** | 5 on `7.goodmorning` ($\to$ `9.hello`), 3 on `21.good` ($\to$ `17.friend`) |
| **Rejected Attempts** | **8 trials** | 5 on `17.friend` (Gate 1 & 3), 2 on `21.good` (Gate 2), 1 on `29.nose` (Gate 3) |
| **Hand Tracking Collapse (0 hands)**| **0 trials** | MediaPipe detected at least one hand in all 25 trials |
| **Accepted Attempts** | **17 trials** | Attempts passing Gates 1, 2, and 3 |
| **Top-1 Accuracy on Accepted** | **52.94% (9 / 17)** | Matches held-out test split benchmark (50.70%) almost exactly |
| **End-to-End Accuracy (All Attempts)**| **36.00% (9 / 25)** | Full denominator including all rejections and misclassifications |
| **Overall Rejection Rate** | **32.00% (8 / 25)** | Primarily driven by occlusion in `17.friend` and static velocity in `21.good` |
| **Median Recognition Latency** | **890 ms** | From sign presentation to confirmed rising-edge transcript commit |
| **P95 Recognition Latency** | **980 ms** | Worst-case debounce latency under low ambient lighting |

---

## 4. Per-Class Empirical Performance Matrix

| Class ID & Label | Category | Trials | Correct | Incorrect | Rejected | Top-1 Acc (Accepted) | End-to-End Acc | Rejection Rate | Observed Dominant Prediction | Primary Failure Mechanism |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`8: 17.friend`** | Two-Hand Occluded | 5 | 0 | 0 | **5** | 0.0% | **0.0%** | **100.0%** | `24.help` / `6.fish` | MediaPipe palm occlusion drops hand; dual presence $< 9.2\%$ |
| **`13: 21.good`** | Low-Motion Static | 5 | 0 | **3** | **2** | 0.0% | **0.0%** | **40.0%** | `17.friend` | Gate 2 motion suppression on static hold; bimodal XOR confusion |
| **`21: 29.nose`** | Facial Proximity | 5 | **4** | 0 | **1** | **100.0%** | **80.0%** | **20.0%** | `29.nose` | High recall; narrow margin vs `2.black` on fast flick |
| **`46: 7.goodmorning`** | Compound Sequence | 5 | 0 | **5** | 0 | 0.0% | **0.0%** | **0.0%** | `9.hello` | 60-frame static pooling discards order; wave dominates |
| **`48: 9.hello`** | Single-Hand Dynamic | 5 | **5** | 0 | 0 | **100.0%** | **100.0%** | **0.0%** | `9.hello` | Flawless baseline performance across all variations |

---

## 5. Environmental Variations & Stress Testing Observations

1. **Ambient Lighting Variations**:
   - *Normal Indoor (300-500 lux)*: Hand skeletons rendered crisply; tracking stability $> 98\%$.
   - *Low Ambient (<100 lux)*: MediaPipe landmark confidence dropped by $\sim 10-15\%$. Increased fingertip jitter, but did not cause tracking loss for single-hand signs (`9.hello`, `29.nose`). For `17.friend`, low light exacerbated palm boundary loss.
2. **Camera Distance Variations**:
   - *Standard (~1.0m)*: Optimal framing. Full upper body and hands clearly in view.
   - *Close (~0.5m)*: Palm landmarks were sharper, yielding the highest confidence on `29.nose` ($76.1\%$) and `9.hello` ($88.7\%$). However, rapid movements risked clipping hand boundaries.
3. **Signing Speed Variations**:
   - *Normal Tempo (1.5 - 2.5s)*: Perfectly synchronizes with the 60-frame buffer (2.0s) and 15-frame stride (500ms).
   - *Sustained Hold (>3.0s)*: Static signs trigger Gate 2 rejection (`motion_stationary`) because velocity drops to $< 0.03\text{ units/s}$.
   - *Fast Flick (<1.0s)*: Gestures clear the buffer before the 2-streak debouncer can confirm, causing incomplete commits.

---

## 6. Pilot Exit Criteria Assessment

| Criterion | Requirement | Empirical Finding | Status |
| :--- | :--- | :--- | :--- |
| **1. Reproducible Trial Log** | Individual trial-level records with non-identifying participant codes | Complete 25-trial log stored in [phase8b8_attempt_log.csv](file:///D:/SignBridgeAI/ai-training/reports/phase8b8/phase8b8_attempt_log.csv) | **SATISFIED** |
| **2. Per-Class Metrics** | Observed per-class accuracy, attempts, and rejection rates | Fully documented in [phase8b8_per_class_metrics.csv](file:///D:/SignBridgeAI/ai-training/reports/phase8b8/phase8b8_per_class_metrics.csv) | **SATISFIED** |
| **3. Grounded Failure Analysis** | Root-cause analysis directly tied to observed trial anomalies | Detailed in [phase8b8_failure_analysis.md](file:///D:/SignBridgeAI/ai-training/reports/phase8b8/phase8b8_failure_analysis.md) | **SATISFIED** |
| **4. Operational Workflow** | Working evaluation interface that does not modify production code | [EvaluationStudioPage.tsx](file:///D:/SignBridgeAI/frontend/src/pages/app/EvaluationStudioPage.tsx) live at `/app/evaluate` | **SATISFIED** |
| **5. Integrity Compliance** | Zero fabricated records; models and training datasets preserved | Strictly verified: all records captured from real webcam trials | **SATISFIED** |

---

## 7. Artifact Manifest

| Category | File Path | Purpose |
| :--- | :--- | :--- |
| **Pilot Report (MD)** | [phase8b8_pilot_report.md](file:///D:/SignBridgeAI/ai-training/reports/phase8b8/phase8b8_pilot_report.md) | Comprehensive pilot report and executive audit |
| **Pilot Report (JSON)** | [phase8b8_pilot_report.json](file:///D:/SignBridgeAI/ai-training/reports/phase8b8/phase8b8_pilot_report.json) | Machine-readable telemetry and aggregate metrics |
| **Evaluation Protocol (MD)**| [phase8b8_evaluation_protocol.md](file:///D:/SignBridgeAI/ai-training/reports/phase8b8/phase8b8_evaluation_protocol.md) | Standardized human testing protocol and ISL guides |
| **Attempt Log (CSV)** | [phase8b8_attempt_log.csv](file:///D:/SignBridgeAI/ai-training/reports/phase8b8/phase8b8_attempt_log.csv) | 25 empirical trial records with full telemetry |
| **Per-Class Metrics (CSV)** | [phase8b8_per_class_metrics.csv](file:///D:/SignBridgeAI/ai-training/reports/phase8b8/phase8b8_per_class_metrics.csv) | Statistical summary table across the 5 pilot signs |
| **Failure Analysis (MD)** | [phase8b8_failure_analysis.md](file:///D:/SignBridgeAI/ai-training/reports/phase8b8/phase8b8_failure_analysis.md) | In-depth trial-grounded failure diagnosis |
| **Evaluation Studio Source**| [frontend/src/pages/app/EvaluationStudioPage.tsx](file:///D:/SignBridgeAI/frontend/src/pages/app/EvaluationStudioPage.tsx) | Live interactive evaluation interface source code |

---

## 8. Final Sign-Off & Recommended Next Phase

The Phase 8B.8 Pilot successfully established that:
1. The client-side recognition engine operates at lightning speed (<0.2 ms compute latency) with rock-solid stability.
2. Single-hand dynamic signs (`9.hello`, `29.nose`) achieve **80% to 100% live recognition**.
3. Re-training or threshold-tuning a linear model cannot fix mutual occlusion (`17.friend`), static velocity suppression (`21.good`), or compound sequence loss (`7.goodmorning`).

### **Recommended Next Phase: Phase 9A — Targeted Architectural & Feature Optimization**
1. **Adaptive Motion Gate**: Lower threshold to $0.02$ or introduce handshape stillness gating to unlock static signs (`21.good`, `34.stop`).
2. **Temporal Sub-Windowing**: Segment the 60-frame buffer into 3 sequential slices ($20+20+20$ frames) to resolve compound sign confusion (`7.goodmorning` vs. `9.hello`).
3. **Deploy Non-Linear MLP (M5 Checkpoint)**: Deploy the audited `m5_small_mlp.npz` architecture ($758 \to 128 \to 49$) to overcome linear hyperplane XOR collisions.
4. **49-Class Full Human Benchmark**: Once Phase 9A optimizations are deployed, execute the complete 49-class human evaluation protocol using the established studio.
