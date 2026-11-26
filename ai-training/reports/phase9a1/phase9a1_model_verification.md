# PHASE 9A.1: MODEL VERIFICATION & MATHEMATICAL PARITY REPORT

**Audit Date**: 2026-11-25  
**Evaluation Scope**: Checkpoint Integrity, Scaler Compatibility, Architecture Parity, and Inference Latency Benchmarking  
**Auditor**: Antigravity Autonomous Scientific Verification Engine  
**Status**: COMPLETE — ALL VERIFICATION GATES PASSED

---

## 1. Executive Summary

As part of **Phase 9A.1 (M5 vs M1 Controlled Paired Evaluation)**, a rigorous structural and mathematical audit was conducted on the preserved M5 Small Multi-Layer Perceptron (MLP) checkpoint (`ai-training/models/phase8b6b/m5_small_mlp.npz`) and the active production M1 champion (`frontend/public/models/sign_classifier.json`).

The objective was to verify that:
1. Both models operate on the exact same 758-dimensional input feature space.
2. Both models utilize the identical 49-class vocabulary and class indexing.
3. Feature standardization parameters (`scaler_mean`, `scaler_scale`) are bit-for-bit identical ($0.0$ difference across all 758 dimensions).
4. An isolated TypeScript inference engine (`m5Evaluator.ts`) replicates the forward pass of the Python reference implementation within strict numerical tolerance ($< 10^{-6}$ max absolute probability difference).
5. Inference latency on the client runtime comfortably satisfies real-time execution constraints ($< 20$ ms budget; target $\le 1$ ms).

### Audit Verdict: **PASS**

---

## 2. Model Checkpoint & Architecture Specifications

| Property | M1 Champion (Production) | M5 Candidate (Evaluation) | Parity Status |
| :--- | :--- | :--- | :--- |
| **Model Type** | L2-Regularized Softmax Logistic Regression | Small Multi-Layer Perceptron (MLP) | Controlled Architecture Comparison |
| **Source Checkpoint** | `frontend/public/models/sign_classifier.json` | `ai-training/models/phase8b6b/m5_small_mlp.npz` | Verified Loaded |
| **Export Checkpoint** | `ai-training/models/phase8b6b/phase8b6b_champion.json` | `frontend/public/models/m5_evaluator.json` | Verified Created |
| **Input Dimensions** | 758 float32 | 758 float32 | **Exact Match (758-D)** |
| **Hidden Layers** | None (Direct Linear Mapping) | 1 Hidden Layer (128 units, ReLU) | Verified |
| **Layer 1 Weights** | $(758, 49)$ | $W_1: (758, 128)$, $b_1: (128)$ | Verified Valid Shapes |
| **Layer 2 Weights** | — | $W_2: (128, 49)$, $b_2: (49)$ | Verified Valid Shapes |
| **Output Classes** | 49 ISL Classes | 49 ISL Classes | **Exact Match (49 Classes)** |
| **Class Ordering** | Identical 0 to 48 | Identical 0 to 48 | **Bit-for-Bit Identical** |
| **Scaler Dimensions** | Mean: 758, Scale: 758 | Mean: 758, Scale: 758 | **Exact Match** |
| **Scaler Diff** | — | $\max \| \mu_{M1} - \mu_{M5} \| = 0.0$, $\max \| s_{M1} - s_{M5} \| = 0.0$ | **Exact $0.0$ Difference** |

---

## 3. Preprocessing & Scaler Verification

Both models were trained using StandardScaler statistics fitted exclusively on the 631 training samples of the audited Phase 8B.5B split (`ai-training/data/staged_phase8b5b_split/train.npz`).

To ensure zero leakage and zero preprocessing divergence:
- Scaler mean vector $\mu \in \mathbb{R}^{758}$ was compared between M1 and M5. Max absolute difference: **$0.00000000$**.
- Scaler scale vector $\sigma \in \mathbb{R}^{758}$ was compared between M1 and M5. Max absolute difference: **$0.00000000$**.
- Class name strings array was checked element by element from index 0 (`1.bird`) to index 48 (`9.hello`). All 49 string labels match identically with zero ordering drift.

---

## 4. Mathematical Parity Verification

An isolated TypeScript evaluator was developed at [frontend/src/utils/m5Evaluator.ts](file:///d:/SignBridgeAI/frontend/src/utils/m5Evaluator.ts) implementing:
1. Feature vector standardization: $\tilde{x}_i = \frac{x_i - \mu_i}{\sigma_i}$.
2. Hidden layer projection: $h = \max(0, \tilde{x} W_1 + b_1)$.
3. Output projection: $z = h W_2 + b_2$.
4. Numerically stable softmax: $p_k = \frac{\exp(z_k - \max(z))}{\sum_j \exp(z_j - \max(z))}$.
5. Preserved Phase 8B.6B rejection gates (hand presence, motion energy, confidence threshold).

### Test Setup & Numerical Results

20 test vectors drawn randomly from the frozen test set (`ai-training/data/staged_phase8b5b_split/test.npz`) were evaluated in Python using Scikit-Learn's native `MLPClassifier.predict_proba` and compared against the TypeScript implementation executed under Node.js v24.19.0.

- **Sample Count**: 20 test vectors
- **Max Logit Difference**: $7.63 \times 10^{-6}$ (attributed to single vs double precision floating-point matrix accumulation)
- **Max Absolute Probability Difference**: **$6.56 \times 10^{-7}$** (Threshold: $< 1.0 \times 10^{-6}$)
- **Top-1 Predicted Class Mismatches**: **0 / 20 (0.00%)**
- **Parity Verdict**: **PASS**

Full verification details are archived in [ai-training/reports/phase9a1/phase9a1_parity_report.json](file:///d:/SignBridgeAI/ai-training/reports/phase9a1/phase9a1_parity_report.json).

---

## 5. Client-Side Inference Latency Benchmarking

To ensure that the non-linear MLP architecture ($758 \to 128 \to 49$) does not introduce frame drops or latency regressions in browser execution, a 1,000-iteration micro-benchmark was conducted under identical hardware conditions.

### Latency Benchmark Results (1,000 iterations, 100 warmup)

| Metric | M1 Linear Softmax | M5 Small MLP | Delta (M5 - M1) | Ratio (M5 / M1) |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Execution Time** | 0.0390 ms | 0.1115 ms | +0.0725 ms | 2.86x |
| **Median Execution Time** | **0.0380 ms** | **0.1087 ms** | **+0.0707 ms** | **2.86x** |
| **90th Percentile (P90)** | 0.0392 ms | 0.1120 ms | +0.0728 ms | 2.86x |
| **95th Percentile (P95)** | 0.0431 ms | 0.1209 ms | +0.0778 ms | 2.80x |
| **99th Percentile (P99)** | **0.0730 ms** | **0.1721 ms** | **+0.0991 ms** | **2.36x** |
| **Minimum Time** | 0.0364 ms | 0.1061 ms | +0.0697 ms | 2.91x |
| **Maximum Time** | 0.1050 ms | 0.3785 ms | +0.2735 ms | 3.60x |

Full benchmark data is archived in [ai-training/reports/phase9a1/phase9a1_latency_benchmark.json](file:///d:/SignBridgeAI/ai-training/reports/phase9a1/phase9a1_latency_benchmark.json).

### Latency Analysis & Budget Compliance
- **Target Budget**: $< 20.0$ ms per classification.
- **Frame Interval (30 FPS)**: 33.3 ms total frame budget.
- **M5 Execution**: $0.1087$ ms consumes $< 0.33\%$ of the available frame budget.
- **Conclusion**: While M5 is ~2.86x slower than the ultra-lean M1 linear classifier, both run well under $0.20$ ms. Computation latency is completely negligible compared to MediaPipe HandLandmarker inference (~15–22 ms). Latency is **NOT** a blocking factor for M5.

---

## 6. Preservation Confirmation

- Production model file `frontend/public/models/sign_classifier.json` was **NOT** modified (SHA-256 remains `c8068a3fedd85645d736d16ebde450bd442970784adba40fdacb8970a4535274`).
- No production inference paths or live application components in `SignToTextPage.tsx` were modified.
- M5 was encapsulated in an isolated development evaluator module (`m5Evaluator.ts`) and loaded alongside M1 only within `EvaluationStudioPage.tsx`.
