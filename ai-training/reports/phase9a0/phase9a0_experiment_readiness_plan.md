# Phase 9A.0: Evidence-Based Experiment Readiness & Architectural Optimization Plan

**Date**: 2026-11-25  
**Status**: Approved Optimization Roadmap  
**Target Platform**: SIGNBRIDGE AI Real-Time Production Inference Engine  
**Baseline Model**: M1 Champion (`frontend/public/models/sign_classifier.json`)  

---

## 1. Executive Summary & Optimization Philosophy

Based on the verified Phase 8B.9 empirical webcam audit (147 trials, 49 classes), the production system suffers from five well-defined failure mechanisms. Rather than attempting a monolithic rewrite, Phase 9 will execute a **strictly controlled, ranked series of isolated experiments**.

### Key Architectural Principles:

- **Zero Speculative Engineering**: Every technical change targets a proven, empirically documented failure mode.
- **Isolation of Concerns**: Low-risk runtime heuristic adjustments are evaluated before introducing non-linear model changes; model changes are evaluated before modifying feature vector dimensionality.
- **Regression Guardrails**: Every optimization must preserve the 22 reliable signs while improving the 14 unreliable signs.

---

## 2. Ranked Experiment Roadmap

| Rank | Experiment Name | Primary Target | Requires Retraining? | Changes 758-D Schema? | Estimated Effort | Risk Tier |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **P1** | **Adaptive Motion Gate** | Static Signs (`21.good`, `34.stop`, `45.hand`) | **NO** | **NO** | Low | Low |
| **P2** | **M5 Non-Linear MLP Evaluation** | Bimodal XOR & Facial Cluster Separation | **NO** (Pre-trained) | **NO** | Low-Med | Low |
| **P3** | **Two-Hand Occlusion Smoothing** | Dual Contact (`17.friend`, `42.work`, `47.umberlla`) | **NO** | **NO** | Med | Med |
| **P4** | **Temporal Sub-Window Pooling** | Compound Signs (`7.goodmorning`, `20.brother`) | **YES** | **YES** | High | Med-High |
| **P5** | **Facial Anchor Offsets** | Facial Proximity Disentanglement (`teeth`, `black`) | **YES** | **YES** | High | High |

---

## 3. Deep-Dive Experiment Specifications

### Experiment 1 (Priority 1): Adaptive Motion Gate for Static Signs

- **Observed Failure Targeted**: Gate 2 (`motion_stationary`) enforces mean velocity $\bar{v} \ge 0.05$ across 60 frames. Static signs held firm drop to $\bar{v} \approx 0.021 - 0.028$, causing 100% false rejections on static holds of `21.good` and 33% on `34.stop` / `45.hand`.
- **Proposed Technical Change**: In `frontend/src/utils/signClassifier.ts`, compute handshape landmark variance $\sigma_{\text{shape}}^2$. If landmark presence $\ge 0.75$ and handshape is stable ($\sigma_{\text{shape}}^2 < 0.015$), relax velocity threshold to $0.015$ or bypass Gate 2 if top prediction is a recognized static sign.
- **Requires Retraining**: No. Purely a runtime heuristic in TypeScript.
- **Changes 758-D Schema**: No.
- **Test Procedures**: Re-run on recorded Phase 8B.9 static trials; test live idle hands vs intentional static holds.
- **Regression Risks**: Idle hands resting in frame might falsely commit.
- **Success Criterion**: Zero Gate 2 false rejections on sustained holds of `21.good`, `34.stop`, `45.hand`, while maintaining $>95\%$ rejection when idle.


### Experiment 2 (Priority 2): M5 Non-Linear Classifier (MLP Checkpoint Evaluation)

- **Observed Failure Targeted**: M1 linear regression cannot draw non-linear boundaries. In bimodal left/right hand executions (`21.good` vs `17.friend`), linear hyperplane splits through intermediate classes. In facial proximity (`16.eat`, `2.black`, `44.teeth`), all signs collapse into `29.nose`.
- **Proposed Technical Change**: Deploy the preserved checkpoint `ai-training/models/phase8b6b/m5_small_mlp.npz` ($758 \to 128 \to 49$, ReLU activations, 103,473 parameters). Export to JSON (`sign_classifier_mlp.json`) and add MLP forward pass in `signClassifier.ts`.
- **Requires Retraining**: No (checkpoint already trained and validated in Phase 8B.6B).
- **Changes 758-D Schema**: No (trained on the exact same 758-D features).
- **Test Procedures**: Mathematical parity test against Python reference ($<10^{-6}$ max absolute error); client CPU latency benchmark; offline replay across all 147 Phase 8B.9 trials.
- **Regression Risks**: Slight accuracy drop on simple linear signs (M5 test accuracy was 47.18% vs M1 50.70%).
- **Success Criterion**: Non-linear separation of `44.teeth` and `2.black` from `29.nose`; client latency $<20\text{ ms}$.


### Experiment 3 (Priority 3): Two-Hand Tracking Persistence & Gate 1 Presence Relaxation

- **Observed Failure Targeted**: When hands clasp in `17.friend`, MediaPipe drops tracking of the non-dominant hand. Left hand presence collapses to $5-17\%$, triggering Gate 1 (`presence_too_low` $<0.25$) in 100% of standard attempts.
- **Proposed Technical Change**: In `featureBuilder.ts`, maintain a 5-frame persistence buffer when wrist Euclidean distance $<0.12$. In `signClassifier.ts`, compute dual presence as $\frac{\text{left} + \text{right}}{2} \ge 0.25$ or allow asymmetric presence if two-hand proximity is detected.
- **Requires Retraining**: No.
- **Changes 758-D Schema**: No.
- **Test Procedures**: Live webcam replay on `17.friend`, `42.work`, and `47.umberlla`.
- **Regression Risks**: Single-hand signs near body could falsely trigger dual presence.
- **Success Criterion**: Dual presence on `17.friend` increases to $\ge 35\%$, eliminating Gate 1 false rejections.


### Experiment 4 (Priority 4): Chronological Temporal Sub-Window Features

- **Observed Failure Targeted**: 60-frame global summary pooling destroys temporal sequence order. Compound signs (`7.goodmorning` = Good $\to$ Wave) collapse into atomic waves (`9.hello`) in 100% of trials.
- **Proposed Technical Change**: Partition the 60-frame buffer into 3 chronological windows (Frames 0–19, 20–39, 40–59) and extract sub-window summary statistics.
- **Requires Retraining**: Yes. Requires re-extracting features on 915 clean samples and retraining M1 and M5.
- **Changes 758-D Schema**: Yes (increases feature dimension to $3 \times 252$ or $758 + \Delta$).
- **Test Procedures**: Full train/val/test retraining protocol; held-out test evaluation on compound signs.
- **Regression Risks**: Overfitting risk due to larger feature dimension with $N=631$ training samples.
- **Success Criterion**: `7.goodmorning` test recall $\ge 50\%$, with false positive collapse into `hello` reduced to $<20\%$.


### Experiment 5 (Priority 5): Facial-Proximity Anchor Offset Representation

- **Observed Failure Targeted**: Coarse hand-to-head coordinates cannot distinguish mouth, nose, teeth, and eyebrow.
- **Proposed Technical Change**: Incorporate explicit facial reference landmarks (nose tip, mouth centroid, eyebrow center) to compute direct displacement vectors.
- **Requires Retraining**: Yes.
- **Changes 758-D Schema**: Yes.
- **Test Procedures**: Landmark extraction on clean video set; retraining and evaluation.
- **Regression Risks**: High complexity; dependency on facial landmark detector stability.
- **Success Criterion**: Disentangling `44.teeth` from `29.nose` with $>40\%$ recall across facial signs.
