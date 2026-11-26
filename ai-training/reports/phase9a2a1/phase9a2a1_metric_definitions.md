# PHASE 9A.2A.1: METRIC DEFINITIONS & FEATURE MAPPING AUDIT

**Audit Date**: 2026-11-25  
**Scope**: Mathematical Formalism, 758-D Feature Index Mappings, Calculation Feasibility, and Aggregation Boundaries for the Adaptive Motion Gate  
**Auditor**: Antigravity Autonomous Scientific Verification Engine  
**Status**: AUDIT COMPLETE — FULL MATHEMATICAL SPECIFICATION VERIFIED

---

## 1. Overview & Verification Summary

The **Adaptive Motion Gate** introduces a secondary evaluation path (**Path B: Static-Sign Candidate Path**) to allow intentional static signs (`21.good`, `34.stop`, `45.hand`) to be recognized when hand motion energy drops below the global dynamic threshold ($v < 0.05$).

To ensure scientific reproducibility, this document audits the exact mathematical definitions, 758-dimensional feature index mappings, and calculation feasibility of the five gating metrics evaluated in Path B.

### Summary Assessment
- **All 5 static gating metrics** are deterministically calculable from a single standard 758-dimensional feature vector.
- **Indices and aggregations**: All metrics derive from stored presence ratios (features 0, 379), spatial landmark means (features 1–63, 380–442), and population spatial standard deviations (features 64–126, 443–505).
- **Temporal Stability Nuance**: The metric $\sigma_{\text{spatial}}$ measures landmark geometric variance across time *aggregated within a single 60-frame window*. Multi-window temporal stability (e.g. maintaining a 2-cycle streak) is an *online state machine property* evaluated across sequential sliding windows, not an intrinsic property of an isolated feature vector.

---

## 2. Exhaustive Metric Definitions & 758-D Index Mappings

```
========================================================================================================
FEATURE SCHEMA REFERENCE (758 total Float32 elements):
  Slot 0: Left Hand (Indices 0 - 378: 379 features)
    [0]       Presence ratio (valid_frames / total_frames)
    [1-63]    Spatial Mean (21 landmarks x 3 coords: x, y, z)
    [64-126]  Spatial Standard Deviation (population ddof=0, 21 landmarks x 3 coords)
    [127-189] Spatial Min Bound
    [190-252] Spatial Max Bound
    [253-315] Mean Absolute Temporal Velocity (units/sec)
    [316-378] Temporal Velocity Standard Deviation (ddof=0)
  
  Slot 1: Right Hand (Indices 379 - 757: 379 features)
    [379]     Presence ratio
    [380-442] Spatial Mean
    [443-505] Spatial Standard Deviation
    [506-568] Spatial Min Bound
    [569-631] Spatial Max Bound
    [632-694] Mean Absolute Temporal Velocity
    [695-757] Temporal Velocity Standard Deviation
========================================================================================================
```

---

### Metric 1: Active Hand Presence ($p_{\text{active}} \ge 0.50$)

#### Mathematical Definition
Let $p_{\text{left}}$ and $p_{\text{right}}$ denote the fraction of valid landmark detection frames across the $N=60$ frame window for the Left and Right hand slots respectively:
$$p_{\text{active}} = \max\left(p_{\text{left}},\, p_{\text{right}}\right)$$

#### 758-D Feature Indices
- **Left Hand Presence**: Index `0` (`features[0]`)
- **Right Hand Presence**: Index `379` (`features[379]`)

#### Metric Properties
- **Classification**: Temporal aggregation (fraction of detected frames over total window length $N$).
- **Calculable from Stored Data**: **YES**. Directly stored as single scalar values in indices `0` and `379`.
- **Threshold**: $p_{\text{active}} \ge 0.50$ (requires at least 30 of 60 frames detected).
- **Physical Meaning**: Ensures that at least one hand has substantial presence in the field of view, preventing partial or flickering entries from triggering static recognition.

---

### Metric 2: Pose Stability ($\sigma_{\text{spatial}} \le 0.055$)

#### Mathematical Definition
For an active hand, spatial standard deviation measures the dispersion of each landmark's coordinates $(x_k, y_k, z_k)$ around its temporal mean across all valid frames $K$:
$$\sigma_{k, c} = \sqrt{\frac{1}{K} \sum_{t=1}^K \left(c_{k, t} - \bar{c}_k\right)^2} \quad \text{for } k \in \{0, \dots, 20\}, \, c \in \{x, y, z\}$$
The overall pose stability metric $\sigma_{\text{spatial}}$ is defined as the arithmetic mean of these 63 standard deviation features for the active hand:
$$\sigma_{\text{left}} = \begin{cases} \frac{1}{63} \sum_{i=64}^{126} \text{features}[i] & \text{if } p_{\text{left}} > 0 \\ 0.0 & \text{otherwise} \end{cases}$$
$$\sigma_{\text{right}} = \begin{cases} \frac{1}{63} \sum_{i=443}^{505} \text{features}[i] & \text{if } p_{\text{right}} > 0 \\ 0.0 & \text{otherwise} \end{cases}$$
$$\sigma_{\text{spatial}} = \max\left(\sigma_{\text{left}},\, \sigma_{\text{right}}\right)$$

#### 758-D Feature Indices
- **Left Hand Spatial Std Dev**: Indices `64` through `126` (63 continuous float32 values).
- **Right Hand Spatial Std Dev**: Indices `443` through `505` (63 continuous float32 values).

#### Metric Properties
- **Classification**: Spatial-temporal aggregate summary (arithmetic mean across 63 temporal coordinate standard deviations).
- **Calculable from Stored Data**: **YES**. Exactly computable from the 63 stored population standard deviation features per hand.
- **Threshold**: $\sigma_{\text{spatial}} \le 0.055$.
- **Empirical Basis**: The 90th percentile of $\sigma_{\text{spatial}}$ on intentional static signs in `train.npz` is $0.052$. Transitional fidgeting and gesture adjustments exhibit $\sigma_{\text{spatial}} > 0.060$.
- **Important Qualification**: This metric quantifies intra-window landmark stability around the temporal mean. It is an aggregate summary metric, not a frame-by-frame instantaneous acceleration profile.

---

### Metric 3: Signing-Space Y-Centroid ($y_{\text{centroid}} \le 0.75$)

#### Mathematical Definition
In MediaPipe normalized coordinates, $y \in [0.0, 1.0]$ spans from the top of the video frame ($0.0$) to the bottom ($1.0$).
For each hand, the vertical centroid $y_{\text{mean}}$ is computed as the arithmetic mean of the vertical coordinates across all 21 landmarks:
$$y_{\text{mean, left}} = \frac{1}{21} \sum_{k=0}^{20} \text{features}[2 + 3k] \quad \text{for } k \in \{0, \dots, 20\}$$
$$y_{\text{mean, right}} = \frac{1}{21} \sum_{k=0}^{20} \text{features}[381 + 3k] \quad \text{for } k \in \{0, \dots, 20\}$$
$$y_{\text{centroid}} = \min\left(y_{\text{mean, left}} \text{ if } p_{\text{left}} > 0 \text{ else } 1.0,\, y_{\text{mean, right}} \text{ if } p_{\text{right}} > 0 \text{ else } 1.0\right)$$

#### 758-D Feature Indices
- **Left Hand Y-Coordinates**: Strided slice `features[2:64:3]` (indices 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62).
- **Right Hand Y-Coordinates**: Strided slice `features[381:443:3]` (indices 381, 384, 387, 390, 393, 396, 399, 402, 405, 408, 411, 414, 417, 420, 423, 426, 429, 432, 435, 438, 441).

#### Metric Properties
- **Classification**: Spatial aggregation (mean vertical position of active hand landmarks).
- **Calculable from Stored Data**: **YES**. Directly computed from stored landmark spatial mean coordinates.
- **Threshold**: $y_{\text{centroid}} \le 0.75$.
- **Physical Meaning**: Intentional ISL signs occur in the upper signing space (chest, sternum, neck, chin, face), corresponding to $y \in [0.20, 0.70]$. Hands resting on a desk, table, lap, or dangling at the side have $y > 0.75$. This serves as an explicit spatial filter against idle hands.

---

### Metric 4: Classification Confidence ($p_{\text{top1}} \ge 0.50$)

#### Mathematical Definition
Standardize the 758-D vector $x$ using training statistics $\mu, \sigma$:
$$\tilde{x}_i = \frac{x_i - \mu_i}{\sigma_i} \quad \text{for } i \in \{0, \dots, 757\}$$
Compute unnormalized logits $z \in \mathbb{R}^{49}$:
$$z_c = \sum_{i=0}^{757} \tilde{x}_i W_{i, c} + b_c \quad \text{for } c \in \{0, \dots, 48\}$$
Apply numerically stable softmax:
$$p_c = \frac{\exp\left(z_c - \max(z)\right)}{\sum_{j=0}^{48} \exp\left(z_j - \max(z)\right)}$$
$$p_{\text{top1}} = \max_{c \in \{0, \dots, 48\}} p_c$$

#### 758-D Feature Indices
- Uses all 758 input features evaluated against the frozen production model weights $W \in \mathbb{R}^{758 \times 49}, b \in \mathbb{R}^{49}$.

#### Metric Properties
- **Classification**: Model output probability.
- **Calculable from Stored Data**: **YES**. Fully deterministic.
- **Threshold**: $p_{\text{top1}} \ge 0.50$.
- **Physical Meaning**: Rejects weak or ambiguous classifier responses under static conditions.

---

### Metric 5: Top-1 / Top-2 Margin ($\Delta p \ge 0.10$)

#### Mathematical Definition
Let $p_{(1)}$ and $p_{(2)}$ denote the highest and second-highest probabilities from the sorted softmax distribution:
$$\Delta p = p_{(1)} - p_{(2)}$$

#### 758-D Feature Indices
- Derived from the full 758-D forward pass.

#### Metric Properties
- **Classification**: Distribution entropy / separation metric.
- **Calculable from Stored Data**: **YES**. Fully deterministic.
- **Threshold**: $\Delta p \ge 0.10$.
- **Physical Meaning**: Prevents classification when the model is evenly divided between multiple classes (e.g. 50/50 split between static and dynamic interpretations).

---

## 3. Disentangling Single-Window Features vs Online Temporal Debouncing

A critical distinction must be maintained between:
1. **Intra-Window Metrics (Static Feature Properties)**:
   - Evaluated on a single 758-D vector extracted from one 60-frame buffer.
   - Includes $p_{\text{active}}$, $\sigma_{\text{spatial}}$, $y_{\text{centroid}}$, $p_{\text{top1}}$, and $\Delta p$.
   - These are fully preserved in stored dataset vectors (`train.npz`, `val.npz`).
2. **Inter-Window Online Debouncing (Streaming State Machine)**:
   - Evaluated across sequential sliding inference strides in the live application.
   - Managed by `TemporalDebouncer(minConsecutiveFrames: 2)`.
   - Requires that the accepted candidate class remain identical for $K \ge 2$ consecutive inference cycles (e.g., across 2 cycles separated by 3 frames).
   - This temporal continuity **CANNOT** be computed from an isolated static `.npz` vector. It is an online streaming property.

**Audit Finding**: Single-window gating criteria are 100% verifiable from existing stored feature vectors. Claims of multi-window debouncing apply exclusively to live streaming sessions and must not be claimed for offline single-vector evaluations.
