# SIGNBRIDGE AI — PHASE 8B.5C: INDEPENDENT SPLIT ARTIFACT INTEGRITY AUDIT

## Executive Summary & Final Verdict

An independent, strictly read-only audit of the Phase 8B.5B split artifacts located in `ai-training/data/staged_phase8b5b_split/` has been completed.
Every sample, feature vector, session group, and label was recomputed directly from the binary NPZ files and compared against the Phase 8B.4.1 clean dataset.

### Final Audit Decision: **PASS**
- **Sample Integrity**: Exactly 915 disjoint samples (Train: 631, Val: 142, Test: 142). Zero missing, zero duplicated.
- **Exact Feature Vector Equality**: 100% of feature vectors match `features_tabular.npz` byte-for-byte (`np.array_equal == True`).
- **Zero Group Leakage**: Exactly 0 of the 282 atomic session bursts cross partitions. Group isolation is 100% intact.
- **Full Class Coverage**: Exactly 49/49 classes present in Train, Validation, and Test partitions. All classes have strictly $\ge 2$ samples in Val and Test.
- **Inferred Signer Separation**: Exactly 0 within-class Train <-> Test overlaps detected for Krishnaraj, Manideep, and Nachiketh.
- **Deterministic Reproducibility**: Allocation procedure re-executed in-memory with seed 42 yielded 100% identical sample assignments.

---

## 1. NPZ Artifacts Direct Inspection

The three partition archives were loaded directly without relying on Phase 8B.5B report metadata:

| Partition Archive | Sample Count | Feature Shape | Feature Dtype | Labels Shape / Dtype | Numerical Integrity | Keys Present |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `train.npz` | **631** | `[631, 758]` | `float32` | `[631]` (`int64`) | 0 NaNs, 0 Infs | `X, y, sample_ids, video_names, class_names` |
| `val.npz` | **142** | `[142, 758]` | `float32` | `[142]` (`int64`) | 0 NaNs, 0 Infs | `X, y, sample_ids, video_names, class_names` |
| `test.npz` | **142** | `[142, 758]` | `float32` | `[142]` (`int64`) | 0 NaNs, 0 Infs | `X, y, sample_ids, video_names, class_names` |

### Sample Disjointness Check:
- $\text{Train} \cap \text{Val} = \emptyset$ (0 shared samples)
- $\text{Train} \cap \text{Test} = \emptyset$ (0 shared samples)
- $\text{Val} \cap \text{Test} = \emptyset$ (0 shared samples)
- $\text{Train} \cup \text{Val} \cup \text{Test} = 915$ unique samples (100% accounting)

---

## 2. Complete Coverage & Exact Feature Vector Equality

Every single sample in `train.npz`, `val.npz`, and `test.npz` was evaluated against its counterpart in `staged_clean_features/features_tabular.npz`:
- **Exact Feature Array Match**: Evaluated via `np.array_equal(partition_X, clean_X)`. Result: **100% byte-for-byte exact match (915 / 915)**.
- **Label & Class Name Match**: 100% identical integer class IDs (0..48) and string class names.
- **Source Hash Match**: 100% of samples match their SHA-256 hash in `phase8b4_feature_source_mapping.csv`.
- **Zero Alteration**: Feature vectors were verified to have been subsetted directly from the clean dataset with zero modification, interpolation, or re-scaling.

---

## 3. Session-Burst Group Isolation Audit

The atomic group assignments from `phase8b5b_group_assignment.csv` were verified against individual sample assignments in `phase8b5b_split_manifest.csv`:

| Metric | Target Specification | Measured Value | Audit Status |
| :--- | :---: | :---: | :---: |
| Total Atomic Session Bursts | Exactly 282 | **282** | **VERIFIED** |
| Training Session Bursts | In Train only | **184** bursts | **VERIFIED** |
| Validation Session Bursts | In Val only | **49** bursts | **VERIFIED** |
| Testing Session Bursts | In Test only | **49** bursts | **VERIFIED** |
| **Session Bursts Crossing Partitions** | **Strictly 0** | **0** | **ZERO LEAKAGE (PASSED)** |

**Conclusion**: Every consecutive take within a session burst is strictly co-located in the same partition. Take-repetition leakage is completely eliminated.

---

## 4. Possible-Signer Separation Audit

An independent audit of possible contributor overlaps within each class was conducted:
- **Signer_Krishnaraj**: **0 within-class Train <-> Test overlaps** across all 46 classes where Krishnaraj is present.
- **Signer_Manideep**: **0 within-class Train <-> Test overlaps** across all 47 classes where Manideep is present.
- **Signer_Nachiketh**: **0 within-class Train <-> Test overlaps** across all 25 classes where Nachiketh is present.
- **Unknown_Signer**: Present across generic/numeric batches in multiple classes, as documented in Phase 8B.5A.

> [!IMPORTANT]
> **Scientific Generalization Boundary**: While within-class take-repetition and possible-signer contamination have been eliminated, **subject-independent generalization cannot be certified** because 62.6% of the raw dataset lacks contributor ground truth. Model performance must be reported as **'Cross-Session Take-Independent'**.

---

## 5. Class Coverage & Balance Verification

Class counts were recomputed directly from the `class_names` array of each partition archive:
- **Classes in Training**: Exactly **49 / 49 (100%)**
- **Classes in Validation**: Exactly **49 / 49 (100%)**
- **Classes in Testing**: Exactly **49 / 49 (100%)**
- **Validation per-class range**: 2 to 3 samples (Minimum $\ge 2$ guaranteed across all 49 classes)
- **Testing per-class range**: 2 to 3 samples (Minimum $\ge 2$ guaranteed across all 49 classes)
- **Recomputed vs CSV Match**: 100% agreement with `phase8b5b_class_distribution.csv`.

| Class ID | Class Name | Total Samples | Train (NPZ) | Val (NPZ) | Test (NPZ) | Val $\ge 2$? | Test $\ge 2$? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `1.bird` | 17 | **11** | **3** | **3** | True | True |
| 1 | `10.youareperfect` | 20 | **14** | **3** | **3** | True | True |
| 2 | `11.monsoon` | 21 | **15** | **3** | **3** | True | True |
| 3 | `12.afternoon` | 15 | **10** | **3** | **2** | True | True |
| 4 | `13.angry` | 19 | **13** | **3** | **3** | True | True |
| 5 | `14.bad` | 17 | **11** | **3** | **3** | True | True |
| 6 | `15.boy` | 16 | **12** | **2** | **2** | True | True |
| 7 | `16.eat` | 20 | **14** | **3** | **3** | True | True |
| 8 | `17.friend` | 20 | **14** | **3** | **3** | True | True |
| 9 | `18.drinking` | 20 | **14** | **3** | **3** | True | True |
| 10 | `19.girl` | 20 | **14** | **3** | **3** | True | True |
| 11 | `2.black` | 17 | **11** | **3** | **3** | True | True |
| 12 | `20.brother` | 17 | **11** | **3** | **3** | True | True |
| 13 | `21.good` | 20 | **14** | **3** | **3** | True | True |
| 14 | `22.father` | 20 | **14** | **3** | **3** | True | True |
| 15 | `23.evening` | 19 | **13** | **3** | **3** | True | True |
| 16 | `24.help` | 18 | **12** | **3** | **3** | True | True |
| 17 | `25.mother` | 20 | **14** | **3** | **3** | True | True |
| 18 | `26.NAME` | 16 | **12** | **2** | **2** | True | True |
| 19 | `27.Night` | 20 | **14** | **3** | **3** | True | True |
| 20 | `28.Music` | 17 | **11** | **3** | **3** | True | True |
| 21 | `29.nose` | 20 | **14** | **3** | **3** | True | True |
| 22 | `3.cat` | 16 | **12** | **2** | **2** | True | True |
| 23 | `30.sleep` | 17 | **11** | **3** | **3** | True | True |
| 24 | `31.sit` | 20 | **14** | **3** | **3** | True | True |
| 25 | `32.sorry` | 17 | **11** | **3** | **3** | True | True |
| 26 | `33.stand` | 20 | **14** | **3** | **3** | True | True |
| 27 | `34.stop` | 20 | **14** | **3** | **3** | True | True |
| 28 | `35.student` | 19 | **13** | **3** | **3** | True | True |
| 29 | `36.study` | 20 | **14** | **3** | **3** | True | True |
| 30 | `37.teacher` | 17 | **11** | **3** | **3** | True | True |
| 31 | `38.thankyou` | 20 | **14** | **3** | **3** | True | True |
| 32 | `39.today` | 20 | **14** | **3** | **3** | True | True |
| 33 | `4.cow` | 17 | **11** | **3** | **3** | True | True |
| 34 | `40.tommorow` | 20 | **14** | **3** | **3** | True | True |
| 35 | `41.welcome` | 20 | **14** | **3** | **3** | True | True |
| 36 | `42.work` | 20 | **14** | **3** | **3** | True | True |
| 37 | `43.yesterday` | 20 | **14** | **3** | **3** | True | True |
| 38 | `44.teeth` | 20 | **14** | **3** | **3** | True | True |
| 39 | `45.hand` | 20 | **14** | **3** | **3** | True | True |
| 40 | `46.write` | 19 | **14** | **2** | **3** | True | True |
| 41 | `47.umberlla` | 16 | **12** | **2** | **2** | True | True |
| 42 | `48.ring` | 21 | **15** | **3** | **3** | True | True |
| 43 | `5.dog` | 18 | **12** | **3** | **3** | True | True |
| 44 | `50.power` | 18 | **12** | **3** | **3** | True | True |
| 45 | `6.fish` | 17 | **11** | **3** | **3** | True | True |
| 46 | `7.goodmorning` | 18 | **12** | **3** | **3** | True | True |
| 47 | `8.grey` | 18 | **12** | **3** | **3** | True | True |
| 48 | `9.hello` | 18 | **12** | **3** | **3** | True | True |

---

## 6. Reproducibility Evidence

- An independent in-memory reproduction of the allocation algorithm was executed using seed `42` on the audited 915 samples.
- The resulting partition assignments were compared against `phase8b5b_split_manifest.csv` and the NPZ files.
- **Result**: **100% match (915 / 915 samples identical)**. Deterministic reproducibility is confirmed beyond doubt.

---

## 7. Deliverables & Preservation Verification

### Audit Deliverables (`ai-training/data/phase8b5c_audit/`)
1. `phase8b5c_independent_audit_report.md`: Comprehensive audit report.
2. `phase8b5c_independent_audit_report.json`: Machine-readable audit results.
3. `phase8b5c_sample_integrity.csv`: 915-row audit mapping exact feature matches, SHA-256 signatures, and row indices.
4. `phase8b5c_group_isolation.csv`: 282-row audit confirming 100% partition isolation for every session burst.
5. `phase8b5c_class_distribution.csv`: 49-row recomputed class distribution confirming full coverage.

### Strict Preservation Verification
- [x] `ai-training/data/staged_phase8b5b_split/`: All partition files (`train.npz`, `val.npz`, `test.npz`) and manifests left completely unmodified.
- [x] `ai-training/data/staged_clean_features/`: Clean feature matrix left completely unmodified.
- [x] `ai-training/data/staged_clean_landmarks/`: Clean landmark archives left completely unmodified.
- [x] `D:\SignLang\sign project`: 978 raw videos left completely unmodified.
- [x] `ai-training/models/`: Baseline model checkpoints left completely unmodified.
- [x] `frontend/`: Application code and exported model left completely unmodified.
- [x] **No model training, fitting, or held-out evaluation performed in Phase 8B.5C.**

---

## 8. Final Readiness Recommendation

The Phase 8B.5B split artifacts are verified, isolated, and numerically pristine.
The project is officially certified to proceed to **Phase 8B.6: Model Training & Architecture Exploration**.