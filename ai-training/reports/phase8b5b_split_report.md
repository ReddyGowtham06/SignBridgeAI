# SIGNBRIDGE AI — PHASE 8B.5B: FRESH SESSION-BURST-GROUPED & STRATIFIED SPLITTING REPORT

## Executive Summary

Phase 8B.5B has successfully created a fresh, fully reproducible, leakage-aware train/validation/test split on the Phase 8B.4.1 validated clean dataset containing exactly **915 samples across all 49 ISL sign classes**.
The split follows the Phase 8B.5A recommendation: **Hybrid Session-Burst Grouped & Stratified Partitioning** using random seed **42**.

### Key Quantitative Metrics

| Partition | Sample Count | Sample Percentage | Session Bursts | Class Coverage | Per-Class Range |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training (`train.npz`)** | **631** | **68.96%** | **184** bursts | **49 / 49 (100%)** | 10 – 15 samples |
| **Validation (`val.npz`)** | **142** | **15.52%** | **49** bursts | **49 / 49 (100%)** | 2 – 3 samples |
| **Testing (`test.npz`)** | **142** | **15.52%** | **49** bursts | **49 / 49 (100%)** | 2 – 3 samples |
| **TOTAL CLEAN DATASET** | **915** | **100.0%** | **282** bursts | **49 / 49 (100%)** | 15 – 21 samples |

> [!IMPORTANT]
> **Scientific Generalization Boundary**: The resulting evaluation is **cross-session / take-independent** under audited grouping assumptions. It is **not certified subject-independent evaluation** because verified signer ground truth is unavailable across 62.6% of the raw dataset.

---

## 1. Group Isolation & Zero Leakage Verification

- **Zero Session-Burst Leakage**: Exactly **0 session-burst groups cross partitions**. Every consecutive take recorded within the same session burst is 100% contained within either Train, Validation, or Test.
- **Zero Sample Overlap**: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
- **Within-Class Possible Signer Overlap**: Exactly **0 within-class overlaps**. For any given sign class, samples from the same possible contributor (`Krishnaraj`, `Manideep`, or `Nachiketh`) never appear in both Train and Test.
- **Deterministic Reproducibility**: Re-running the allocation with seed 42 produces a 100% byte-for-byte identical sample assignment.

---

## 2. Inferred Contributor Distribution Across Partitions

| Inferred Contributor Label | Total Samples | Train Samples (%) | Val Samples (%) | Test Samples (%) | Within-Class Tr-Te Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Signer_Krishnaraj` | **133** | 83 (62.4%) | 18 (13.5%) | 32 (24.1%) | **0** |
| `Signer_Manideep` | **137** | 69 (50.4%) | 44 (32.1%) | 24 (17.5%) | **0** |
| `Signer_Nachiketh` | **72** | 49 (68.1%) | 11 (15.3%) | 12 (16.7%) | **0** |
| `Unknown_Signer` | **573** | 430 (75.0%) | 69 (12.0%) | 74 (12.9%) | **26** |

---

## 3. Complete 49-Class Distribution Table

Every single sign class achieves full representation across all three partitions, with strictly $\ge 2$ samples in Validation and $\ge 2$ samples in Test:

| Class ID | Class Name | Total | Train Count (%) | Val Count (%) | Test Count (%) | Val $\ge 2$? | Test $\ge 2$? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `1.bird` | 17 | 11 (64.7%) | 3 (17.6%) | 3 (17.6%) | True | True |
| 1 | `10.youareperfect` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 2 | `11.monsoon` | 21 | 15 (71.4%) | 3 (14.3%) | 3 (14.3%) | True | True |
| 3 | `12.afternoon` | 15 | 10 (66.7%) | 3 (20.0%) | 2 (13.3%) | True | True |
| 4 | `13.angry` | 19 | 13 (68.4%) | 3 (15.8%) | 3 (15.8%) | True | True |
| 5 | `14.bad` | 17 | 11 (64.7%) | 3 (17.6%) | 3 (17.6%) | True | True |
| 6 | `15.boy` | 16 | 12 (75.0%) | 2 (12.5%) | 2 (12.5%) | True | True |
| 7 | `16.eat` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 8 | `17.friend` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 9 | `18.drinking` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 10 | `19.girl` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 11 | `2.black` | 17 | 11 (64.7%) | 3 (17.6%) | 3 (17.6%) | True | True |
| 12 | `20.brother` | 17 | 11 (64.7%) | 3 (17.6%) | 3 (17.6%) | True | True |
| 13 | `21.good` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 14 | `22.father` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 15 | `23.evening` | 19 | 13 (68.4%) | 3 (15.8%) | 3 (15.8%) | True | True |
| 16 | `24.help` | 18 | 12 (66.7%) | 3 (16.7%) | 3 (16.7%) | True | True |
| 17 | `25.mother` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 18 | `26.NAME` | 16 | 12 (75.0%) | 2 (12.5%) | 2 (12.5%) | True | True |
| 19 | `27.Night` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 20 | `28.Music` | 17 | 11 (64.7%) | 3 (17.6%) | 3 (17.6%) | True | True |
| 21 | `29.nose` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 22 | `3.cat` | 16 | 12 (75.0%) | 2 (12.5%) | 2 (12.5%) | True | True |
| 23 | `30.sleep` | 17 | 11 (64.7%) | 3 (17.6%) | 3 (17.6%) | True | True |
| 24 | `31.sit` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 25 | `32.sorry` | 17 | 11 (64.7%) | 3 (17.6%) | 3 (17.6%) | True | True |
| 26 | `33.stand` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 27 | `34.stop` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 28 | `35.student` | 19 | 13 (68.4%) | 3 (15.8%) | 3 (15.8%) | True | True |
| 29 | `36.study` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 30 | `37.teacher` | 17 | 11 (64.7%) | 3 (17.6%) | 3 (17.6%) | True | True |
| 31 | `38.thankyou` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 32 | `39.today` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 33 | `4.cow` | 17 | 11 (64.7%) | 3 (17.6%) | 3 (17.6%) | True | True |
| 34 | `40.tommorow` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 35 | `41.welcome` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 36 | `42.work` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 37 | `43.yesterday` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 38 | `44.teeth` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 39 | `45.hand` | 20 | 14 (70.0%) | 3 (15.0%) | 3 (15.0%) | True | True |
| 40 | `46.write` | 19 | 14 (73.7%) | 2 (10.5%) | 3 (15.8%) | True | True |
| 41 | `47.umberlla` | 16 | 12 (75.0%) | 2 (12.5%) | 2 (12.5%) | True | True |
| 42 | `48.ring` | 21 | 15 (71.4%) | 3 (14.3%) | 3 (14.3%) | True | True |
| 43 | `5.dog` | 18 | 12 (66.7%) | 3 (16.7%) | 3 (16.7%) | True | True |
| 44 | `50.power` | 18 | 12 (66.7%) | 3 (16.7%) | 3 (16.7%) | True | True |
| 45 | `6.fish` | 17 | 11 (64.7%) | 3 (17.6%) | 3 (17.6%) | True | True |
| 46 | `7.goodmorning` | 18 | 12 (66.7%) | 3 (16.7%) | 3 (16.7%) | True | True |
| 47 | `8.grey` | 18 | 12 (66.7%) | 3 (16.7%) | 3 (16.7%) | True | True |
| 48 | `9.hello` | 18 | 12 (66.7%) | 3 (16.7%) | 3 (16.7%) | True | True |

---

## 4. Generated Artifacts & File Inventory

All Phase 8B.5B outputs are saved in `ai-training/data/staged_phase8b5b_split/`:

1. `train.npz`: Feature matrix `(631, 758)` float32, labels `(631,)` int64, sample IDs, video names, class names.
2. `val.npz`: Feature matrix `(142, 758)` float32, labels `(142,)` int64, sample IDs, video names, class names.
3. `test.npz`: Feature matrix `(142, 758)` float32, labels `(142,)` int64, sample IDs, video names, class names.
4. `phase8b5b_split_manifest.csv`: Comprehensive 915-row manifest mapping each sample to its source path, SHA-256, session burst, signer category, and partition.
5. `phase8b5b_group_assignment.csv`: 282-row manifest mapping every atomic session burst to its partition.
6. `phase8b5b_class_distribution.csv`: 49-row per-class sample counts and partition percentages.
7. `phase8b5b_possible_signer_leakage.csv`: Audit of contributor representations across partitions.
8. `phase8b5b_split_report.json`: Machine-readable audit and validation results.
9. `phase8b5b_split_report.md`: Complete human-readable report.

---

## 5. Strict Phase Boundary Compliance

- [x] Original source videos in `D:\SignLang\sign project` left completely untouched.
- [x] Baseline raw landmarks in `ai-training/data/raw_landmarks/` left completely untouched.
- [x] Baseline processed splits in `ai-training/data/processed/` left completely untouched.
- [x] Staged clean landmarks in `ai-training/data/staged_clean_landmarks/` left completely untouched.
- [x] Baseline model checkpoints in `ai-training/models/` left completely untouched.
- [x] Frontend application and inference code left completely untouched.
- [x] **No model training, fitting, or held-out evaluation performed in Phase 8B.5B.**