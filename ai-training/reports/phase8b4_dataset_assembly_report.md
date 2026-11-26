# SIGNBRIDGE AI — PHASE 8B.4: UNIFIED DATASET ASSEMBLY & FEATURE RECONSTRUCTION

## Executive Summary

Phase 8B.4 safely assembled the verified original 862 MP4 dataset and newly ingested 112 converted MOV dataset into one decontaminated, unified Indian Sign Language (ISL) dataset.
Strict duplicate exclusion and quarantine protocols were enforced without modifying any original source assets, baseline models, or frontend code.
The standard 758-dimensional tabular feature representation was reconstructed with zero data leakage, zero NaN/Inf values, and strict preservation of MediaPipe hand slot conventions (left hand: Slot 0, right hand: Slot 1).

### Key Quantitative Metrics

| Metric | Count / Value | Status |
| :--- | :--- | :--- |
| Original MP4 Candidates | **862** | Verified |
| Staged Converted MOV Candidates | **112** | Verified |
| **Total Initial Candidates** | **974** | 100% Accounted |
| Quarantined Inter-Class Conflict Samples | **8** (4 pairs) | Quarantined |
| Excluded Intra-Class Duplicate Copies | **51** | Excluded |
| Retained Canonical Duplicate Samples | **51** | Retained |
| **Clean Retained Training Samples** | **915** | **100% Decontaminated** |
| Unique SHA-256 Signatures in Clean Set | **915** / 915 | **0 Duplicates** |
| Recognized Sign Classes | **49** (All 49) | Full Coverage |
| Feature Matrix Shape | **(915, 758)** | Matches Pipeline |
| Feature Dtype | **float32** | Standard |
| NaN / Inf Feature Count | **0 / 0** | Numerically Stable |

---

## 1. Inventory & Reconciliation Math

The complete raw project repository contains 978 video recordings (`D:\SignLang\sign project`).

```
Raw Project Video Universe (978 Files)
├── 862 Original MP4 Videos
└── 116 Original MOV Videos
    ├── 4 Duplicate Copies Pruned in Phase 8B.2 (from 3 internal MOV duplicate groups)
    └── 112 Unique Converted MP4 Videos Staged
Total Candidates Ingested into Phase 8B.4 = 862 + 112 = 974 Candidates

Decontamination Filtering (974 Candidates):
├── 8 Quarantined Inter-Class Conflict Samples (4 groups: 13.angry vs 14.bad, 15.boy vs 20.brother)
├── 51 Excluded Intra-Class Duplicate Copies (from 51 duplicate groups among MP4s)
└── 915 Clean Unique Retained Samples (All 49 classes represented)

Complete Project Reconciliation:
978 Raw Videos = 8 Quarantined + 55 Intra-Class Duplicates (4 MOV + 51 MP4) + 915 Clean Retained
```

---

## 2. Duplicate Handling & Conflict Quarantine Manifests

### A. Inter-Class Label Conflicts (8 Samples Quarantined)
The following 8 recordings share identical SHA-256 hashes across differing class labels, representing contradictory ground-truth sign annotations. They are quarantined and strictly excluded from training:

1. **Group INTER_DUP_01** (`26f88ea8e412...`):
   - `13.angry/krishnarajangry3.mp4` (Historical Split: `val`)
   - `14.bad/krishnarajbad1.mp4` (Historical Split: `train`)
2. **Group INTER_DUP_02** (`ecf216b387a1...`):
   - `15.boy/manideepboy-1.mp4` (Historical Split: `train`)
   - `20.brother/manideepbrother-2.mp4` (Historical Split: `train`)
3. **Group INTER_DUP_03** (`470a5b350bee...`):
   - `15.boy/manideepboy-2.mp4` (Historical Split: `test` - historical leakage)
   - `20.brother/manideepbrother-3.mp4` (Historical Split: `train`)
4. **Group INTER_DUP_04** (`1a3c0242eee8...`):
   - `15.boy/manideepboy-3.mp4` (Historical Split: `train`)
   - `20.brother/manideepbrother-1.mp4` (Historical Split: `test` - historical leakage)

### B. Intra-Class Deduplication (51 Copies Excluded)
- **51 duplicate groups** were detected among the MP4 candidate sources.
- For each group, exactly one canonical sample was selected deterministically (prioritizing training split retention and alphabetical stability).
- Exactly 51 duplicate copies were excluded and logged in `phase8b4_duplicate_exclusion_manifest.csv`.

---

## 3. Class Distribution of Clean Retained Dataset (915 Samples)

Every single sign class from 1 to 50 (49 classes total) has at least 15 clean, unique samples:

| Class ID | Class Name | Retained Samples | Min / Max Bound |
| :--- | :--- | :--- | :--- |
| 1 | `1.bird` | **17** | Balanced (15-21) |
| 10 | `10.youareperfect` | **20** | Balanced (15-21) |
| 11 | `11.monsoon` | **21** | Balanced (15-21) |
| 12 | `12.afternoon` | **15** | Balanced (15-21) |
| 13 | `13.angry` | **19** | Balanced (15-21) |
| 14 | `14.bad` | **17** | Balanced (15-21) |
| 15 | `15.boy` | **16** | Balanced (15-21) |
| 16 | `16.eat` | **20** | Balanced (15-21) |
| 17 | `17.friend` | **20** | Balanced (15-21) |
| 18 | `18.drinking` | **20** | Balanced (15-21) |
| 19 | `19.girl` | **20** | Balanced (15-21) |
| 2 | `2.black` | **17** | Balanced (15-21) |
| 20 | `20.brother` | **17** | Balanced (15-21) |
| 21 | `21.good` | **20** | Balanced (15-21) |
| 22 | `22.father` | **20** | Balanced (15-21) |
| 23 | `23.evening` | **19** | Balanced (15-21) |
| 24 | `24.help` | **18** | Balanced (15-21) |
| 25 | `25.mother` | **20** | Balanced (15-21) |
| 26 | `26.NAME` | **16** | Balanced (15-21) |
| 27 | `27.Night` | **20** | Balanced (15-21) |
| 28 | `28.Music` | **17** | Balanced (15-21) |
| 29 | `29.nose` | **20** | Balanced (15-21) |
| 3 | `3.cat` | **16** | Balanced (15-21) |
| 30 | `30.sleep` | **17** | Balanced (15-21) |
| 31 | `31.sit` | **20** | Balanced (15-21) |
| 32 | `32.sorry` | **17** | Balanced (15-21) |
| 33 | `33.stand` | **20** | Balanced (15-21) |
| 34 | `34.stop` | **20** | Balanced (15-21) |
| 35 | `35.student` | **19** | Balanced (15-21) |
| 36 | `36.study` | **20** | Balanced (15-21) |
| 37 | `37.teacher` | **17** | Balanced (15-21) |
| 38 | `38.thankyou` | **20** | Balanced (15-21) |
| 39 | `39.today` | **20** | Balanced (15-21) |
| 4 | `4.cow` | **17** | Balanced (15-21) |
| 40 | `40.tommorow` | **20** | Balanced (15-21) |
| 41 | `41.welcome` | **20** | Balanced (15-21) |
| 42 | `42.work` | **20** | Balanced (15-21) |
| 43 | `43.yesterday` | **20** | Balanced (15-21) |
| 44 | `44.teeth` | **20** | Balanced (15-21) |
| 45 | `45.hand` | **20** | Balanced (15-21) |
| 46 | `46.write` | **19** | Balanced (15-21) |
| 47 | `47.umberlla` | **16** | Balanced (15-21) |
| 48 | `48.ring` | **21** | Balanced (15-21) |
| 5 | `5.dog` | **18** | Balanced (15-21) |
| 50 | `50.power` | **18** | Balanced (15-21) |
| 6 | `6.fish` | **17** | Balanced (15-21) |
| 7 | `7.goodmorning` | **18** | Balanced (15-21) |
| 8 | `8.grey` | **18** | Balanced (15-21) |
| 9 | `9.hello` | **18** | Balanced (15-21) |

**Special Class Highlight**: `17.friend` retains **20 clean samples** (zero duplicates, fully verified landmarks).

---

## 4. Landmark Validation & Feature Reconstruction

### A. Landmark Dataset Integrity (`ai-training/data/staged_clean_landmarks/`)
- All 915 staged archives conform strictly to shape `(N, 2, 21, 3)`.
- Coordinates: 100% finite (0 NaNs, 0 Infs).
- Presence masks: valid boolean arrays `(N, 2)`.
- Timestamps: strictly increasing monotonic integer millisecond sequences (`diff > 0`).

### B. Tabular Feature Representation (`ai-training/data/staged_clean_features/`)
- Feature Builder: Reused verified `compute_video_feature_vector` from `dataset_builder.py`.
- Feature Dimension: Exactly **758** float32 values ($379$ per hand slot $\times 2$ slots).
- Slot 0: Left Hand statistics (presence ratio, active coordinate mean, std, min, max, temporal velocity mean, velocity std).
- Slot 1: Right Hand statistics (same structure).
- **No Horizontal Hand Sorting**: Strict left/right slot mapping retained as directed. No spatial swapping based on x-coordinate.
- Feature Array: `(915, 758)` float32, 0 NaNs, 0 Infs.

---

## 5. Deliverables & Preservation Verification

### Generated Reports & Manifests
1. `ai-training/reports/phase8b4_dataset_assembly_report.md`
2. `ai-training/reports/phase8b4_dataset_assembly_report.json`
3. `ai-training/reports/phase8b4_unified_source_manifest.csv` (974 rows)
4. `ai-training/reports/phase8b4_duplicate_exclusion_manifest.csv` (51 rows)
5. `ai-training/reports/phase8b4_quarantine_manifest.csv` (8 rows)
6. `ai-training/reports/phase8b4_feature_source_mapping.csv` (915 rows)

### Staged Datasets
- Clean Landmarks: `ai-training/data/staged_clean_landmarks/` (915 `.npz` files in 49 class subdirectories)
- Clean Feature Matrix: `ai-training/data/staged_clean_features/features_tabular.npz` (`X`, `y`, `sample_ids`, `video_names`, `class_names`)
- Clean Label Mapping: `ai-training/data/staged_clean_features/label_mapping.json` (49 classes)

### Strict Phase Boundary Compliance
- [x] Original source videos in `D:\SignLang\sign project` left completely untouched.
- [x] Original landmark archives in `ai-training/data/raw_landmarks/` left completely untouched.
- [x] Historical processed splits in `ai-training/data/processed/` left completely untouched.
- [x] Baseline trained models in `ai-training/models/` left completely untouched.
- [x] Frontend application and inference code left completely untouched.
- [x] **No model training, splitting, or evaluation performed in Phase 8B.4.**