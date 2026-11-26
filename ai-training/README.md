# SIGNBRIDGE AI — Model Training Pipeline

Welcome to the **SIGNBRIDGE AI** training pipeline. This modular system processes raw Indian Sign Language (ISL) video recordings, extracts 3D hand skeletal landmarks using Google's **MediaPipe Hand Landmarker**, constructs feature representations, and trains baseline gesture classification models.

---

## 🏛️ Pipeline Architecture

```
ai-training/
├── config.py                 # Central configuration (paths, hyperparameters, feature dimensions)
├── dataset_scanner.py        # Stage 1: Scans class subfolders & validates video readability
├── video_processor.py        # Read-only OpenCV frame generator with millisecond timestamps
├── landmark_extractor.py     # Stage 2: MediaPipe Hand Landmarker batch extraction (resumable)
├── dataset_builder.py        # Stage 3: Feature vector engineering & label mapping
├── dataset_splitter.py       # Stage 4: Video-level stratified train / validation / test split
├── train.py                  # Stage 5: Baseline classifier training (pure NumPy, zero extra deps)
├── evaluate.py               # Stage 6: Evaluates accuracy, precision, recall, F1, and weak classes
├── run_all.py                # Master CLI orchestrator to run all stages or single steps
├── inspect_dataset.py        # Standalone dataset verification script
├── test_hand_detection.py    # Standalone single-video MediaPipe test
├── extract_sample_sequence.py# Standalone single-video sequence extractor
├── requirements.txt          # Python environment package specification
└── .gitignore                # Ignores large binary artifacts, virtualenvs, and raw landmarks
```

---

## 📦 System & Environment Requirements

- **Operating System**: Windows 10 / 11 (x64)
- **Python**: Python 3.14.0 (installed inside `ai-training/.venv`)
- **Core Dependencies**:
  - `mediapipe == 1.2.0` (MediaPipe Tasks Python API)
  - `opencv-python == 5.0.0.93` (Video container decoding)
  - `numpy == 2.5.4` (Matrix operations, feature statistics, model math)
- **Model Asset**: `frontend/public/models/hand_landmarker.task` (7.8 MB)
- **Dataset Location**: `D:\SignLang\sign project` (49 sign classes, 862 MP4 videos)

---

## 🚀 Step-by-Step Execution Guide (PowerShell)

All commands should be executed from the `ai-training` directory using the project virtual environment.

### 1. Activating the Environment (or invoking directly)

```powershell
cd D:\SignBridgeAI\ai-training

# Option A: Run directly with the virtualenv Python executable
& ".\.venv\Scripts\python.exe" run_all.py --help

# Option B: Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

---

### 2. Running Individual Pipeline Stages

You can execute each stage independently:

#### **Stage 1 — Scan and Validate Dataset**
Scans all 49 classes and 862 videos, validates OpenCV decodability, and saves a manifest to `reports/dataset_manifest.json`:
```powershell
& ".\.venv\Scripts\python.exe" dataset_scanner.py
```

#### **Stage 2 — Extract MediaPipe Hand Landmarks (Resumable)**
Processes videos through `hand_landmarker.task` in `VIDEO` mode, extracting 21 landmarks per hand:
```powershell
# Smoke test on 5 videos first:
& ".\.venv\Scripts\python.exe" landmark_extractor.py --limit 5

# Full batch extraction across all videos (automatically skips already extracted files):
& ".\.venv\Scripts\python.exe" landmark_extractor.py
```

#### **Stage 3 — Build Feature Matrices**
Aggregates the individual `.npz` files into consolidated feature vectors and creates `label_mapping.json`:
```powershell
& ".\.venv\Scripts\python.exe" dataset_builder.py
```

#### **Stage 4 — Stratified Dataset Splitting**
Partitions data into Train (70%), Validation (15%), and Test (15%) splits without video leakage:
```powershell
& ".\.venv\Scripts\python.exe" dataset_splitter.py
```

#### **Stage 5 — Train Baseline Model**
Trains a pure NumPy Softmax Logistic Regression or Nearest Centroid baseline model:
```powershell
# Softmax Logistic Regression (Default):
& ".\.venv\Scripts\python.exe" train.py

# Nearest Centroid baseline:
& ".\.venv\Scripts\python.exe" run_all.py --train --model-type nearest_centroid
```

#### **Stage 6 — Evaluate on Held-out Test Set**
Evaluates accuracy, macro/weighted F1, confusion matrix, and weak classes:
```powershell
& ".\.venv\Scripts\python.exe" evaluate.py
```

---

### 3. Running the Complete Pipeline in One Command

To run all six stages in sequence:
```powershell
& ".\.venv\Scripts\python.exe" run_all.py --all
```

---

## 📐 Data Representation & Hand Slot Conventions

To guarantee consistency across single-hand and dual-hand signs:
- **Slot 0**: Assigned to the **Left Hand**.
- **Slot 1**: Assigned to the **Right Hand**.
- **Presence Mask**: A boolean array of shape `(N, 2)` explicitly indicates whether a genuine detection occurred in that slot for frame `N`.
- **Zero Fabrication**: When a hand is absent, its presence mask is `False` and coordinate values are `0.0`. Landmarks are **never invented or interpolated**.

---

## ⚖️ Dataset Attribution & Licensing

The video recordings used for training this model are derived from the Indian Sign Language Dataset:
- **Dataset License**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Attribution**: Indian Sign Language Research Dataset.
- **Integrity**: Original video recordings in `D:\SignLang\sign project` are accessed in strictly read-only mode and are never modified, renamed, or relocated.

---

## ⚠️ Scientific Integrity & Honesty Notice

- Landmark extraction reflects **geometric hand tracking only**.
- It does **not** constitute a working sign language translator until the full classifier has been trained, evaluated, and validated against the held-out test split.
- Evaluation metrics are computed strictly from test predictions and are never fabricated or exaggerated.

