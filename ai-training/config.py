r"""
SIGNBRIDGE AI — Central Pipeline Configuration
================================================
Defines all directory paths, feature dimensions, extraction parameters,
and model hyperparameters for the Indian Sign Language (ISL) training pipeline.
"""

from pathlib import Path

# =============================================================================
# 1. CORE DIRECTORY PATHS
# =============================================================================
PROJECT_ROOT = Path(r"D:\SignBridgeAI")
TRAINING_DIR = PROJECT_ROOT / "ai-training"

# Raw dataset location
DATASET_DIR = Path(r"D:\SignLang\sign project")

# Pretrained MediaPipe Task Model
MODEL_TASK_PATH = PROJECT_ROOT / "frontend" / "public" / "models" / "hand_landmarker.task"

# Local WASM directory (for browser consistency)
WASM_DIR = PROJECT_ROOT / "frontend" / "public" / "wasm"

# Pipeline data directories
DATA_DIR = TRAINING_DIR / "data"
RAW_LANDMARKS_DIR = DATA_DIR / "raw_landmarks"       # Per-video .npz landmark sequences
PROCESSED_DATA_DIR = DATA_DIR / "processed"          # Built feature tables & train/val/test splits

# Model weights & reports
MODELS_DIR = TRAINING_DIR / "models"
REPORTS_DIR = TRAINING_DIR / "reports"

# Ensure runtime directories exist
for directory in [DATA_DIR, RAW_LANDMARKS_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. LANDMARK EXTRACTION CONFIGURATION
# =============================================================================
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3  # (x, y, z)

# MediaPipe detection thresholds
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Hand slot conventions (strictly maintained across the entire pipeline)
SLOT_LEFT = 0   # Left hand coordinates
SLOT_RIGHT = 1  # Right hand coordinates
SLOT_MAPPING = {
    SLOT_LEFT: "Left",
    SLOT_RIGHT: "Right",
}


# =============================================================================
# 3. FEATURE ENGINEERING & DATASET BUILDING
# =============================================================================
# Maximum sequence length for temporal padding / sampling
MAX_SEQUENCE_FRAMES = 60

# Summary feature dimension:
# For each of the 2 hands:
#   21 landmarks * 3 coords = 63 coordinates
#   4 summary statistics per coordinate: [mean, std, min, max] = 252 features
#   Plus velocity statistics (mean & std of frame-to-frame deltas) = 126 features
#   Plus hand presence ratio (1 feature per hand) = 2 features
# Total feature dimension = (252 + 126) * 2 + 2 = 758 features
FEATURE_VECTOR_DIM = 758


# =============================================================================
# 4. DATASET SPLITTING RATIOS
# =============================================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


# =============================================================================
# 5. BASELINE MODEL HYPERPARAMETERS
# =============================================================================
# Model choice for Python 3.14 Windows compatibility (pure NumPy implementation):
# Options: 'nearest_centroid', 'knn', 'softmax_regression'
DEFAULT_MODEL_TYPE = "softmax_regression"

# Hyperparameters for Softmax Logistic Regression
LEARNING_RATE = 0.05
MAX_EPOCHS = 300
BATCH_SIZE = 32
L2_REGULARIZATION = 1e-4

# Hyperparameters for k-NN baseline
KNN_NEIGHBORS = 5
KNN_METRIC = "cosine"  # 'cosine' or 'euclidean'

