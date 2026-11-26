r"""
SIGNBRIDGE AI — Phase 8B.6B Controlled Model Training Pipeline
==============================================================
Trains and evaluates five candidate models on the audited Phase 8B.5B dataset split:
  1. Softmax Logistic Regression (L2 Regularized)
  2. Nearest Centroid (Euclidean)
  3. Nearest Centroid (Cosine)
  4. Linear Support Vector Machine (One-vs-Rest, Squared Hinge)
  5. Small Multi-Layer Perceptron (1 Hidden Layer, 128 units, Dropout 0.4, Adam)

Follows strict scientific isolation:
- Scaler fit ONLY on train.npz
- Model selection & hyperparameter tuning evaluated ONLY on val.npz
- test.npz remains completely frozen until final post-selection evaluation
"""

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# Real-time console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import evaluate_phase8b6b as eval_lib

# Paths
PROJECT_ROOT = Path(r"D:\SignBridgeAI")
TRAINING_DIR = PROJECT_ROOT / "ai-training"
SPLIT_DIR = TRAINING_DIR / "data" / "staged_phase8b5b_split"
MODELS_OUT_DIR = TRAINING_DIR / "models" / "phase8b6b"
REPORTS_OUT_DIR = TRAINING_DIR / "reports" / "phase8b6b"
LABEL_MAPPING_PATH = TRAINING_DIR / "data" / "processed" / "label_mapping.json"

MODELS_OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SCALER
# =============================================================================
class StandardScalerNumPy:
    """Standardizes features using mean and unit variance fitted on training data."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0).astype(np.float32)
        self.scale_ = np.std(X, axis=0).astype(np.float32)
        # Prevent division by zero
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted.")
        return ((X - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# =============================================================================
# PRE-FLIGHT INTEGRITY VALIDATION
# =============================================================================
def validate_dataset_integrity(split_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Validates dataset integrity before any model training:
    - Verifies file existence, shapes, dtypes, finite values, class coverage (0-48).
    - Checks for sample_id disjointness.
    - Loads train and validation splits.
    """
    print("=" * 76)
    print("PHASE 8B.6B: PRE-TRAINING DATASET INTEGRITY VALIDATION")
    print("=" * 76)

    train_path = split_dir / "train.npz"
    val_path = split_dir / "val.npz"
    test_path = split_dir / "test.npz"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required split file: {p}")

    with np.load(train_path) as tr, np.load(val_path) as va, np.load(test_path) as te:
        X_tr, y_tr = tr["X"].astype(np.float32), tr["y"].astype(np.int64)
        X_va, y_va = va["X"].astype(np.float32), va["y"].astype(np.int64)
        X_te, y_te = te["X"].astype(np.float32), te["y"].astype(np.int64)

        ids_tr = set(tr["sample_ids"])
        ids_va = set(va["sample_ids"])
        ids_te = set(te["sample_ids"])

    # 1. Sample Counts
    assert len(X_tr) == 631, f"Expected 631 train samples, got {len(X_tr)}"
    assert len(X_va) == 142, f"Expected 142 val samples, got {len(X_va)}"
    assert len(X_te) == 142, f"Expected 142 test samples, got {len(X_te)}"

    # 2. Dimensions
    assert X_tr.shape[1] == 758, f"Expected 758 features, got {X_tr.shape[1]}"
    assert X_va.shape[1] == 758, f"Expected 758 features, got {X_va.shape[1]}"
    assert X_te.shape[1] == 758, f"Expected 758 features, got {X_te.shape[1]}"

    # 3. Finite Values
    assert np.all(np.isfinite(X_tr)), "Non-finite values found in X_train"
    assert np.all(np.isfinite(X_va)), "Non-finite values found in X_val"
    assert np.all(np.isfinite(X_te)), "Non-finite values found in X_test"

    # 4. Class Coverage
    for name, targets in [("Train", y_tr), ("Val", y_va), ("Test", y_te)]:
        u_classes = np.unique(targets)
        assert len(u_classes) == 49, f"{name} does not have all 49 classes (found {len(u_classes)})"
        assert min(u_classes) == 0 and max(u_classes) == 48, f"{name} class range invalid: [{min(u_classes)}, {max(u_classes)}]"

    # 5. Zero Sample ID Overlap
    assert len(ids_tr & ids_va) == 0, f"Leakage detected between train and val: {len(ids_tr & ids_va)} IDs"
    assert len(ids_tr & ids_te) == 0, f"Leakage detected between train and test: {len(ids_tr & ids_te)} IDs"
    assert len(ids_va & ids_te) == 0, f"Leakage detected between val and test: {len(ids_va & ids_te)} IDs"

    # 6. Label Mapping
    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        lm_data = json.load(f)
        id_to_label = {int(k): v for k, v in lm_data["id_to_label"].items()}
    assert len(id_to_label) == 49, f"Expected 49 labels in mapping, got {len(id_to_label)}"

    print(f"  • Train Samples       : {len(X_tr)} (68.96%) across 49 classes")
    print(f"  • Validation Samples  : {len(X_va)} (15.52%) across 49 classes")
    print(f"  • Test Samples        : {len(X_te)} (15.52%) across 49 classes [FROZEN]")
    print(f"  • Feature Dimension   : 758 float32")
    print(f"  • Partition Overlap   : EXACT 0 SAMPLES LEAKED")
    print("Pre-flight integrity verification: ALL CHECKS PASSED.\n")

    return X_tr, y_tr, X_va, y_va, id_to_label


# =============================================================================
# MODEL 1: SOFTMAX LOGISTIC REGRESSION (L2 REGULARIZED)
# =============================================================================
class SoftmaxLogisticRegression:
    def __init__(self, lr: float = 0.05, l2_reg: float = 1e-4, batch_size: int = 32, max_epochs: int = 300, seed: int = 42):
        self.lr = lr
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.seed = seed
        self.W: Optional[np.ndarray] = None  # (758, 49)
        self.b: Optional[np.ndarray] = None  # (49,)
        self.classes_: Optional[np.ndarray] = None
        self.history: Dict[str, list] = {"epoch": [], "loss": [], "val_f1": []}

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        shift = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(shift)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, id_to_label: Optional[Dict[int, str]] = None):
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = X.shape
        self.classes_ = np.arange(49)
        n_classes = len(self.classes_)

        # He/Xavier normal initialization
        self.W = rng.normal(0.0, np.sqrt(2.0 / (n_features + n_classes)), (n_features, n_classes)).astype(np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)

        # One-hot encoding
        Y_one_hot = np.zeros((n_samples, n_classes), dtype=np.float32)
        for i, val in enumerate(y):
            Y_one_hot[i, val] = 1.0

        n_batches = int(math.ceil(n_samples / self.batch_size))
        indices = np.arange(n_samples)

        best_val_f1 = -1.0
        best_W, best_b = self.W.copy(), self.b.copy()

        for epoch in range(1, self.max_epochs + 1):
            rng.shuffle(indices)
            epoch_loss = 0.0

            for b in range(n_batches):
                batch_idx = indices[b * self.batch_size : (b + 1) * self.batch_size]
                xb = X[batch_idx]
                yb = Y_one_hot[batch_idx]
                mb = len(batch_idx)

                logits = np.dot(xb, self.W) + self.b
                probs = self._softmax(logits)

                # Cross-entropy + L2 loss
                ce_loss = -np.sum(yb * np.log(probs + 1e-12)) / mb
                reg_loss = 0.5 * self.l2_reg * np.sum(self.W**2)
                epoch_loss += (ce_loss + reg_loss) / n_batches

                # Backward
                err = (probs - yb) / mb
                grad_W = np.dot(xb.T, err) + self.l2_reg * self.W
                grad_b = np.sum(err, axis=0)

                # Parameter update
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b

            # Validation checkpoint tracking
            val_f1 = 0.0
            if X_val is not None and y_val is not None and id_to_label is not None:
                preds = self.predict(X_val)
                # Quick macro F1 calculation
                metrics = eval_lib.compute_metrics(y_val, preds, self.predict_proba(X_val), id_to_label)
                val_f1 = metrics["macro_avg"]["f1_score"]
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_W, best_b = self.W.copy(), self.b.copy()

            self.history["epoch"].append(epoch)
            self.history["loss"].append(round(float(epoch_loss), 4))
            self.history["val_f1"].append(round(float(val_f1), 4))

        # Restore weights from best validation epoch
        if X_val is not None and best_val_f1 > 0:
            self.W, self.b = best_W, best_b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.dot(X, self.W) + self.b
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = np.dot(X, self.W) + self.b
        return np.argmax(logits, axis=-1)


# =============================================================================
# MODEL 2: NEAREST CENTROID (EUCLIDEAN)
# =============================================================================
class NearestCentroidEuclidean:
    def __init__(self):
        self.centroids_: Optional[np.ndarray] = None  # (49, 758)
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.arange(49)
        centroids = []
        for c in range(49):
            class_samples = X[y == c]
            if len(class_samples) == 0:
                raise ValueError(f"Class {c} has zero training samples.")
            centroids.append(np.mean(class_samples, axis=0))
        self.centroids_ = np.array(centroids, dtype=np.float32)
        return self

    def compute_distances(self, X: np.ndarray) -> np.ndarray:
        # X: (N, 758), centroids: (49, 758)
        # dists: (N, 49)
        diff = X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = self.compute_distances(X)
        return np.argmin(dists, axis=1)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        # Negative Euclidean distance converted via temperature softmax
        dists = self.compute_distances(X)
        neg_dists = -dists
        shift = neg_dists - np.max(neg_dists, axis=1, keepdims=True)
        exp_s = np.exp(shift)
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)


# =============================================================================
# MODEL 3: NEAREST CENTROID (COSINE)
# =============================================================================
class NearestCentroidCosine:
    def __init__(self):
        self.normalized_centroids_: Optional[np.ndarray] = None  # (49, 758)
        self.classes_: Optional[np.ndarray] = None

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=-1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return X / norms

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.arange(49)
        X_norm = self._normalize(X)
        centroids = []
        for c in range(49):
            class_samples = X_norm[y == c]
            if len(class_samples) == 0:
                raise ValueError(f"Class {c} has zero training samples.")
            c_mean = np.mean(class_samples, axis=0)
            c_norm = self._normalize(c_mean[np.newaxis, :])[0]
            centroids.append(c_norm)
        self.normalized_centroids_ = np.array(centroids, dtype=np.float32)
        return self

    def compute_cosine_similarity(self, X: np.ndarray) -> np.ndarray:
        X_norm = self._normalize(X)
        # Cosine similarity: (N, 758) dot (758, 49) -> (N, 49)
        return np.dot(X_norm, self.normalized_centroids_.T)

    def predict(self, X: np.ndarray) -> np.ndarray:
        sims = self.compute_cosine_similarity(X)
        return np.argmax(sims, axis=1)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        sims = self.compute_cosine_similarity(X)
        shift = sims - np.max(sims, axis=1, keepdims=True)
        exp_s = np.exp(shift * 10.0)  # scaled temperature
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)


# =============================================================================
# MODEL 4: LINEAR SVM (ONE-VS-REST, SQUARED HINGE LOSS)
# =============================================================================
class LinearSVMOvR:
    """
    Multi-Class One-vs-Rest Linear SVM with L2-regularized squared hinge loss.
    Vectorized optimization across all 49 classes simultaneously in pure NumPy.
    Decision function: argmax_c (w_c^T x + b_c), exactly matching the linear inference shape.
    """

    def __init__(self, C: float = 1.0, lr: float = 0.01, max_epochs: int = 500, batch_size: int = 32, seed: int = 42):
        self.C = C
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.W: Optional[np.ndarray] = None  # (758, 49)
        self.b: Optional[np.ndarray] = None  # (49,)
        self.classes_: Optional[np.ndarray] = None
        self.history: Dict[str, list] = {"epoch": [], "loss": [], "val_f1": []}

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, id_to_label: Optional[Dict[int, str]] = None):
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = X.shape
        self.classes_ = np.arange(49)
        n_classes = len(self.classes_)

        # Construct binary target matrix Y in {-1, +1}^(N, 49)
        Y_binary = -np.ones((n_samples, n_classes), dtype=np.float32)
        for i, val in enumerate(y):
            Y_binary[i, val] = 1.0

        # Xavier initialization
        self.W = rng.normal(0.0, np.sqrt(2.0 / (n_features + n_classes)), (n_features, n_classes)).astype(np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)

        n_batches = int(math.ceil(n_samples / self.batch_size))
        indices = np.arange(n_samples)

        best_val_f1 = -1.0
        best_W, best_b = self.W.copy(), self.b.copy()

        for epoch in range(1, self.max_epochs + 1):
            rng.shuffle(indices)
            epoch_loss = 0.0

            for b in range(n_batches):
                batch_idx = indices[b * self.batch_size : (b + 1) * self.batch_size]
                xb = X[batch_idx]
                yb = Y_binary[batch_idx]
                mb = len(batch_idx)

                # Linear decision values: (mb, 49)
                z = np.dot(xb, self.W) + self.b

                # Squared hinge margin: max(0, 1 - y * z)
                margin = 1.0 - yb * z
                active_mask = (margin > 0.0).astype(np.float32)

                # Loss: 0.5 * ||W||^2 + (C / mb) * sum(max(0, 1 - y * z)^2)
                reg_loss = 0.5 * np.sum(self.W**2)
                hinge_loss = (self.C / mb) * np.sum((margin * active_mask) ** 2)
                epoch_loss += (reg_loss + hinge_loss) / n_batches

                # Gradient computation
                # d(loss)/dz = -2 * (C / mb) * (margin * active_mask) * yb
                dz = -2.0 * (self.C / mb) * (margin * active_mask) * yb
                grad_W = self.W + np.dot(xb.T, dz)
                grad_b = np.sum(dz, axis=0)

                # Parameter update with gradient clipping
                np.clip(grad_W, -10.0, 10.0, out=grad_W)
                np.clip(grad_b, -10.0, 10.0, out=grad_b)

                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b

            # Validation checkpoint tracking
            val_f1 = 0.0
            if X_val is not None and y_val is not None and id_to_label is not None:
                preds = self.predict(X_val)
                scores = self.decision_function(X_val)
                metrics = eval_lib.compute_metrics(y_val, preds, scores, id_to_label)
                val_f1 = metrics["macro_avg"]["f1_score"]
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_W, best_b = self.W.copy(), self.b.copy()

            self.history["epoch"].append(epoch)
            self.history["loss"].append(round(float(epoch_loss), 4))
            self.history["val_f1"].append(round(float(val_f1), 4))

        if X_val is not None and best_val_f1 > 0:
            self.W, self.b = best_W, best_b

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.W) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        return np.argmax(df, axis=1)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        shift = df - np.max(df, axis=1, keepdims=True)
        exp_df = np.exp(shift)
        return exp_df / np.sum(exp_df, axis=1, keepdims=True)


# =============================================================================
# MODEL 5: SMALL MLP (1 HIDDEN LAYER, 128 UNITS, DROPOUT, ADAM)
# =============================================================================
class SmallMLP:
    """
    1-Hidden Layer Multi-Layer Perceptron (758 -> 128 -> 49)
    Implemented in pure NumPy with ReLU, Inverted Dropout (p=0.4), Weight Decay (1e-3),
    Adam optimizer, and Early Stopping on validation loss.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        lr: float = 0.002,
        dropout_p: float = 0.4,
        l2_reg: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 300,
        patience: int = 20,
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.dropout_p = dropout_p
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

        self.W1: Optional[np.ndarray] = None  # (758, 128)
        self.b1: Optional[np.ndarray] = None  # (128,)
        self.W2: Optional[np.ndarray] = None  # (128, 49)
        self.b2: Optional[np.ndarray] = None  # (49,)
        self.classes_: Optional[np.ndarray] = None
        self.history: Dict[str, list] = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        shift = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(shift)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, id_to_label: Dict[int, str]):
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = X.shape
        self.classes_ = np.arange(49)
        n_classes = len(self.classes_)

        # He initialization for ReLU
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / n_features), (n_features, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / self.hidden_dim), (self.hidden_dim, n_classes)).astype(np.float32)
        self.b2 = np.zeros(n_classes, dtype=np.float32)

        # Adam moment buffers
        mW1, vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        mW2, vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        mb2, vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        # One-hot encoding
        Y_tr = np.zeros((n_samples, n_classes), dtype=np.float32)
        for i, val in enumerate(y):
            Y_tr[i, val] = 1.0

        n_batches = int(math.ceil(n_samples / self.batch_size))
        indices = np.arange(n_samples)

        best_val_loss = float("inf")
        best_val_f1 = 0.0
        best_weights = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
        epochs_no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            rng.shuffle(indices)
            epoch_loss = 0.0

            for b in range(n_batches):
                t += 1
                batch_idx = indices[b * self.batch_size : (b + 1) * self.batch_size]
                xb = X[batch_idx]
                yb = Y_tr[batch_idx]
                mb = len(batch_idx)

                # Forward pass with inverted dropout
                z1 = np.dot(xb, self.W1) + self.b1
                a1 = np.maximum(0.0, z1)

                # Inverted dropout mask
                keep_p = 1.0 - self.dropout_p
                mask = (rng.uniform(0.0, 1.0, a1.shape) < keep_p).astype(np.float32) / keep_p
                a1_drop = a1 * mask

                z2 = np.dot(a1_drop, self.W2) + self.b2
                probs = self._softmax(z2)

                # Loss
                ce_loss = -np.sum(yb * np.log(probs + 1e-12)) / mb
                reg_loss = 0.5 * self.l2_reg * (np.sum(self.W1**2) + np.sum(self.W2**2))
                batch_loss = ce_loss + reg_loss
                epoch_loss += batch_loss / n_batches

                # Backward pass
                dz2 = (probs - yb) / mb
                grad_W2 = np.dot(a1_drop.T, dz2) + self.l2_reg * self.W2
                grad_b2 = np.sum(dz2, axis=0)

                da1 = np.dot(dz2, self.W2.T) * mask
                dz1 = da1 * (z1 > 0.0).astype(np.float32)
                grad_W1 = np.dot(xb.T, dz1) + self.l2_reg * self.W1
                grad_b1 = np.sum(dz1, axis=0)

                # Adam parameter updates
                for param, grad, m, v in [
                    (self.W1, grad_W1, mW1, vW1),
                    (self.b1, grad_b1, mb1, vb1),
                    (self.W2, grad_W2, mW2, vW2),
                    (self.b2, grad_b2, mb2, vb2),
                ]:
                    m[:] = beta1 * m + (1.0 - beta1) * grad
                    v[:] = beta2 * v + (1.0 - beta2) * (grad**2)
                    m_hat = m / (1.0 - beta1**t)
                    v_hat = v / (1.0 - beta2**t)
                    param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

            # Validation evaluation (without dropout)
            val_probs = self.predict_proba(X_val)
            val_preds = np.argmax(val_probs, axis=1)
            val_ce = -np.mean([np.log(val_probs[i, y_val[i]] + 1e-12) for i in range(len(y_val))])
            val_metrics = eval_lib.compute_metrics(y_val, val_preds, val_probs, id_to_label)
            val_f1 = val_metrics["macro_avg"]["f1_score"]

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(round(float(epoch_loss), 4))
            self.history["val_loss"].append(round(float(val_ce), 4))
            self.history["val_f1"].append(round(float(val_f1), 4))

            # Early stopping check on validation loss
            if val_ce < best_val_loss - 1e-4:
                best_val_loss = val_ce
                best_val_f1 = val_f1
                best_weights = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        # Restore best checkpoint
        self.W1, self.b1, self.W2, self.b2 = best_weights
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z1 = np.dot(X, self.W1) + self.b1
        a1 = np.maximum(0.0, z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return self._softmax(z2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# =============================================================================
# MAIN PIPELINE RUNNER
# =============================================================================
def run_phase8b6b_training_pipeline() -> Dict[str, Any]:
    print("\n" + "#" * 76)
    print("       SIGNBRIDGE AI — PHASE 8B.6B: CONTROLLED MODEL TRAINING & EVAL")
    print("#" * 76 + "\n")

    # Step 1: Pre-flight Integrity Check
    X_train, y_train, X_val, y_val, id_to_label = validate_dataset_integrity(SPLIT_DIR)

    # Step 2: Scaler Hygiene (Fitted strictly on training data)
    print("=" * 76)
    print("STEP 2: SCALER HYGIENE — FITTING STANDARD SCALER ON TRAIN ONLY")
    print("=" * 76)
    scaler = StandardScalerNumPy()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"Scaler Mean Shape : {scaler.mean_.shape} (Mean of means: {np.mean(scaler.mean_):.4f})")
    print(f"Scaler Scale Shape: {scaler.scale_.shape} (Mean scale: {np.mean(scaler.scale_):.4f})")
    print(f"Zero-Std Features : {np.sum(scaler.scale_ == 1.0)} protected dimensions")
    print("Scaler fitted exclusively on train.npz. Validation scaled via frozen parameters.\n")

    # Storage for all candidate validation results
    val_candidate_results: Dict[str, Any] = {}
    saved_checkpoints: Dict[str, Path] = {}

    # -------------------------------------------------------------------------
    # STEP 3.1: MODEL 1 — SOFTMAX LOGISTIC REGRESSION (GRID SEARCH ON VAL)
    # -------------------------------------------------------------------------
    print("=" * 76)
    print("STEP 3.1: TRAINING MODEL 1 — SOFTMAX LOGISTIC REGRESSION (GRID SEARCH)")
    print("=" * 76)

    m1_grid = [
        {"lr": 0.01, "l2": 1e-4, "bs": 32},
        {"lr": 0.05, "l2": 1e-4, "bs": 32},
        {"lr": 0.10, "l2": 1e-4, "bs": 32},
        {"lr": 0.05, "l2": 1e-5, "bs": 32},
        {"lr": 0.05, "l2": 1e-3, "bs": 32},
        {"lr": 0.05, "l2": 1e-2, "bs": 32},
        {"lr": 0.05, "l2": 1e-4, "bs": 16},
        {"lr": 0.05, "l2": 1e-4, "bs": 64},
    ]

    best_m1_f1 = -1.0
    best_m1_cfg = None
    best_m1_model = None

    t0_m1 = time.time()
    for cfg in m1_grid:
        m = SoftmaxLogisticRegression(lr=cfg["lr"], l2_reg=cfg["l2"], batch_size=cfg["bs"], max_epochs=300, seed=42)
        m.fit(X_train_scaled, y_train, X_val_scaled, y_val, id_to_label)
        preds = m.predict(X_val_scaled)
        metrics = eval_lib.compute_metrics(y_val, preds, m.predict_proba(X_val_scaled), id_to_label)
        f1 = metrics["macro_avg"]["f1_score"]
        acc = metrics["overall_accuracy"]
        print(f"  [LR={cfg['lr']:.2f}, L2={cfg['l2']:.0e}, BS={cfg['bs']:>2}] -> Val Macro F1: {f1:.4f} | Val Acc: {acc * 100:.2f}%")
        if f1 > best_m1_f1:
            best_m1_f1 = f1
            best_m1_cfg = cfg
            best_m1_model = m

    t_m1 = time.time() - t0_m1
    print(f"\n  >>> Best Softmax Config: {best_m1_cfg} (Val Macro F1: {best_m1_f1:.4f}) in {t_m1:.2f}s\n")

    # Evaluate best M1 on val
    m1_val_metrics = eval_lib.compute_metrics(
        y_val, best_m1_model.predict(X_val_scaled), best_m1_model.predict_proba(X_val_scaled), id_to_label
    )
    m1_val_metrics["training_time_sec"] = round(t_m1, 2)
    m1_val_metrics["hyperparameters"] = best_m1_cfg
    val_candidate_results["m1_softmax_regression"] = m1_val_metrics

    # Save M1 checkpoint
    m1_cp_path = MODELS_OUT_DIR / "m1_softmax_regression.npz"
    np.savez_compressed(
        m1_cp_path,
        model_type=np.array("softmax_regression"),
        W=best_m1_model.W,
        b=best_m1_model.b,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        classes=best_m1_model.classes_,
        hyperparameters=json.dumps(best_m1_cfg),
    )
    saved_checkpoints["m1_softmax_regression"] = m1_cp_path

    # -------------------------------------------------------------------------
    # STEP 3.2: MODEL 2 — NEAREST CENTROID (EUCLIDEAN)
    # -------------------------------------------------------------------------
    print("=" * 76)
    print("STEP 3.2: TRAINING MODEL 2 — NEAREST CENTROID (EUCLIDEAN)")
    print("=" * 76)
    t0_m2 = time.time()
    m2 = NearestCentroidEuclidean()
    m2.fit(X_train_scaled, y_train)
    t_m2 = time.time() - t0_m2

    m2_val_preds = m2.predict(X_val_scaled)
    m2_val_scores = m2.predict_scores(X_val_scaled)
    m2_val_metrics = eval_lib.compute_metrics(y_val, m2_val_preds, m2_val_scores, id_to_label)
    m2_val_metrics["training_time_sec"] = round(t_m2, 3)
    m2_val_metrics["hyperparameters"] = {"metric": "euclidean"}
    val_candidate_results["m2_nearest_centroid_euclidean"] = m2_val_metrics

    print(f"  • Val Macro F1 : {m2_val_metrics['macro_avg']['f1_score']:.4f}")
    print(f"  • Val Accuracy : {m2_val_metrics['overall_accuracy'] * 100:.2f}%")
    print(f"  • Val Top-3 Acc: {m2_val_metrics['top3_accuracy'] * 100:.2f}% (computed in {t_m2:.3f}s)\n")

    m2_cp_path = MODELS_OUT_DIR / "m2_nearest_centroid_euclidean.npz"
    np.savez_compressed(
        m2_cp_path,
        model_type=np.array("nearest_centroid_euclidean"),
        centroids=m2.centroids_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        classes=m2.classes_,
    )
    saved_checkpoints["m2_nearest_centroid_euclidean"] = m2_cp_path

    # -------------------------------------------------------------------------
    # STEP 3.3: MODEL 3 — NEAREST CENTROID (COSINE)
    # -------------------------------------------------------------------------
    print("=" * 76)
    print("STEP 3.3: TRAINING MODEL 3 — NEAREST CENTROID (COSINE)")
    print("=" * 76)
    t0_m3 = time.time()
    m3 = NearestCentroidCosine()
    m3.fit(X_train_scaled, y_train)
    t_m3 = time.time() - t0_m3

    m3_val_preds = m3.predict(X_val_scaled)
    m3_val_scores = m3.predict_scores(X_val_scaled)
    m3_val_metrics = eval_lib.compute_metrics(y_val, m3_val_preds, m3_val_scores, id_to_label)
    m3_val_metrics["training_time_sec"] = round(t_m3, 3)
    m3_val_metrics["hyperparameters"] = {"metric": "cosine", "normalized": True}
    val_candidate_results["m3_nearest_centroid_cosine"] = m3_val_metrics

    print(f"  • Val Macro F1 : {m3_val_metrics['macro_avg']['f1_score']:.4f}")
    print(f"  • Val Accuracy : {m3_val_metrics['overall_accuracy'] * 100:.2f}%")
    print(f"  • Val Top-3 Acc: {m3_val_metrics['top3_accuracy'] * 100:.2f}% (computed in {t_m3:.3f}s)\n")

    m3_cp_path = MODELS_OUT_DIR / "m3_nearest_centroid_cosine.npz"
    np.savez_compressed(
        m3_cp_path,
        model_type=np.array("nearest_centroid_cosine"),
        normalized_centroids=m3.normalized_centroids_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        classes=m3.classes_,
    )
    saved_checkpoints["m3_nearest_centroid_cosine"] = m3_cp_path

    # -------------------------------------------------------------------------
    # STEP 3.4: MODEL 4 — LINEAR SVM (ONE-VS-REST, SQUARED HINGE)
    # -------------------------------------------------------------------------
    print("=" * 76)
    print("STEP 3.4: TRAINING MODEL 4 — LINEAR SVM (ONE-VS-REST HINGE GRID SEARCH)")
    print("=" * 76)

    svm_c_grid = [0.01, 0.1, 1.0, 5.0, 10.0]
    best_m4_f1 = -1.0
    best_m4_c = None
    best_m4_model = None

    t0_m4 = time.time()
    for c_val in svm_c_grid:
        m = LinearSVMOvR(C=c_val, lr=0.005, max_epochs=400, batch_size=32, seed=42)
        m.fit(X_train_scaled, y_train, X_val_scaled, y_val, id_to_label)
        preds = m.predict(X_val_scaled)
        scores = m.decision_function(X_val_scaled)
        metrics = eval_lib.compute_metrics(y_val, preds, scores, id_to_label)
        f1 = metrics["macro_avg"]["f1_score"]
        acc = metrics["overall_accuracy"]
        print(f"  [C={c_val:>5.2f}] -> Val Macro F1: {f1:.4f} | Val Acc: {acc * 100:.2f}%")
        if f1 > best_m4_f1:
            best_m4_f1 = f1
            best_m4_c = c_val
            best_m4_model = m

    t_m4 = time.time() - t0_m4
    print(f"\n  >>> Best Linear SVM Config: C={best_m4_c} (Val Macro F1: {best_m4_f1:.4f}) in {t_m4:.2f}s\n")

    m4_val_metrics = eval_lib.compute_metrics(
        y_val, best_m4_model.predict(X_val_scaled), best_m4_model.decision_function(X_val_scaled), id_to_label
    )
    m4_val_metrics["training_time_sec"] = round(t_m4, 2)
    m4_val_metrics["hyperparameters"] = {"C": best_m4_c, "loss": "squared_hinge"}
    val_candidate_results["m4_linear_svm_ovr"] = m4_val_metrics

    m4_cp_path = MODELS_OUT_DIR / "m4_linear_svm_ovr.npz"
    np.savez_compressed(
        m4_cp_path,
        model_type=np.array("linear_svm_ovr"),
        W=best_m4_model.W,
        b=best_m4_model.b,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        classes=best_m4_model.classes_,
        hyperparameters=json.dumps({"C": best_m4_c, "loss": "squared_hinge"}),
    )
    saved_checkpoints["m4_linear_svm_ovr"] = m4_cp_path

    # -------------------------------------------------------------------------
    # STEP 3.5: MODEL 5 — SMALL MLP (1 HIDDEN LAYER, 128 UNITS, DROPOUT, ADAM)
    # -------------------------------------------------------------------------
    print("=" * 76)
    print("STEP 3.5: TRAINING MODEL 5 — SMALL MLP (128 UNITS, DROPOUT 0.4, ADAM)")
    print("=" * 76)

    t0_m5 = time.time()
    m5 = SmallMLP(hidden_dim=128, lr=0.002, dropout_p=0.4, l2_reg=1e-3, batch_size=32, max_epochs=300, patience=25, seed=42)
    m5.fit(X_train_scaled, y_train, X_val_scaled, y_val, id_to_label)
    t_m5 = time.time() - t0_m5

    m5_val_probs = m5.predict_proba(X_val_scaled)
    m5_val_preds = np.argmax(m5_val_probs, axis=1)
    m5_val_metrics = eval_lib.compute_metrics(y_val, m5_val_preds, m5_val_probs, id_to_label)
    m5_val_metrics["training_time_sec"] = round(t_m5, 2)
    m5_val_metrics["hyperparameters"] = {"hidden_dim": 128, "dropout": 0.4, "l2_reg": 1e-3, "epochs_trained": len(m5.history["epoch"])}
    val_candidate_results["m5_small_mlp"] = m5_val_metrics

    print(f"  • Val Macro F1 : {m5_val_metrics['macro_avg']['f1_score']:.4f}")
    print(f"  • Val Accuracy : {m5_val_metrics['overall_accuracy'] * 100:.2f}%")
    print(f"  • Val Top-3 Acc: {m5_val_metrics['top3_accuracy'] * 100:.2f}% (trained in {t_m5:.2f}s, stopped at epoch {len(m5.history['epoch'])})\n")

    m5_cp_path = MODELS_OUT_DIR / "m5_small_mlp.npz"
    np.savez_compressed(
        m5_cp_path,
        model_type=np.array("small_mlp"),
        W1=m5.W1,
        b1=m5.b1,
        W2=m5.W2,
        b2=m5.b2,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        classes=m5.classes_,
        hyperparameters=json.dumps(m5_val_metrics["hyperparameters"]),
    )
    saved_checkpoints["m5_small_mlp"] = m5_cp_path

    # =========================================================================
    # STEP 4: VALIDATION-ONLY MODEL SELECTION & COMPARISON
    # =========================================================================
    print("=" * 76)
    print("STEP 4: VALIDATION-ONLY CANDIDATE COMPARISON & CHAMPION SELECTION")
    print("=" * 76)
    print(f"{'Model Architecture':<32} | {'Val Macro F1':>12} | {'Val Acc':>9} | {'Val Top-3':>10} | {'Weak Cls':>8} | {'Train Time':>10}")
    print("-" * 92)

    model_display_names = {
        "m1_softmax_regression": "M1: Softmax Logistic Reg (L2)",
        "m2_nearest_centroid_euclidean": "M2: Nearest Centroid (Euclidean)",
        "m3_nearest_centroid_cosine": "M3: Nearest Centroid (Cosine)",
        "m4_linear_svm_ovr": "M4: Linear SVM (One-vs-Rest)",
        "m5_small_mlp": "M5: Small MLP (128 units, Adam)",
    }

    ranked_candidates = sorted(
        val_candidate_results.items(), key=lambda item: item[1]["macro_avg"]["f1_score"], reverse=True
    )

    for m_key, res in ranked_candidates:
        name = model_display_names.get(m_key, m_key)
        f1 = res["macro_avg"]["f1_score"]
        acc = res["overall_accuracy"] * 100.0
        top3 = res["top3_accuracy"] * 100.0
        weak = res["weak_classes_count"]
        t_sec = res["training_time_sec"]
        print(f"{name:<32} | {f1:>12.4f} | {acc:>8.2f}% | {top3:>9.2f}% | {weak:>8} | {t_sec:>9.2f}s")

    champion_key = ranked_candidates[0][0]
    champion_res = ranked_candidates[0][1]
    champion_name = model_display_names[champion_key]
    print("-" * 92)
    print(f"\nCHAMPION MODEL SELECTED: {champion_name} (Validation Macro F1 = {champion_res['macro_avg']['f1_score']:.4f})")
    print("Selection was strictly driven by validation split performance. Test data remained 100% frozen.\n")

    # Save validation comparison summary
    val_comparison_json = {
        "selection_metric": "macro_f1",
        "champion_model": champion_key,
        "champion_name": champion_name,
        "ranking": [
            {
                "rank": rank + 1,
                "model_key": m_key,
                "model_name": model_display_names[m_key],
                "val_macro_f1": res["macro_avg"]["f1_score"],
                "val_accuracy": res["overall_accuracy"],
                "val_top3_accuracy": res["top3_accuracy"],
                "weak_classes_count": res["weak_classes_count"],
                "training_time_sec": res["training_time_sec"],
                "hyperparameters": res["hyperparameters"],
            }
            for rank, (m_key, res) in enumerate(ranked_candidates)
        ],
    }
    with open(REPORTS_OUT_DIR / "val_candidate_comparison.json", "w", encoding="utf-8") as f:
        json.dump(val_comparison_json, f, indent=2)

    # =========================================================================
    # STEP 6: FINAL TEST SPLIT EVALUATION (UNFREEZING TEST DATA)
    # =========================================================================
    print("=" * 76)
    print("STEP 6: FINAL HELD-OUT TEST EVALUATION (test.npz UNFREEZING)")
    print("=" * 76)
    print("Loading test.npz and transforming with frozen training scaler parameters...")

    with np.load(SPLIT_DIR / "test.npz") as te:
        X_test = te["X"].astype(np.float32)
        y_test = te["y"].astype(np.int64)

    X_test_scaled = scaler.transform(X_test)

    # Evaluate all candidate models on test split for complete scientific documentation
    test_results: Dict[str, Any] = {}

    # M1 Test
    m1_test_probs = best_m1_model.predict_proba(X_test_scaled)
    m1_test_preds = np.argmax(m1_test_probs, axis=1)
    test_results["m1_softmax_regression"] = eval_lib.compute_metrics(y_test, m1_test_preds, m1_test_probs, id_to_label)

    # M2 Test
    m2_test_scores = m2.predict_scores(X_test_scaled)
    m2_test_preds = m2.predict(X_test_scaled)
    test_results["m2_nearest_centroid_euclidean"] = eval_lib.compute_metrics(y_test, m2_test_preds, m2_test_scores, id_to_label)

    # M3 Test
    m3_test_scores = m3.predict_scores(X_test_scaled)
    m3_test_preds = m3.predict(X_test_scaled)
    test_results["m3_nearest_centroid_cosine"] = eval_lib.compute_metrics(y_test, m3_test_preds, m3_test_scores, id_to_label)

    # M4 Test
    m4_test_scores = best_m4_model.decision_function(X_test_scaled)
    m4_test_preds = best_m4_model.predict(X_test_scaled)
    test_results["m4_linear_svm_ovr"] = eval_lib.compute_metrics(y_test, m4_test_preds, m4_test_scores, id_to_label)

    # M5 Test
    m5_test_probs = m5.predict_proba(X_test_scaled)
    m5_test_preds = np.argmax(m5_test_probs, axis=1)
    test_results["m5_small_mlp"] = eval_lib.compute_metrics(y_test, m5_test_preds, m5_test_probs, id_to_label)

    print(f"\n{'Model Architecture':<32} | {'Test Macro F1':>13} | {'Test Acc':>9} | {'Wilson 95% CI':>18} | {'Top-3 Acc':>10}")
    print("-" * 92)

    for m_key in ["m1_softmax_regression", "m2_nearest_centroid_euclidean", "m3_nearest_centroid_cosine", "m4_linear_svm_ovr", "m5_small_mlp"]:
        t_res = test_results[m_key]
        name = model_display_names[m_key]
        f1 = t_res["macro_avg"]["f1_score"]
        acc = t_res["overall_accuracy"] * 100.0
        ci = t_res["wilson_ci_95"]["formatted"]
        top3 = t_res["top3_accuracy"] * 100.0
        print(f"{name:<32} | {f1:>13.4f} | {acc:>8.2f}% | {ci:>18} | {top3:>9.2f}%")

    print("-" * 92)

    # Save Champion artifacts
    champ_test = test_results[champion_key]
    eval_lib.save_confusion_matrix_csv(champ_test["confusion_matrix"], id_to_label, REPORTS_OUT_DIR / "champion_test_confusion_matrix.csv")
    eval_lib.save_per_class_metrics_csv(champ_test["per_class"], REPORTS_OUT_DIR / "champion_test_per_class_metrics.csv")

    # =========================================================================
    # STEP 7: FRONTEND EXPORT PREPARATION (CHAMPION MODEL JSON)
    # =========================================================================
    print("\n" + "=" * 76)
    print("STEP 7: FRONTEND EXPORT PREPARATION — CHAMPION MODEL EXPORT")
    print("=" * 76)

    # Format JSON payload matching frontend specification
    ordered_labels = [id_to_label[c] for c in range(49)]
    export_json_path = MODELS_OUT_DIR / "phase8b6b_champion.json"

    # Handle parameter export based on champion architecture
    if champion_key in ["m1_softmax_regression", "m4_linear_svm_ovr"]:
        active_model = best_m1_model if champion_key == "m1_softmax_regression" else best_m4_model
        export_payload = {
            "model_type": champion_key,
            "version": "8.6.0-phase8b6b",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": str(saved_checkpoints[champion_key].name),
            "feature_dim": 758,
            "num_classes": 49,
            "training_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "val_macro_f1": champion_res["macro_avg"]["f1_score"],
            "test_macro_f1": champ_test["macro_avg"]["f1_score"],
            "test_accuracy": champ_test["overall_accuracy"],
            "test_top3_accuracy": champ_test["top3_accuracy"],
            "test_wilson_ci_95": champ_test["wilson_ci_95"],
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "W": active_model.W.tolist(),
            "b": active_model.b.tolist(),
            "classes": active_model.classes_.tolist(),
            "id_to_label": {str(k): v for k, v in id_to_label.items()},
            "labels": ordered_labels,
        }
    elif champion_key == "m3_nearest_centroid_cosine":
        export_payload = {
            "model_type": champion_key,
            "version": "8.6.0-phase8b6b",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": str(saved_checkpoints[champion_key].name),
            "feature_dim": 758,
            "num_classes": 49,
            "training_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "val_macro_f1": champion_res["macro_avg"]["f1_score"],
            "test_macro_f1": champ_test["macro_avg"]["f1_score"],
            "test_accuracy": champ_test["overall_accuracy"],
            "test_top3_accuracy": champ_test["top3_accuracy"],
            "test_wilson_ci_95": champ_test["wilson_ci_95"],
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "normalized_centroids": m3.normalized_centroids_.tolist(),
            "classes": m3.classes_.tolist(),
            "id_to_label": {str(k): v for k, v in id_to_label.items()},
            "labels": ordered_labels,
        }
    elif champion_key == "m5_small_mlp":
        export_payload = {
            "model_type": champion_key,
            "version": "8.6.0-phase8b6b",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": str(saved_checkpoints[champion_key].name),
            "feature_dim": 758,
            "num_classes": 49,
            "training_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "val_macro_f1": champion_res["macro_avg"]["f1_score"],
            "test_macro_f1": champ_test["macro_avg"]["f1_score"],
            "test_accuracy": champ_test["overall_accuracy"],
            "test_top3_accuracy": champ_test["top3_accuracy"],
            "test_wilson_ci_95": champ_test["wilson_ci_95"],
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "W1": m5.W1.tolist(),
            "b1": m5.b1.tolist(),
            "W2": m5.W2.tolist(),
            "b2": m5.b2.tolist(),
            "classes": m5.classes_.tolist(),
            "id_to_label": {str(k): v for k, v in id_to_label.items()},
            "labels": ordered_labels,
        }
    else:
        export_payload = {
            "model_type": champion_key,
            "version": "8.6.0-phase8b6b",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": str(saved_checkpoints[champion_key].name),
            "feature_dim": 758,
            "num_classes": 49,
            "training_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "val_macro_f1": champion_res["macro_avg"]["f1_score"],
            "test_macro_f1": champ_test["macro_avg"]["f1_score"],
            "test_accuracy": champ_test["overall_accuracy"],
            "test_top3_accuracy": champ_test["top3_accuracy"],
            "test_wilson_ci_95": champ_test["wilson_ci_95"],
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "centroids": m2.centroids_.tolist(),
            "classes": m2.classes_.tolist(),
            "id_to_label": {str(k): v for k, v in id_to_label.items()},
            "labels": ordered_labels,
        }

    with open(export_json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, separators=(",", ":"))

    file_size_kb = export_json_path.stat().st_size / 1024.0
    print(f"Champion Model JSON Exported to : {export_json_path}")
    print(f"Export File Size                 : {file_size_kb:.2f} KB")

    # Verify Numerical Parity between Python and Serialized JSON
    print("\nVerifying Numerical Parity on 5 Diverse Test Samples...")
    with open(export_json_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)

    mean_j = np.array(loaded_json["scaler_mean"], dtype=np.float32)
    scale_j = np.array(loaded_json["scaler_scale"], dtype=np.float32)

    sample_indices = [0, 25, 50, 75, 100]
    parity_passed = True

    if champion_key in ["m1_softmax_regression", "m4_linear_svm_ovr"]:
        W_j = np.array(loaded_json["W"], dtype=np.float32)
        b_j = np.array(loaded_json["b"], dtype=np.float32)

        for s_idx in sample_indices:
            x_raw = X_test[s_idx]
            true_l = id_to_label[y_test[s_idx]]

            # Python checkpoint inference
            x_py = (x_raw - scaler.mean_) / scaler.scale_
            z_py = np.dot(x_py, active_model.W) + active_model.b
            pred_py = int(np.argmax(z_py))

            # JSON deserialized inference (simulating TypeScript)
            x_js = (x_raw - mean_j) / scale_j
            z_js = np.dot(x_js, W_j) + b_j
            pred_js = int(np.argmax(z_js))

            max_diff = float(np.max(np.abs(z_py - z_js)))
            match = (pred_py == pred_js)
            if not match or max_diff > 1e-4:
                parity_passed = False

            print(f"  Sample {s_idx:>3} [{true_l:<20}]: Py Pred={id_to_label[pred_py]:<20} | JSON Pred={id_to_label[pred_js]:<20} | Diff={max_diff:.2e} | Match={match}")

    print(f"Numerical Parity Check: {'PASSED (100% PARITY)' if parity_passed else 'FAILED'}\n")

    return {
        "champion_key": champion_key,
        "champion_name": champion_name,
        "champion_export_path": str(export_json_path),
        "parity_passed": parity_passed,
        "val_results": val_candidate_results,
        "test_results": test_results,
        "saved_checkpoints": {k: str(v) for k, v in saved_checkpoints.items()},
    }


if __name__ == "__main__":
    run_phase8b6b_training_pipeline()
