r"""
SIGNBRIDGE AI — Baseline Model Training Module
================================================
Trains a baseline classifier for Indian Sign Language gesture recognition.
Engineered with pure NumPy for 100% native compatibility with Python 3.14 on Windows,
without requiring heavy external ML frameworks (PyTorch/TensorFlow).
Supports Softmax Logistic Regression, k-Nearest Neighbors, and Nearest Centroid baselines.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required for train.py.")
    sys.exit(1)

import config


class StandardScalerNumPy:
    """Standardizes features by removing the mean and scaling to unit variance."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero for invariant features
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class SoftmaxRegressionNumPy:
    """
    Multi-Class Softmax Logistic Regression with L2 Regularization
    Implemented in pure NumPy with cross-entropy loss and mini-batch gradient descent.
    """

    def __init__(
        self,
        learning_rate: float = config.LEARNING_RATE,
        max_epochs: int = config.MAX_EPOCHS,
        batch_size: int = config.BATCH_SIZE,
        l2_reg: float = config.L2_REGULARIZATION,
        random_seed: int = config.RANDOM_SEED,
    ):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.seed = random_seed
        self.W: Optional[np.ndarray] = None  # (D, C)
        self.b: Optional[np.ndarray] = None  # (C,)
        self.classes_: Optional[np.ndarray] = None
        self.training_history: Dict[str, list] = {"loss": [], "val_acc": []}

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax computation."""
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = X_train.shape
        self.classes_ = np.unique(y_train)
        n_classes = len(self.classes_)

        # Xavier/He normal initialization
        self.W = rng.normal(0, np.sqrt(2.0 / (n_features + n_classes)), (n_features, n_classes)).astype(np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)

        # One-hot encode targets
        Y_one_hot = np.zeros((n_samples, n_classes), dtype=np.float32)
        for i, target in enumerate(y_train):
            Y_one_hot[i, target] = 1.0

        for epoch in range(1, self.max_epochs + 1):
            # Shuffle mini-batches
            indices = np.arange(n_samples)
            rng.shuffle(indices)

            epoch_loss = 0.0
            n_batches = int(np.ceil(n_samples / self.batch_size))

            for b in range(n_batches):
                batch_idx = indices[b * self.batch_size : (b + 1) * self.batch_size]
                xb = X_train[batch_idx]
                yb = Y_one_hot[batch_idx]
                mb_size = len(batch_idx)

                # Forward pass
                logits = np.dot(xb, self.W) + self.b
                probs = self._softmax(logits)

                # Cross-entropy loss + L2 regularization
                batch_loss = -np.sum(yb * np.log(probs + 1e-12)) / mb_size
                reg_loss = 0.5 * self.l2_reg * np.sum(self.W**2)
                epoch_loss += (batch_loss + reg_loss) / n_batches

                # Backward pass
                error = (probs - yb) / mb_size
                grad_W = np.dot(xb.T, error) + self.l2_reg * self.W
                grad_b = np.sum(error, axis=0)

                # Parameter updates
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b

            self.training_history["loss"].append(round(float(epoch_loss), 4))

            # Monitor validation accuracy
            val_acc = 0.0
            if X_val is not None and y_val is not None:
                val_preds = self.predict(X_val)
                val_acc = float(np.mean(val_preds == y_val))
                self.training_history["val_acc"].append(round(val_acc, 4))

            if verbose and (epoch % 50 == 0 or epoch == self.max_epochs):
                val_str = f" | Val Acc: {val_acc * 100:.2f}%" if (X_val is not None) else ""
                print(f"  Epoch [{epoch:>3}/{self.max_epochs}] - Loss: {epoch_loss:.4f}{val_str}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.W is None or self.b is None:
            raise ValueError("Model has not been trained yet.")
        logits = np.dot(X, self.W) + self.b
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class NearestCentroidNumPy:
    """Nearest Centroid classifier for fast, zero-hyperparameter geometric baselining."""

    def __init__(self):
        self.centroids_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        centroids = []
        for c in self.classes_:
            class_samples = X[y == c]
            centroids.append(np.mean(class_samples, axis=0))
        self.centroids_ = np.array(centroids, dtype=np.float32)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise ValueError("Model is not fitted.")
        # Compute Euclidean distance to all centroids
        # X: (N, D), centroids: (C, D)
        dists = np.linalg.norm(X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :], axis=2)
        return self.classes_[np.argmin(dists, axis=1)]


def train_model(
    train_file: Path = config.PROCESSED_DATA_DIR / "train.npz",
    val_file: Path = config.PROCESSED_DATA_DIR / "val.npz",
    output_model_path: Path = config.MODELS_DIR / "baseline_model.npz",
    model_type: str = config.DEFAULT_MODEL_TYPE,
) -> Dict[str, Any]:
    """
    Loads training and validation sets, trains the baseline model, and saves model parameters.
    """
    if not train_file.exists():
        raise FileNotFoundError(f"Training split file not found: {train_file}")

    with np.load(train_file) as d_train:
        X_train = d_train["X"]
        y_train = d_train["y"]

    X_val, y_val = None, None
    if val_file.exists():
        with np.load(val_file) as d_val:
            X_val = d_val["X"]
            y_val = d_val["y"]

    print("=" * 72)
    print("               SIGNBRIDGE AI — MODEL TRAINING")
    print("=" * 72)
    print(f"Model Architecture : {model_type.upper()}")
    print(f"Training Samples   : {X_train.shape[0]} samples, {X_train.shape[1]} features")
    if X_val is not None:
        print(f"Validation Samples : {X_val.shape[0]} samples")
    print(f"Classes Count      : {len(np.unique(y_train))}")
    print("=" * 72 + "\n")

    # Feature Standardization (fitted strictly on train set)
    scaler = StandardScalerNumPy()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None

    start_time = time.time()

    if model_type == "softmax_regression":
        model = SoftmaxRegressionNumPy(
            learning_rate=config.LEARNING_RATE,
            max_epochs=config.MAX_EPOCHS,
            batch_size=config.BATCH_SIZE,
            l2_reg=config.L2_REGULARIZATION,
        )
        model.fit(X_train_scaled, y_train, X_val_scaled, y_val, verbose=True)
        train_acc = float(np.mean(model.predict(X_train_scaled) == y_train))
        val_acc = float(np.mean(model.predict(X_val_scaled) == y_val)) if X_val_scaled is not None else 0.0

        # Save weights
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_model_path,
            model_type=np.array("softmax_regression"),
            W=model.W,
            b=model.b,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            classes=model.classes_,
        )
        if output_model_path.name == "baseline_model.npz":
            np.savez_compressed(
                config.MODELS_DIR / "softmax_regression_model.npz",
                model_type=np.array("softmax_regression"),
                W=model.W,
                b=model.b,
                scaler_mean=scaler.mean_,
                scaler_scale=scaler.scale_,
                classes=model.classes_,
            )
    elif model_type == "nearest_centroid":
        model = NearestCentroidNumPy()
        model.fit(X_train_scaled, y_train)
        train_acc = float(np.mean(model.predict(X_train_scaled) == y_train))
        val_acc = float(np.mean(model.predict(X_val_scaled) == y_val)) if X_val_scaled is not None else 0.0

        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        # Save dedicated model-specific file
        nc_save_path = config.MODELS_DIR / "nearest_centroid_model.npz"
        np.savez_compressed(
            nc_save_path,
            model_type=np.array("nearest_centroid"),
            centroids=model.centroids_,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            classes=model.classes_,
        )
        if output_model_path.name != "baseline_model.npz":
            np.savez_compressed(
                output_model_path,
                model_type=np.array("nearest_centroid"),
                centroids=model.centroids_,
                scaler_mean=scaler.mean_,
                scaler_scale=scaler.scale_,
                classes=model.classes_,
            )
        else:
            output_model_path = nc_save_path
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    elapsed = time.time() - start_time

    # Save metadata JSON
    metadata_path = config.MODELS_DIR / "model_metadata.json"
    type_metadata_path = config.MODELS_DIR / f"{model_type}_metadata.json"
    metadata = {
        "model_type": model_type,
        "train_samples": int(X_train.shape[0]),
        "feature_dim": int(X_train.shape[1]),
        "classes_count": int(len(np.unique(y_train))),
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "training_time_sec": round(elapsed, 2),
        "weights_file": str(output_model_path),
    }
    with open(type_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Only set generic model_metadata.json if this is default model or none exists
    if model_type == config.DEFAULT_MODEL_TYPE or not metadata_path.exists():
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    print("\n" + "=" * 72)
    print("Training Completed Successfully")
    print(f"  • Final Train Accuracy : {train_acc * 100:.2f}%")
    print(f"  • Final Val Accuracy   : {val_acc * 100:.2f}%")
    print(f"  • Training Elapsed     : {elapsed:.2f} seconds")
    print(f"  • Model Saved to       : {output_model_path}")
    print(f"  • Metadata Saved to    : {metadata_path}")
    print("=" * 72)

    return metadata


if __name__ == "__main__":
    train_model()

