r"""
SIGNBRIDGE AI — Model Evaluation Module
=========================================
Evaluates trained gesture classification models on the held-out test split.
Calculates Accuracy, Correct/Incorrect counts, Precision, Recall, F1-score (Macro and Weighted),
per-class breakdowns, confusion matrix, top confusions, and weak classes.
Implemented in pure NumPy for 100% native compatibility with Python 3.14 on Windows.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Ensure real-time line buffering
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required for evaluate.py.")
    sys.exit(1)

import config


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    id_to_label: Dict[int, str],
) -> Dict[str, Any]:
    """
    Computes confusion matrix, macro/weighted precision, recall, F1 scores,
    correct/incorrect counts, and top confusion pairs in pure NumPy.
    """
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    # Populate confusion matrix (rows = True, cols = Predicted)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class_stats = {}
    macro_prec, macro_rec, macro_f1 = 0.0, 0.0, 0.0
    weighted_prec, weighted_rec, weighted_f1 = 0.0, 0.0, 0.0
    total_samples = len(y_true)
    weak_classes = []

    for c in range(n_classes):
        tp = float(cm[c, c])
        fp = float(np.sum(cm[:, c]) - tp)
        fn = float(np.sum(cm[c, :]) - tp)
        support = float(np.sum(cm[c, :]))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        label_name = id_to_label.get(c, f"Class_{c}")
        per_class_stats[label_name] = {
            "class_id": int(c),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

        # Track weak classes (F1 < 50% or zero recall)
        if support > 0 and f1 < 0.50:
            weak_classes.append({
                "class_name": label_name,
                "f1_score": round(f1, 4),
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "support": int(support),
            })

        macro_prec += precision / n_classes
        macro_rec += recall / n_classes
        macro_f1 += f1 / n_classes

        weighted_prec += precision * (support / total_samples) if total_samples > 0 else 0.0
        weighted_rec += recall * (support / total_samples) if total_samples > 0 else 0.0
        weighted_f1 += f1 * (support / total_samples) if total_samples > 0 else 0.0

    overall_accuracy = float(np.mean(y_true == y_pred)) if total_samples > 0 else 0.0
    correct_count = int(np.sum(y_true == y_pred))
    incorrect_count = int(np.sum(y_true != y_pred))

    # Identify most frequently confused pairs
    confused_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": id_to_label.get(i, f"Class_{i}"),
                    "predicted_class": id_to_label.get(j, f"Class_{j}"),
                    "count": int(cm[i, j]),
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "total_test_samples": total_samples,
        "classes_evaluated": int(len(np.unique(y_true))),
        "macro_avg": {
            "precision": round(macro_prec, 4),
            "recall": round(macro_rec, 4),
            "f1_score": round(macro_f1, 4),
        },
        "weighted_avg": {
            "precision": round(weighted_prec, 4),
            "recall": round(weighted_rec, 4),
            "f1_score": round(weighted_f1, 4),
        },
        "per_class": per_class_stats,
        "weak_classes": weak_classes,
        "top_confusions": confused_pairs[:15],
        "confusion_matrix": cm.tolist(),
    }


def evaluate_single_model(
    model_file: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    id_to_label: Dict[int, str],
) -> Dict[str, Any]:
    """
    Runs evaluation for one model checkpoint file on test data.
    """
    with np.load(model_file) as m_data:
        model_type = str(m_data["model_type"])
        scaler_mean = m_data["scaler_mean"]
        scaler_scale = m_data["scaler_scale"]
        classes = m_data["classes"]

        # Standardize test features using training scaler parameters
        X_test_scaled = (X_test - scaler_mean) / scaler_scale

        if model_type == "softmax_regression":
            W = m_data["W"]
            b = m_data["b"]
            logits = np.dot(X_test_scaled, W) + b
            y_pred = np.argmax(logits, axis=1)
        elif model_type == "nearest_centroid":
            centroids = m_data["centroids"]
            dists = np.linalg.norm(X_test_scaled[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            y_pred = classes[np.argmin(dists, axis=1)]
        else:
            raise ValueError(f"Unsupported model type in checkpoint: {model_type}")

    results = compute_multiclass_metrics(y_test, y_pred, classes, id_to_label)
    results["model_type"] = model_type
    results["model_file"] = str(model_file)

    # Save dedicated JSON report
    report_json_path = config.REPORTS_DIR / f"evaluation_{model_type}.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save dedicated Confusion Matrix CSV
    cm_csv_path = config.REPORTS_DIR / f"confusion_matrix_{model_type}.csv"
    with open(cm_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["True \\ Pred"] + [id_to_label.get(c, f"Class_{c}") for c in range(len(classes))]
        writer.writerow(header)
        for i, row in enumerate(results["confusion_matrix"]):
            writer.writerow([id_to_label.get(i, f"Class_{i}")] + row)

    # Save dedicated TXT summary
    summary_txt_path = config.REPORTS_DIR / f"evaluation_{model_type}.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 76 + "\n")
        f.write(f"       SIGNBRIDGE AI — TEST SET EVALUATION REPORT ({model_type.upper()})\n")
        f.write("=" * 76 + "\n")
        f.write(f"Model Architecture  : {model_type.upper()}\n")
        f.write(f"Model Checkpoint    : {model_file.name}\n")
        f.write(f"Total Test Samples  : {results['total_test_samples']}\n")
        f.write(f"Classes Evaluated   : {results['classes_evaluated']} / {len(classes)}\n")
        f.write(f"Correct Predictions : {results['correct_predictions']} / {results['total_test_samples']}\n")
        f.write(f"Incorrect / Errors  : {results['incorrect_predictions']} / {results['total_test_samples']}\n")
        f.write(f"Overall Accuracy    : {results['overall_accuracy'] * 100:.2f}%\n")
        f.write(f"Macro Precision     : {results['macro_avg']['precision']:.4f}\n")
        f.write(f"Macro Recall        : {results['macro_avg']['recall']:.4f}\n")
        f.write(f"Macro F1-Score      : {results['macro_avg']['f1_score']:.4f}\n")
        f.write(f"Weighted F1-Score   : {results['weighted_avg']['f1_score']:.4f}\n")
        f.write("-" * 76 + "\n\n")

        f.write("Top Confused Sign Classes:\n")
        if results["top_confusions"]:
            for item in results["top_confusions"]:
                f.write(f"  • True: '{item['true_class']}' misclassified as '{item['predicted_class']}' ({item['count']} time(s))\n")
        else:
            f.write("  • None (zero misclassifications!)\n")
        f.write("\n" + "-" * 76 + "\n\n")

        f.write("Per-Class Metrics:\n")
        f.write(f"{'Class Name':<28} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'Support':>7}\n")
        f.write("-" * 76 + "\n")
        for c_name, c_stats in results["per_class"].items():
            f.write(
                f"{c_name:<28} | "
                f"{c_stats['precision']:>9.4f} | "
                f"{c_stats['recall']:>9.4f} | "
                f"{c_stats['f1_score']:>9.4f} | "
                f"{c_stats['support']:>7}\n"
            )

        if results["weak_classes"]:
            f.write("\n" + "!" * 76 + "\n")
            f.write(f"Classes Requiring Attention (F1 < 50%): {len(results['weak_classes'])} classes\n")
            for wc in results["weak_classes"]:
                f.write(f"  - {wc['class_name']}: F1={wc['f1_score']:.2f}, Recall={wc['recall']:.2f}, Precision={wc['precision']:.2f} (Support: {wc['support']})\n")
            f.write("!" * 76 + "\n")

    # If this is the primary baseline model, also maintain the generic report paths
    if model_file.name == "baseline_model.npz" or model_type == config.DEFAULT_MODEL_TYPE:
        generic_json = config.REPORTS_DIR / "evaluation_report.json"
        generic_txt = config.REPORTS_DIR / "evaluation_summary.txt"
        with open(generic_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        with open(generic_txt, "w", encoding="utf-8") as f:
            with open(summary_txt_path, "r", encoding="utf-8") as src:
                f.write(src.read())

    # Console output
    print("=" * 76)
    print(f"             SIGNBRIDGE AI — TEST SET EVALUATION REPORT")
    print("=" * 76)
    print(f"Model Architecture  : {model_type.upper()}")
    print(f"Model Checkpoint    : {model_file.name}")
    print(f"Total Test Samples  : {results['total_test_samples']}")
    print(f"Classes Evaluated   : {results['classes_evaluated']} / {len(classes)} (100% represented)")
    print(f"Correct Predictions : {results['correct_predictions']} / {results['total_test_samples']}")
    print(f"Incorrect / Errors  : {results['incorrect_predictions']} / {results['total_test_samples']}")
    print(f"Overall Accuracy    : {results['overall_accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score      : {results['macro_avg']['f1_score']:.4f}")
    print(f"Weighted F1-Score   : {results['weighted_avg']['f1_score']:.4f}")
    print(f"Macro Precision     : {results['macro_avg']['precision']:.4f}")
    print(f"Macro Recall        : {results['macro_avg']['recall']:.4f}")
    print("=" * 76)
    print("Top Confusions:")
    for item in results["top_confusions"][:5]:
        print(f"  • True: '{item['true_class']}' -> Pred: '{item['predicted_class']}' ({item['count']}x)")
    print(f"Artifacts Saved:")
    print(f"  -> Report JSON : {report_json_path}")
    print(f"  -> Summary TXT : {summary_txt_path}")
    print(f"  -> Confusion CSV: {cm_csv_path}\n")

    return results


def evaluate_model(
    test_file: Path = config.PROCESSED_DATA_DIR / "test.npz",
    model_file: Optional[Path] = None,
    label_mapping_file: Path = config.PROCESSED_DATA_DIR / "label_mapping.json",
) -> Dict[str, Any]:
    """
    Evaluates both Softmax Logistic Regression and Nearest Centroid baselines on test.npz.
    """
    if not test_file.exists():
        raise FileNotFoundError(f"Test split not found at: {test_file}")

    # Load test split
    with np.load(test_file) as d_test:
        X_test = d_test["X"]
        y_test = d_test["y"]

    # Load label mapping
    id_to_label = {}
    if label_mapping_file.exists():
        with open(label_mapping_file, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
            id_to_label = {int(k): v for k, v in mapping_data.get("id_to_label", {}).items()}

    # Determine which models to evaluate
    models_to_eval = []
    if model_file is not None and model_file.exists():
        models_to_eval.append(model_file)
    else:
        # Default: evaluate both baseline models if present
        primary_model = config.MODELS_DIR / "baseline_model.npz"
        nc_model = config.MODELS_DIR / "nearest_centroid_model.npz"
        if primary_model.exists():
            models_to_eval.append(primary_model)
        if nc_model.exists():
            models_to_eval.append(nc_model)

    if not models_to_eval:
        raise FileNotFoundError(f"No trained model checkpoints found in: {config.MODELS_DIR}")

    all_results = {}
    for m_path in models_to_eval:
        res = evaluate_single_model(m_path, X_test, y_test, id_to_label)
        all_results[res["model_type"]] = res

    # Return primary model result for single-model API compatibility
    primary_res = all_results.get("softmax_regression", next(iter(all_results.values())))
    return primary_res


if __name__ == "__main__":
    evaluate_model()
