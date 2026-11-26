r"""
SIGNBRIDGE AI — Phase 8B.6B Evaluation Module
=============================================
Calculates comprehensive evaluation metrics for Indian Sign Language (ISL) models:
- Macro & Weighted Precision, Recall, F1-Score
- Top-1 & Top-3 Accuracy
- 49-Class Confusion Matrix (49x49)
- Per-Class Breakdown (Precision, Recall, F1, Support, TP, FP, FN)
- Weak Class Detection (F1 < 0.50 or Recall == 0)
- Top Confused Class Pairs
- Wilson 95% Confidence Interval for Overall Accuracy

Implemented in pure NumPy for 100% native compatibility with Python 3.14 on Windows.
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


def compute_wilson_interval(successes: int, total: int, z: float = 1.95996) -> Tuple[float, float]:
    """Computes the two-sided Wilson score confidence interval."""
    if total == 0:
        return 0.0, 0.0
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    centre = (p_hat + (z**2) / (2.0 * total)) / denom
    spread = (z * math.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * total)) / total)) / denom
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return round(float(lower), 4), round(float(upper), 4)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs_or_scores: Optional[np.ndarray],
    id_to_label: Dict[int, str],
    num_classes: int = 49,
) -> Dict[str, Any]:
    """
    Computes all standard and granular evaluation metrics.
    """
    total_samples = len(y_true)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class_stats = {}
    macro_prec, macro_rec, macro_f1 = 0.0, 0.0, 0.0
    weighted_prec, weighted_rec, weighted_f1 = 0.0, 0.0, 0.0
    weak_classes = []

    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(np.sum(cm[:, c]) - tp)
        fn = float(np.sum(cm[c, :]) - tp)
        support = float(np.sum(cm[c, :]))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

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

        if support > 0 and (f1 < 0.50 or recall == 0.0):
            weak_classes.append({
                "class_id": int(c),
                "class_name": label_name,
                "f1_score": round(f1, 4),
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "support": int(support),
            })

        macro_prec += precision / num_classes
        macro_rec += recall / num_classes
        macro_f1 += f1 / num_classes

        if total_samples > 0:
            weighted_prec += precision * (support / total_samples)
            weighted_rec += recall * (support / total_samples)
            weighted_f1 += f1 * (support / total_samples)

    overall_accuracy = float(np.mean(y_true == y_pred)) if total_samples > 0 else 0.0
    correct_count = int(np.sum(y_true == y_pred))
    incorrect_count = int(np.sum(y_true != y_pred))

    # Top-3 Accuracy
    top3_accuracy = 0.0
    if probs_or_scores is not None and probs_or_scores.ndim == 2:
        top3_indices = np.argsort(probs_or_scores, axis=1)[:, -3:]
        top3_hits = sum(1 for i in range(total_samples) if y_true[i] in top3_indices[i])
        top3_accuracy = float(top3_hits / total_samples) if total_samples > 0 else 0.0
    else:
        top3_accuracy = overall_accuracy

    # Top confused pairs
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class_id": int(i),
                    "true_class": id_to_label.get(i, f"Class_{i}"),
                    "pred_class_id": int(j),
                    "predicted_class": id_to_label.get(j, f"Class_{j}"),
                    "count": int(cm[i, j]),
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    # 95% Wilson Confidence Interval for Overall Accuracy
    ci_lower, ci_upper = compute_wilson_interval(correct_count, total_samples)

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "top3_accuracy": round(top3_accuracy, 4),
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "total_samples": total_samples,
        "classes_evaluated": int(len(np.unique(y_true))),
        "wilson_ci_95": {
            "lower": ci_lower,
            "upper": ci_upper,
            "formatted": f"[{ci_lower * 100:.1f}%, {ci_upper * 100:.1f}%]"
        },
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
        "weak_classes": weak_classes,
        "weak_classes_count": len(weak_classes),
        "top_confusions": confused_pairs[:15],
        "per_class": per_class_stats,
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix_csv(cm: List[List[int]], id_to_label: Dict[int, str], out_path: Path):
    """Saves the 49x49 confusion matrix to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_classes = len(cm)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["True \\ Pred"] + [id_to_label.get(c, f"Class_{c}") for c in range(num_classes)]
        writer.writerow(header)
        for i in range(num_classes):
            row = [id_to_label.get(i, f"Class_{i}")] + cm[i]
            writer.writerow(row)


def save_per_class_metrics_csv(per_class: Dict[str, Any], out_path: Path):
    """Saves per-class metrics to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "precision", "recall", "f1_score", "support", "tp", "fp", "fn"])
        for c_name, stats in per_class.items():
            writer.writerow([
                stats["class_id"],
                c_name,
                stats["precision"],
                stats["recall"],
                stats["f1_score"],
                stats["support"],
                stats["tp"],
                stats["fp"],
                stats["fn"],
            ])
