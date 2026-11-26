import json
import os
import csv
import numpy as np

OUT_DIR = "ai-training/reports/phase9a1"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load test data
test = np.load("ai-training/data/staged_phase8b5b_split/test.npz")
X_test = test["X"].astype(np.float32)
y_test = test["y"]
num_samples = len(y_test)
num_classes = 49

# 2. Load M1
with open("frontend/public/models/sign_classifier.json", "r", encoding="utf-8") as f:
    m1_json = json.load(f)

m1_labels = m1_json["labels"]
m1_mean = np.array(m1_json["scaler_mean"], dtype=np.float32)
m1_scale = np.array(m1_json["scaler_scale"], dtype=np.float32)
m1_W = np.array(m1_json["W"], dtype=np.float32)
m1_b = np.array(m1_json["b"], dtype=np.float32)

# M1 forward pass
X_m1_scaled = (X_test - m1_mean) / m1_scale
logits_m1 = np.dot(X_m1_scaled, m1_W) + m1_b
exp_m1 = np.exp(logits_m1 - np.max(logits_m1, axis=1, keepdims=True))
probs_m1 = exp_m1 / np.sum(exp_m1, axis=1, keepdims=True)
preds_m1 = np.argmax(probs_m1, axis=1)

# Top 3 M1
top3_m1_indices = np.argsort(-probs_m1, axis=1)[:, :3]
top3_m1_correct = [y_test[i] in top3_m1_indices[i] for i in range(num_samples)]

# 3. Load M5
m5 = np.load("ai-training/models/phase8b6b/m5_small_mlp.npz")
m5_mean = m5["scaler_mean"].astype(np.float32)
m5_scale = m5["scaler_scale"].astype(np.float32)
W1 = m5["W1"].astype(np.float32)
b1 = m5["b1"].astype(np.float32)
W2 = m5["W2"].astype(np.float32)
b2 = m5["b2"].astype(np.float32)

# M5 forward pass
X_m5_scaled = (X_test - m5_mean) / m5_scale
h1 = np.maximum(0, np.dot(X_m5_scaled, W1) + b1)
logits_m5 = np.dot(h1, W2) + b2
exp_m5 = np.exp(logits_m5 - np.max(logits_m5, axis=1, keepdims=True))
probs_m5 = exp_m5 / np.sum(exp_m5, axis=1, keepdims=True)
preds_m5 = np.argmax(probs_m5, axis=1)

# Top 3 M5
top3_m5_indices = np.argsort(-probs_m5, axis=1)[:, :3]
top3_m5_correct = [y_test[i] in top3_m5_indices[i] for i in range(num_samples)]

# 4. Overall metrics calculation
def calc_metrics(preds, probs, y_true):
    top1_acc = np.mean(preds == y_true)
    top3_idx = np.argsort(-probs, axis=1)[:, :3]
    top3_acc = np.mean([y_true[i] in top3_idx[i] for i in range(len(y_true))])

    # Per-class precision, recall, F1
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for c in range(num_classes):
        tp = np.sum((preds == c) & (y_true == c))
        fp = np.sum((preds == c) & (y_true != c))
        fn = np.sum((preds != c) & (y_true == c))
        sup = np.sum(y_true == c)
        supports.append(sup)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    weighted_f1 = np.sum(np.array(f1s) * np.array(supports)) / np.sum(supports)

    return {
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precisions": precisions,
        "recalls": recalls,
        "f1s": f1s,
        "supports": supports,
    }

m1_res = calc_metrics(preds_m1, probs_m1, y_test)
m5_res = calc_metrics(preds_m5, probs_m5, y_test)

# McNemar's paired analysis
both_correct = np.sum((preds_m1 == y_test) & (preds_m5 == y_test))
m1_only = np.sum((preds_m1 == y_test) & (preds_m5 != y_test))
m5_only = np.sum((preds_m1 != y_test) & (preds_m5 == y_test))
both_wrong = np.sum((preds_m1 != y_test) & (preds_m5 != y_test))

# Disagreements
disagreements = np.sum(preds_m1 != preds_m5)

print("=== FROZEN TEST COMPARISON (142 SAMPLES) ===")
print(f"M1 Top-1 Accuracy: {m1_res['top1_acc']*100:.2f}% ({np.sum(preds_m1 == y_test)}/142)")
print(f"M5 Top-1 Accuracy: {m5_res['top1_acc']*100:.2f}% ({np.sum(preds_m5 == y_test)}/142)")
print(f"M1 Top-3 Accuracy: {m1_res['top3_acc']*100:.2f}%")
print(f"M5 Top-3 Accuracy: {m5_res['top3_acc']*100:.2f}%")
print(f"M1 Macro F1:       {m1_res['macro_f1']:.4f}")
print(f"M5 Macro F1:       {m5_res['macro_f1']:.4f}")
print(f"M1 Weighted F1:    {m1_res['weighted_f1']:.4f}")
print(f"M5 Weighted F1:    {m5_res['weighted_f1']:.4f}")
print("--- McNemar Contingency Table ---")
print(f"  Both Correct:   {both_correct}")
print(f"  M1 Only Correct:{m1_only}")
print(f"  M5 Only Correct:{m5_only}")
print(f"  Both Wrong:     {both_wrong}")
print(f"  Disagreements:  {disagreements} / 142 ({disagreements/142*100:.1f}%)")

# 5. Output phase9a1_frozen_test_comparison.csv
cmp_csv_path = os.path.join(OUT_DIR, "phase9a1_frozen_test_comparison.csv")
with open(cmp_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "m1_champion", "m5_small_mlp", "absolute_delta", "relative_change_pct"])
    writer.writerow(["top1_accuracy_pct", f"{m1_res['top1_acc']*100:.2f}%", f"{m5_res['top1_acc']*100:.2f}%", f"{(m5_res['top1_acc'] - m1_res['top1_acc'])*100:+.2f}%", f"{(m5_res['top1_acc'] - m1_res['top1_acc'])/m1_res['top1_acc']*100:+.2f}%"])
    writer.writerow(["top3_accuracy_pct", f"{m1_res['top3_acc']*100:.2f}%", f"{m5_res['top3_acc']*100:.2f}%", f"{(m5_res['top3_acc'] - m1_res['top3_acc'])*100:+.2f}%", f"{(m5_res['top3_acc'] - m1_res['top3_acc'])/m1_res['top3_acc']*100:+.2f}%"])
    writer.writerow(["macro_precision", f"{m1_res['macro_p']:.4f}", f"{m5_res['macro_p']:.4f}", f"{m5_res['macro_p'] - m1_res['macro_p']:+.4f}", f"{(m5_res['macro_p'] - m1_res['macro_p'])/m1_res['macro_p']*100:+.2f}%"])
    writer.writerow(["macro_recall", f"{m1_res['macro_r']:.4f}", f"{m5_res['macro_r']:.4f}", f"{m5_res['macro_r'] - m1_res['macro_r']:+.4f}", f"{(m5_res['macro_r'] - m1_res['macro_r'])/m1_res['macro_r']*100:+.2f}%"])
    writer.writerow(["macro_f1", f"{m1_res['macro_f1']:.4f}", f"{m5_res['macro_f1']:.4f}", f"{m5_res['macro_f1'] - m1_res['macro_f1']:+.4f}", f"{(m5_res['macro_f1'] - m1_res['macro_f1'])/m1_res['macro_f1']*100:+.2f}%"])
    writer.writerow(["weighted_f1", f"{m1_res['weighted_f1']:.4f}", f"{m5_res['weighted_f1']:.4f}", f"{m5_res['weighted_f1'] - m1_res['weighted_f1']:+.4f}", f"{(m5_res['weighted_f1'] - m1_res['weighted_f1'])/m1_res['weighted_f1']*100:+.2f}%"])
    writer.writerow(["both_correct", str(both_correct), str(both_correct), "0", "0.0%"])
    writer.writerow(["m1_only_correct", str(m1_only), "0", f"-{m1_only}", "-100.0%"])
    writer.writerow(["m5_only_correct", "0", str(m5_only), f"+{m5_only}", "+100.0%"])
    writer.writerow(["both_incorrect", str(both_wrong), str(both_wrong), "0", "0.0%"])
    writer.writerow(["disagreements_count", str(disagreements), str(disagreements), "0", "0.0%"])
print(f"Wrote {cmp_csv_path}")

# 6. Output phase9a1_per_class_comparison.csv
pc_csv_path = os.path.join(OUT_DIR, "phase9a1_per_class_comparison.csv")
with open(pc_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "class_id", "class_label", "support",
        "m1_precision", "m5_precision", "delta_precision",
        "m1_recall", "m5_recall", "delta_recall",
        "m1_f1", "m5_f1", "delta_f1", "class_effect"
    ])
    for c in range(num_classes):
        lbl = m1_labels[c]
        sup = m1_res["supports"][c]
        p1, p5 = m1_res["precisions"][c], m5_res["precisions"][c]
        r1, r5 = m1_res["recalls"][c], m5_res["recalls"][c]
        f1, f5 = m1_res["f1s"][c], m5_res["f1s"][c]
        d_f1 = f5 - f1

        if d_f1 > 0.05:
            effect = "M5_IMPROVED"
        elif d_f1 < -0.05:
            effect = "M5_DEGRADED"
        else:
            effect = "PARITY"

        writer.writerow([
            c, lbl, sup,
            f"{p1:.4f}", f"{p5:.4f}", f"{p5-p1:+.4f}",
            f"{r1:.4f}", f"{r5:.4f}", f"{r5-r1:+.4f}",
            f"{f1:.4f}", f"{f5:.4f}", f"{d_f1:+.4f}",
            effect
        ])
print(f"Wrote {pc_csv_path}")

# 7. Output phase9a1_confusion_comparison.csv
cm_m1 = np.zeros((num_classes, num_classes), dtype=int)
cm_m5 = np.zeros((num_classes, num_classes), dtype=int)
for i in range(num_samples):
    cm_m1[y_test[i], preds_m1[i]] += 1
    cm_m5[y_test[i], preds_m5[i]] += 1

cm_diff = cm_m5 - cm_m1

cm_csv_path = os.path.join(OUT_DIR, "phase9a1_confusion_comparison.csv")
with open(cm_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    header = ["ground_truth_label"] + [f"m5_vs_m1_{lbl}" for lbl in m1_labels]
    writer.writerow(header)
    for c in range(num_classes):
        row = [m1_labels[c]] + [int(x) for x in cm_diff[c]]
        writer.writerow(row)
print(f"Wrote {cm_csv_path}")

# Targeted Diagnostic Analysis
target_classes = [8, 13, 21, 46, 36, 41, 7, 38, 27]
print("\n=== TARGETED DIAGNOSTIC CLASSES (TEST SET SAMPLES) ===")
for cid in target_classes:
    lbl = m1_labels[cid]
    idx = np.where(y_test == cid)[0]
    print(f"Class {cid} ({lbl}) [N={len(idx)}]:")
    for i in idx:
        print(f"  sample {i}: GT={lbl} | M1_pred={m1_labels[preds_m1[i]]} (conf={probs_m1[i, preds_m1[i]]:.3f}) | M5_pred={m1_labels[preds_m5[i]]} (conf={probs_m5[i, preds_m5[i]]:.3f})")
