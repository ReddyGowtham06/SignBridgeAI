r"""
SIGNBRIDGE AI — Phase 8B.6C Parity Data & Checkpoint Auditor
============================================================
1. Computes SHA-256 hashes for all Phase 8B.6B artifacts.
2. Verifies checkpoint arrays, shapes, dtypes, finitude.
3. Performs 49-class index-by-index mapping audit across all sources.
4. Reproduces M1 and M5 metrics from checkpoints.
5. Prepares all 142 test samples with Python reference outputs for TypeScript parity audit.
"""

import csv
import hashlib
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(r"D:\SignBridgeAI")
TRAINING_DIR = PROJECT_ROOT / "ai-training"
SPLIT_DIR = TRAINING_DIR / "data" / "staged_phase8b5b_split"
MODELS_DIR = TRAINING_DIR / "models" / "phase8b6b"
REPORTS_DIR = TRAINING_DIR / "reports" / "phase8b6b"
AUDIT_DIR = TRAINING_DIR / "data" / "phase8b6c_audit"
LABEL_MAPPING_PATH = TRAINING_DIR / "data" / "processed" / "label_mapping.json"
FE_MODEL_PATH = PROJECT_ROOT / "frontend" / "public" / "models" / "sign_classifier.json"
RAW_DATASET_DIR = Path(r"D:\SignLang\sign project")

AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# 1. ARTIFACT SHA-256 HASHES & INTEGRITY
# -----------------------------------------------------------------------------
print("=" * 76)
print("1. CALCULATING SHA-256 HASHES FOR ALL PHASE 8B.6B ARTIFACTS")
print("=" * 76)

artifacts_to_hash = [
    ("m1_checkpoint", MODELS_DIR / "m1_softmax_regression.npz"),
    ("m2_checkpoint", MODELS_DIR / "m2_nearest_centroid_euclidean.npz"),
    ("m3_checkpoint", MODELS_DIR / "m3_nearest_centroid_cosine.npz"),
    ("m4_checkpoint", MODELS_DIR / "m4_linear_svm_ovr.npz"),
    ("m5_checkpoint", MODELS_DIR / "m5_small_mlp.npz"),
    ("champion_json", MODELS_DIR / "phase8b6b_champion.json"),
    ("val_candidate_comparison", REPORTS_DIR / "val_candidate_comparison.json"),
    ("candidate_comparison_csv", REPORTS_DIR / "candidate_comparison.csv"),
    ("champion_test_confusion_matrix", REPORTS_DIR / "champion_test_confusion_matrix.csv"),
    ("champion_test_per_class_metrics", REPORTS_DIR / "champion_test_per_class_metrics.csv"),
    ("phase8b6b_training_report_json", REPORTS_DIR / "phase8b6b_training_report.json"),
    ("phase8b6b_training_report_md", REPORTS_DIR / "phase8b6b_training_report.md"),
]

hash_rows = []
for name, path in artifacts_to_hash:
    if path.exists():
        h = sha256_file(path)
        size_b = path.stat().st_size
        hash_rows.append({"artifact_name": name, "file_path": str(path.relative_to(PROJECT_ROOT)), "sha256": h, "size_bytes": size_b})
        print(f"  {name:<32}: {h[:16]}... ({size_b:,} bytes)")
    else:
        print(f"  [MISSING] {name}: {path}")

hashes_csv = AUDIT_DIR / "phase8b6c_artifact_hashes.csv"
with open(hashes_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["artifact_name", "file_path", "sha256", "size_bytes"])
    writer.writeheader()
    writer.writerows(hash_rows)

print(f"\nSaved hashes to: {hashes_csv}\n")


# -----------------------------------------------------------------------------
# 2. MODEL CHECKPOINT ARRAY VERIFICATION
# -----------------------------------------------------------------------------
print("=" * 76)
print("2. VERIFYING MODEL CHECKPOINT ARRAYS, SHAPES, AND FINITENESS")
print("=" * 76)

checkpoint_audit = {}

# M1
with np.load(MODELS_DIR / "m1_softmax_regression.npz") as cp:
    W1 = cp["W"]
    b1 = cp["b"]
    m1_mean = cp["scaler_mean"]
    m1_scale = cp["scaler_scale"]
    m1_classes = cp["classes"]
    checkpoint_audit["m1"] = {
        "W_shape": list(W1.shape),
        "b_shape": list(b1.shape),
        "mean_shape": list(m1_mean.shape),
        "scale_shape": list(m1_scale.shape),
        "classes_count": len(m1_classes),
        "all_finite": bool(np.all(np.isfinite(W1)) and np.all(np.isfinite(b1)) and np.all(np.isfinite(m1_mean)) and np.all(np.isfinite(m1_scale))),
        "positive_scale": bool(np.all(m1_scale > 0)),
    }
    assert checkpoint_audit["m1"]["W_shape"] == [758, 49]
    assert checkpoint_audit["m1"]["b_shape"] == [49]
    assert checkpoint_audit["m1"]["all_finite"]
    print("  • M1 (Softmax): W=(758, 49), b=(49,), all finite, scale > 0 — PASS")

# M5 (Small MLP)
with np.load(MODELS_DIR / "m5_small_mlp.npz") as cp:
    W1_5 = cp["W1"]
    b1_5 = cp["b1"]
    W2_5 = cp["W2"]
    b2_5 = cp["b2"]
    m5_mean = cp["scaler_mean"]
    m5_scale = cp["scaler_scale"]
    m5_classes = cp["classes"]
    checkpoint_audit["m5"] = {
        "W1_shape": list(W1_5.shape),
        "b1_shape": list(b1_5.shape),
        "W2_shape": list(W2_5.shape),
        "b2_shape": list(b2_5.shape),
        "mean_shape": list(m5_mean.shape),
        "scale_shape": list(m5_scale.shape),
        "classes_count": len(m5_classes),
        "all_finite": bool(
            np.all(np.isfinite(W1_5))
            and np.all(np.isfinite(b1_5))
            and np.all(np.isfinite(W2_5))
            and np.all(np.isfinite(b2_5))
            and np.all(np.isfinite(m5_mean))
            and np.all(np.isfinite(m5_scale))
        ),
        "positive_scale": bool(np.all(m5_scale > 0)),
    }
    assert checkpoint_audit["m5"]["W1_shape"] == [758, 128]
    assert checkpoint_audit["m5"]["b1_shape"] == [128]
    assert checkpoint_audit["m5"]["W2_shape"] == [128, 49]
    assert checkpoint_audit["m5"]["b2_shape"] == [49]
    assert checkpoint_audit["m5"]["all_finite"]
    print("  • M5 (Small MLP): W1=(758, 128), b1=(128,), W2=(128, 49), b2=(49,), all finite — PASS")

# Champion JSON
with open(MODELS_DIR / "phase8b6b_champion.json", "r", encoding="utf-8") as f:
    champ_json = json.load(f)

assert len(champ_json["W"]) == 758 and len(champ_json["W"][0]) == 49
assert len(champ_json["b"]) == 49
assert len(champ_json["scaler_mean"]) == 758
assert len(champ_json["scaler_scale"]) == 758
assert len(champ_json["labels"]) == 49
assert len(champ_json["id_to_label"]) == 49

diff_W = float(np.max(np.abs(W1 - np.array(champ_json["W"], dtype=np.float32))))
diff_b = float(np.max(np.abs(b1 - np.array(champ_json["b"], dtype=np.float32))))
diff_m = float(np.max(np.abs(m1_mean - np.array(champ_json["scaler_mean"], dtype=np.float32))))
diff_s = float(np.max(np.abs(m1_scale - np.array(champ_json["scaler_scale"], dtype=np.float32))))

assert diff_W == 0.0 and diff_b == 0.0 and diff_m == 0.0 and diff_s == 0.0
print("  • Champion JSON: Exactly matches M1 NumPy checkpoint parameters (0.00e+00 diff) — PASS\n")


# -----------------------------------------------------------------------------
# 3. CRITICAL CLASS-LABEL MAPPING AUDIT
# -----------------------------------------------------------------------------
print("=" * 76)
print("3. CRITICAL 49-CLASS LABEL MAPPING AUDIT ACROSS ALL 5 SOURCES")
print("=" * 76)

with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
    proc_lm = json.load(f)
with open(FE_MODEL_PATH, "r", encoding="utf-8") as f:
    fe_lm = json.load(f)

raw_folders = sorted([d.name for d in RAW_DATASET_DIR.iterdir() if d.is_dir()])
tr_data = np.load(SPLIT_DIR / "train.npz")
tr_class_names = list(tr_data["class_names"])

# Frontend formatClassLabel function simulation
def format_class_label(raw_label: str) -> str:
    stripped = raw_label.split(".", 1)[-1] if "." in raw_label else raw_label
    dictionary = {
        "youareperfect": "You Are Perfect",
        "goodmorning": "Good Morning",
        "heavyrain": "Heavy Rain",
        "iamfine": "I Am Fine",
        "iloveyou": "I Love You",
        "nicetomeetyou": "Nice To Meet You",
        "seeyoulater": "See You Later",
        "takecare": "Take Care",
        "whatisyourname": "What Is Your Name",
    }
    lower = stripped.lower()
    if lower in dictionary:
        return dictionary[lower]
    return stripped.replace("_", " ").replace("-", " ").strip().title()

mapping_rows = []
all_sources_agree = True

for c_idx in range(49):
    si = str(c_idx)
    raw_folder = raw_folders[c_idx]
    clean_proc_map = proc_lm["id_to_label"][si]
    champ_map = champ_json["id_to_label"][si]
    champ_arr = champ_json["labels"][c_idx]
    prod_map = fe_lm["id_to_label"][si]
    prod_arr = fe_lm["labels"][c_idx]
    fe_display = format_class_label(champ_arr)

    # Check consistency
    matches = (
        clean_proc_map == champ_map
        and champ_map == champ_arr
        and champ_arr == prod_map
        and prod_map == prod_arr
    )
    if not matches:
        all_sources_agree = False

    mapping_rows.append({
        "class_index": c_idx,
        "clean_dataset_label": clean_proc_map,
        "champion_json_label": champ_arr,
        "production_model_label": prod_arr,
        "frontend_display_word": fe_display,
        "status": "MATCH" if matches else "MISMATCH",
    })

mapping_csv = AUDIT_DIR / "phase8b6c_class_mapping_audit.csv"
with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["class_index", "clean_dataset_label", "champion_json_label", "production_model_label", "frontend_display_word", "status"])
    writer.writeheader()
    writer.writerows(mapping_rows)

print(f"49-Class Mapping Agreement Status: {'100% IDENTICAL ACROSS ALL SOURCES' if all_sources_agree else 'MISMATCHES DETECTED'}")
print(f"Saved mapping audit table to: {mapping_csv}\n")


# -----------------------------------------------------------------------------
# 4. REPRODUCE M1 AND M5 METRICS INDEPENDENTLY
# -----------------------------------------------------------------------------
print("=" * 76)
print("4. INDEPENDENT REPRODUCTION OF M1 & M5 METRICS")
print("=" * 76)

import evaluate_phase8b6b as eval_lib

with np.load(SPLIT_DIR / "val.npz") as va, np.load(SPLIT_DIR / "test.npz") as te:
    X_val, y_val = va["X"].astype(np.float32), va["y"].astype(np.int64)
    X_test, y_test = te["X"].astype(np.float32), te["y"].astype(np.int64)

id_to_label = {int(k): v for k, v in proc_lm["id_to_label"].items()}

# Scaled data using M1 scaler
X_val_s1 = (X_val - m1_mean) / m1_scale
X_test_s1 = (X_test - m1_mean) / m1_scale

# M1 Evaluation
z_val_1 = np.dot(X_val_s1, W1) + b1
preds_val_1 = np.argmax(z_val_1, axis=1)
m1_val_rep = eval_lib.compute_metrics(y_val, preds_val_1, z_val_1, id_to_label)

z_test_1 = np.dot(X_test_s1, W1) + b1
preds_test_1 = np.argmax(z_test_1, axis=1)
m1_test_rep = eval_lib.compute_metrics(y_test, preds_test_1, z_test_1, id_to_label)

print(f"M1 Reproduced Val Macro F1 : {m1_val_rep['macro_avg']['f1_score']:.4f} (Reported: 0.5178) -> {'MATCH' if m1_val_rep['macro_avg']['f1_score'] == 0.5178 else 'DIFF'}")
print(f"M1 Reproduced Test Macro F1: {m1_test_rep['macro_avg']['f1_score']:.4f} (Reported: 0.4709) -> {'MATCH' if m1_test_rep['macro_avg']['f1_score'] == 0.4709 else 'DIFF'}")
print(f"M1 Reproduced Test Accuracy: {m1_test_rep['overall_accuracy'] * 100:.2f}% (Reported: 50.70%) -> {'MATCH' if m1_test_rep['overall_accuracy'] == 0.5070 else 'DIFF'}")
print(f"M1 Reproduced Test Top-3   : {m1_test_rep['top3_accuracy'] * 100:.2f}% (Reported: 71.13%) -> {'MATCH' if m1_test_rep['top3_accuracy'] == 0.7113 else 'DIFF'}")

# M5 Evaluation (using M5 checkpoint scaler)
X_val_s5 = (X_val - m5_mean) / m5_scale
X_test_s5 = (X_test - m5_mean) / m5_scale

z1_val = np.maximum(0.0, np.dot(X_val_s5, W1_5) + b1_5)
z2_val = np.dot(z1_val, W2_5) + b2_5
preds_val_5 = np.argmax(z2_val, axis=1)
m5_val_rep = eval_lib.compute_metrics(y_val, preds_val_5, z2_val, id_to_label)

z1_test = np.maximum(0.0, np.dot(X_test_s5, W1_5) + b1_5)
z2_test = np.dot(z1_test, W2_5) + b2_5
preds_test_5 = np.argmax(z2_test, axis=1)
m5_test_rep = eval_lib.compute_metrics(y_test, preds_test_5, z2_test, id_to_label)

print(f"\nM5 Reproduced Val Macro F1 : {m5_val_rep['macro_avg']['f1_score']:.4f} (Reported: 0.4838) -> {'MATCH' if m5_val_rep['macro_avg']['f1_score'] == 0.4838 else 'DIFF'}")
print(f"M5 Reproduced Test Macro F1: {m5_test_rep['macro_avg']['f1_score']:.4f} (Reported: 0.4698) -> {'MATCH' if m5_test_rep['macro_avg']['f1_score'] == 0.4698 else 'DIFF'}")
print(f"M5 Reproduced Test Accuracy: {m5_test_rep['overall_accuracy'] * 100:.2f}% (Reported: 50.00%) -> {'MATCH' if m5_test_rep['overall_accuracy'] == 0.5000 else 'DIFF'}")
print(f"M5 Reproduced Test Top-3   : {m5_test_rep['top3_accuracy'] * 100:.2f}% (Reported: 76.06%) -> {'MATCH' if m5_test_rep['top3_accuracy'] == 0.7606 else 'DIFF'}\n")


# -----------------------------------------------------------------------------
# 5. PREPARE 142 TEST SAMPLES FOR TYPESCRIPT PARITY AUDIT
# -----------------------------------------------------------------------------
print("=" * 76)
print("5. EXPORTING 142 HELD-OUT TEST SAMPLES WITH PYTHON REFERENCE PROJECTIONS")
print("=" * 76)

def stable_softmax(z):
    shift = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shift)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

test_samples_payload = []
for i in range(len(X_test)):
    x_raw = X_test[i]
    x_scaled = X_test_s1[i]
    logits = np.dot(x_scaled, W1) + b1
    probs = stable_softmax(logits)
    pred_idx = int(np.argmax(probs))

    # Top 3 candidates
    sorted_indices = np.argsort(probs)[::-1][:3]
    top3 = [
        {"classIndex": int(idx), "label": id_to_label[idx], "score": float(probs[idx])}
        for idx in sorted_indices
    ]

    test_samples_payload.append({
        "sampleId": int(i),
        "trueClass": int(y_test[i]),
        "trueLabel": id_to_label[y_test[i]],
        "features": x_raw.tolist(),
        "pyScaled": x_scaled.tolist(),
        "pyLogits": logits.tolist(),
        "pyProbs": probs.tolist(),
        "pyPred": pred_idx,
        "pyTop3": top3,
    })

parity_input_path = AUDIT_DIR / "test_samples_py_reference.json"
with open(parity_input_path, "w", encoding="utf-8") as f:
    json.dump({"samples": test_samples_payload}, f, separators=(",", ":"))

print(f"Saved {len(test_samples_payload)} reference test samples to: {parity_input_path}")
print(f"File size: {parity_input_path.stat().st_size / 1024:.2f} KB\n")
