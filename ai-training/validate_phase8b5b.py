import json
import csv
from pathlib import Path
import numpy as np

SPLIT_DIR = Path("ai-training/data/staged_phase8b5b_split")
REPORTS_DIR = Path("ai-training/reports")
RAW_VIDEO_DIR = Path(r"D:\SignLang\sign project")
RAW_LANDMARKS_DIR = Path("ai-training/data/raw_landmarks")
PROCESSED_DIR = Path("ai-training/data/processed")
MODELS_DIR = Path("ai-training/models")

print("=== INDEPENDENT POST-SPLIT VALIDATION ===")

# 1. Check NPZ files
npz_files = ["train.npz", "val.npz", "test.npz"]
expected_counts = {"train.npz": 631, "val.npz": 142, "test.npz": 142}

loaded_data = {}
for fname in npz_files:
    p = SPLIT_DIR / fname
    assert p.exists(), f"Missing {fname}"
    with np.load(p) as data:
        X = data["X"]
        y = data["y"]
        sids = [str(s) for s in data["sample_ids"]]
        vnames = [str(v) for v in data["video_names"]]
        cnames = [str(c) for c in data["class_names"]]
        loaded_data[fname] = {
            "X": X, "y": y, "sample_ids": sids,
            "video_names": vnames, "class_names": cnames
        }
    exp_n = expected_counts[fname]
    assert X.shape == (exp_n, 758), f"{fname} shape {X.shape} != ({exp_n}, 758)"
    assert X.dtype == np.float32, f"{fname} dtype {X.dtype} != float32"
    assert not np.isnan(X).any(), f"NaN in {fname}"
    assert not np.isinf(X).any(), f"Inf in {fname}"
    assert len(y) == exp_n
    assert len(sids) == exp_n
    print(f"[OK] {fname}: shape={X.shape}, dtype={X.dtype}, 0 NaNs, 0 Infs.")

# 2. Check Sample Disjointness
tr_sids = set(loaded_data["train.npz"]["sample_ids"])
vl_sids = set(loaded_data["val.npz"]["sample_ids"])
te_sids = set(loaded_data["test.npz"]["sample_ids"])

assert len(tr_sids) == 631
assert len(vl_sids) == 142
assert len(te_sids) == 142

assert len(tr_sids.intersection(vl_sids)) == 0, "Train-Val sample overlap!"
assert len(tr_sids.intersection(te_sids)) == 0, "Train-Test sample overlap!"
assert len(vl_sids.intersection(te_sids)) == 0, "Val-Test sample overlap!"
assert len(tr_sids.union(vl_sids).union(te_sids)) == 915, "Union != 915!"
print("[OK] Sample Disjointness: exactly 631 + 142 + 142 = 915 disjoint samples.")

# 3. Check CSV Manifests
with open(SPLIT_DIR / "phase8b5b_split_manifest.csv", "r", encoding="utf-8") as f:
    split_manifest = list(csv.DictReader(f))
assert len(split_manifest) == 915, f"Split manifest len {len(split_manifest)} != 915"

with open(SPLIT_DIR / "phase8b5b_group_assignment.csv", "r", encoding="utf-8") as f:
    group_manifest = list(csv.DictReader(f))
assert len(group_manifest) == 282, f"Group manifest len {len(group_manifest)} != 282"

with open(SPLIT_DIR / "phase8b5b_class_distribution.csv", "r", encoding="utf-8") as f:
    class_dist = list(csv.DictReader(f))
assert len(class_dist) == 49, f"Class dist len {len(class_dist)} != 49"

print("[OK] Manifests verified: 915 split rows, 282 group rows, 49 class rows.")

# 4. Check Group Leakage
group_partitions = {}
for r in split_manifest:
    gid = r["session_burst_id"]
    part = r["assigned_partition"]
    if gid in group_partitions:
        assert group_partitions[gid] == part, f"Group {gid} split across partitions!"
    else:
        group_partitions[gid] = part

assert len(group_partitions) == 282
print("[OK] Zero Group Leakage: all 282 session bursts are 100% contained within a single partition.")

# 5. Check Class Coverage
tr_classes = set(loaded_data["train.npz"]["class_names"])
vl_classes = set(loaded_data["val.npz"]["class_names"])
te_classes = set(loaded_data["test.npz"]["class_names"])

assert len(tr_classes) == 49, f"Train classes {len(tr_classes)} != 49"
assert len(vl_classes) == 49, f"Val classes {len(vl_classes)} != 49"
assert len(te_classes) == 49, f"Test classes {len(te_classes)} != 49"

for r in class_dist:
    assert int(r["train_samples"]) >= 10, f"Train samples for {r['class_name']} < 10"
    assert int(r["val_samples"]) >= 2, f"Val samples for {r['class_name']} < 2"
    assert int(r["test_samples"]) >= 2, f"Test samples for {r['class_name']} < 2"

print("[OK] Full Class Coverage: all 49 classes present in Train, Val, Test. Val & Test >= 2 per class.")

# 6. Check Preservation
raw_videos_count = sum(1 for _ in RAW_VIDEO_DIR.glob("*/*.*"))
assert raw_videos_count == 978, f"Raw video count {raw_videos_count} != 978"
raw_lm_count = sum(1 for _ in RAW_LANDMARKS_DIR.glob("*/*.npz"))
assert raw_lm_count == 862, f"Raw landmarks count {raw_lm_count} != 862"
assert (PROCESSED_DIR / "features_tabular.npz").exists(), "Baseline processed missing"
assert (MODELS_DIR / "baseline_model.npz").exists(), "Baseline model missing"

print("[OK] Preservation Check: 978 raw videos, 862 raw landmarks, baseline features/models intact.")
print("=== ALL INDEPENDENT POST-SPLIT VALIDATIONS PASSED CLEANLY! ===")
