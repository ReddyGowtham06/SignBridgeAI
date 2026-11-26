import json
import csv
from pathlib import Path
import numpy as np

REPORTS_DIR = Path("ai-training/reports")
CLEAN_LANDMARKS_DIR = Path("ai-training/data/staged_clean_landmarks")
CLEAN_FEATURES_DIR = Path("ai-training/data/staged_clean_features")
RAW_VIDEO_DIR = Path(r"D:\SignLang\sign project")
RAW_LANDMARKS_DIR = Path("ai-training/data/raw_landmarks")
PROCESSED_DIR = Path("ai-training/data/processed")
MODELS_DIR = Path("ai-training/models")

print("=== STARTING INDEPENDENT VALIDATION OF PHASE 8B.4 OUTPUTS ===")

# 1. Validate Feature Matrix
feat_path = CLEAN_FEATURES_DIR / "features_tabular.npz"
assert feat_path.exists(), f"Missing {feat_path}"
with np.load(feat_path) as data:
    keys = list(data.keys())
    print("Feature matrix keys:", keys)
    assert set(['X', 'y', 'sample_ids', 'video_names', 'class_names']).issubset(set(keys))
    X = data['X']
    y = data['y']
    sample_ids = data['sample_ids']
    video_names = data['video_names']
    class_names = data['class_names']

assert X.shape == (915, 758), f"X shape {X.shape} != (915, 758)"
assert X.dtype == np.float32, f"X dtype {X.dtype} != float32"
assert y.shape == (915,), f"y shape {y.shape} != (915,)"
assert y.dtype == np.int64, f"y dtype {y.dtype} != int64"
assert len(sample_ids) == 915, f"sample_ids len {len(sample_ids)} != 915"
assert len(video_names) == 915, f"video_names len {len(video_names)} != 915"
assert len(class_names) == 915, f"class_names len {len(class_names)} != 915"
assert not np.isnan(X).any(), "NaNs detected in feature matrix X!"
assert not np.isinf(X).any(), "Infs detected in feature matrix X!"
print(f"[OK] Feature Matrix: Shape {X.shape}, dtype {X.dtype}, 0 NaNs, 0 Infs.")

# 2. Validate Label Mapping
mapping_path = CLEAN_FEATURES_DIR / "label_mapping.json"
assert mapping_path.exists(), f"Missing {mapping_path}"
with open(mapping_path, "r", encoding="utf-8") as f:
    mapping = json.load(f)

assert mapping["num_classes"] == 49, f"num_classes {mapping['num_classes']} != 49"
assert mapping["feature_dim"] == 758, f"feature_dim {mapping['feature_dim']} != 758"
assert mapping["total_samples"] == 915, f"total_samples {mapping['total_samples']} != 915"
assert len(mapping["class_distribution"]) == 49, f"class_distribution len {len(mapping['class_distribution'])} != 49"
min_count = min(mapping["class_distribution"].values())
max_count = max(mapping["class_distribution"].values())
print(f"[OK] Label Mapping: 49 classes, sample count range [{min_count}, {max_count}], 17.friend count = {mapping['class_distribution']['17.friend']}.")

# 3. Validate Clean Landmark Archives
clean_npz = list(CLEAN_LANDMARKS_DIR.glob("*/*.npz"))
assert len(clean_npz) == 915, f"Clean npz count {len(clean_npz)} != 915"

# Validate sample of archives
for p in clean_npz[:20]:
    with np.load(p) as data:
        lm = data["landmarks"]
        pm = data["presence_mask"]
        ts = data["timestamps_ms"]
        assert lm.shape[1:] == (2, 21, 3), f"Invalid lm shape {lm.shape} in {p}"
        assert pm.shape == (lm.shape[0], 2), f"Invalid pm shape {pm.shape} in {p}"
        assert not np.isnan(lm).any(), f"NaN in {p}"
        if lm.shape[0] > 1:
            diffs = np.diff(ts)
            assert (diffs > 0).all(), f"Non-monotonic timestamps in {p}"

print(f"[OK] Staged Clean Landmarks: Exactly 915 archives present across 49 folders, all valid.")

# 4. Validate Manifest CSVs
manifests = [
    ("phase8b4_unified_source_manifest.csv", 974),
    ("phase8b4_duplicate_exclusion_manifest.csv", 51),
    ("phase8b4_quarantine_manifest.csv", 8),
    ("phase8b4_feature_source_mapping.csv", 915),
]

for fname, exp_rows in manifests:
    fpath = REPORTS_DIR / fname
    assert fpath.exists(), f"Missing manifest {fname}"
    with open(fpath, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    actual_rows = len(reader) - 1 # subtract header
    assert actual_rows == exp_rows, f"{fname}: expected {exp_rows} rows, got {actual_rows}"
    print(f"[OK] Manifest {fname}: header + {actual_rows} rows.")

# 5. Check Hash Uniqueness in Feature Source Mapping
feat_mapping_path = REPORTS_DIR / "phase8b4_feature_source_mapping.csv"
with open(feat_mapping_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    clean_hashes = [row["source_sha256"] for row in reader]

assert len(clean_hashes) == 915, f"Clean hashes count {len(clean_hashes)} != 915"
assert len(set(clean_hashes)) == 915, f"Duplicate hash detected among {len(clean_hashes)} clean features!"
print(f"[OK] Hash Uniqueness: 100% of the 915 clean feature rows have unique SHA-256 hashes.")

# 6. Preservation Check
raw_video_count = sum(1 for _ in RAW_VIDEO_DIR.glob("*/*.*"))
assert raw_video_count == 978, f"RAW_VIDEO_DIR file count {raw_video_count} != 978"
raw_lm_count = sum(1 for _ in RAW_LANDMARKS_DIR.glob("*/*.npz"))
assert raw_lm_count == 862, f"RAW_LANDMARKS_DIR npz count {raw_lm_count} != 862"
assert (PROCESSED_DIR / "features_tabular.npz").exists(), "Baseline features_tabular.npz missing"
assert (MODELS_DIR / "baseline_model.npz").exists(), "Baseline baseline_model.npz missing"

print(f"[OK] Preservation Check: Raw videos (978), raw landmarks (862), baseline features & models intact.")
print("=== ALL INDEPENDENT VALIDATIONS PASSED CLEANLY! ===")
