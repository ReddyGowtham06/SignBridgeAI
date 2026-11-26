r"""
SIGNBRIDGE AI — Inference Engine Parity & Gate Verification Test
==================================================================
Runs automated end-to-end parity testing between the Python reference model
(`baseline_model.npz`) and the TypeScript inference engine (`signClassifier.ts`).

Tests:
1. All 109 samples from the held-out test split (`test.npz`).
2. Verification across all 49 sign classes.
3. Logit and probability numerical precision.
4. Top-1 and Top-3 candidate ranking agreement.
5. Execution of all four rejection gates:
   - Gate 1: Presence Rejection
   - Gate 2: Stationary Motion Rejection
   - Gate 3: Confidence Threshold Rejection
   - Gate 4: Temporal Debounce & Candidate Streak Tracking
6. Microsecond-level browser inference latency benchmarking.
"""

import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

# Line buffering
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required.")
    sys.exit(1)

import config

MODEL_CHECKPOINT = config.MODELS_DIR / "baseline_model.npz"
EXPORTED_JSON = config.PROJECT_ROOT / "frontend" / "public" / "models" / "sign_classifier.json"
TEST_DATA_PATH = config.PROCESSED_DATA_DIR / "test.npz"
LABEL_MAPPING_PATH = config.PROCESSED_DATA_DIR / "label_mapping.json"
JS_RUNNER = config.TRAINING_DIR / "test_inference_parity.mjs"
REPORTS_DIR = config.REPORTS_DIR


def stable_softmax(z: np.ndarray) -> np.ndarray:
    shift = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shift)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def generate_python_reference_data(X: np.ndarray, y: np.ndarray, id_to_label: Dict[int, str]) -> List[Dict[str, Any]]:
    with np.load(MODEL_CHECKPOINT) as m:
        W = m["W"].astype(np.float32)
        b = m["b"].astype(np.float32)
        mean = m["scaler_mean"].astype(np.float32)
        scale = m["scaler_scale"].astype(np.float32)

    X_scaled = (X - mean) / scale
    logits = np.dot(X_scaled, W) + b
    probs = stable_softmax(logits)
    preds = np.argmax(probs, axis=1)

    samples = []
    for i in range(len(X)):
        prob_row = probs[i]
        top3_indices = np.argsort(prob_row)[::-1][:3]
        top3 = [
            {
                "classIndex": int(idx),
                "label": id_to_label[int(idx)],
                "score": float(prob_row[idx]),
            }
            for idx in top3_indices
        ]
        samples.append({
            "sampleIndex": i,
            "trueClass": int(y[i]),
            "trueLabel": id_to_label[int(y[i])],
            "features": X[i].tolist(),
            "pyLogits": logits[i].tolist(),
            "pyProbs": prob_row.tolist(),
            "pyPred": int(preds[i]),
            "pyPredLabel": id_to_label[int(preds[i])],
            "pyTop3": top3,
        })

    return samples


def run_node_parity_runner(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as in_f:
        json.dump({"samples": samples}, in_f)
        in_path = in_f.name

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as out_f:
        out_path = out_f.name

    try:
        cmd = [
            "node",
            str(JS_RUNNER),
            str(EXPORTED_JSON),
            in_path,
            out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

        with open(out_path, "r", encoding="utf-8") as f:
            res_data = json.load(f)
        return res_data
    finally:
        try:
            Path(in_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass


def main():
    print("=" * 75)
    print("SIGNBRIDGE AI — PHASE 7C: TYPESCRIPT INFERENCE ENGINE PARITY & GATE TESTS")
    print("=" * 75)

    if not MODEL_CHECKPOINT.exists():
        raise FileNotFoundError(f"Model checkpoint missing: {MODEL_CHECKPOINT}")
    if not EXPORTED_JSON.exists():
        raise FileNotFoundError(f"Exported JSON model missing: {EXPORTED_JSON}")
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test data missing: {TEST_DATA_PATH}")

    # Load label mapping
    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        lbl_data = json.load(f)
        id_to_label = {int(k): v for k, v in lbl_data["id_to_label"].items()}

    # Load test split (109 samples across all 49 classes)
    with np.load(TEST_DATA_PATH) as td:
        X_test = td["X"].astype(np.float32)
        y_test = td["y"].astype(np.int64)

    n_samples = len(X_test)
    print(f"Generating Python checkpoint reference outputs for {n_samples} test samples...")
    t0 = time.time()
    samples = generate_python_reference_data(X_test, y_test, id_to_label)
    py_time = time.time() - t0
    print(f"Python evaluation completed in {py_time:.3f} s.")

    print(f"\nInvoking Node TypeScript inference runner: {JS_RUNNER.name}...")
    t0 = time.time()
    test_results = run_node_parity_runner(samples)
    runner_time = time.time() - t0
    print(f"Node execution completed in {runner_time:.3f} s.")

    parity = test_results["parity"]
    gates = test_results["rejectionGates"]

    # 1. Print Numerical Parity Results
    print("\n" + "-" * 75)
    print("1. NUMERICAL PARITY & PREDICTION ACCURACY (HELD-OUT TEST SET)")
    print("-" * 75)
    print(f"  Total Samples Evaluated:            {parity['totalSamples']}")
    print(f"  Classes Represented:                {len(set(y_test))} of {len(id_to_label)}")
    print(f"  Max Absolute Logit Difference:      {parity['maxLogitDiff']:.2e}")
    print(f"  Mean Absolute Logit Difference:     {parity['meanLogitDiff']:.2e}")
    print(f"  Max Absolute Probability Difference:{parity['maxProbDiff']:.2e}")
    print(f"  Mean Absolute Probability Diff:     {parity['meanProbDiff']:.2e}")
    print(f"  Top-1 Prediction Class Agreement:   {parity['predAgreementPercent']:.2f}% ({parity['predMatches']}/{parity['totalSamples']})")
    print(f"  Top-3 Candidate Ranking Agreement:  {parity['top3AgreementPercent']:.2f}% ({parity['top3Matches']}/{parity['totalSamples']})")
    print(f"  Mean Inference Execution Latency:   {parity['meanInferenceTimeUs']:.1f} µs ({parity['meanInferenceTimeMs']:.3f} ms / sample)")

    FULL_DATA_PATH = config.PROCESSED_DATA_DIR / "features_tabular.npz"
    parity_full = None
    if FULL_DATA_PATH.exists():
        with np.load(FULL_DATA_PATH) as fd:
            X_full = fd["X"].astype(np.float32)
            y_full = fd["y"].astype(np.int64)
        print(f"\nGenerating Python reference outputs for Full Dataset ({len(X_full)} samples)...")
        samples_full = generate_python_reference_data(X_full, y_full, id_to_label)
        results_full = run_node_parity_runner(samples_full)
        parity_full = results_full["parity"]

        print("-" * 75)
        print("2. NUMERICAL PARITY ON FULL FEATURE DATASET (862 SAMPLES)")
        print("-" * 75)
        print(f"  Total Samples Evaluated:            {parity_full['totalSamples']}")
        print(f"  Classes Represented:                {len(set(y_full))} of {len(id_to_label)}")
        print(f"  Max Absolute Logit Difference:      {parity_full['maxLogitDiff']:.2e}")
        print(f"  Max Absolute Probability Difference:{parity_full['maxProbDiff']:.2e}")
        print(f"  Top-1 Prediction Class Agreement:   {parity_full['predAgreementPercent']:.2f}% ({parity_full['predMatches']}/{parity_full['totalSamples']})")
        print(f"  Top-3 Candidate Ranking Agreement:  {parity_full['top3AgreementPercent']:.2f}% ({parity_full['top3Matches']}/{parity_full['totalSamples']})")
        print(f"  Mean Inference Latency (Full Run):  {parity_full['meanInferenceTimeUs']:.1f} µs ({parity_full['meanInferenceTimeMs']:.3f} ms / sample)")

    # 3. Print Gesture Rejection Gate Results
    print("\n" + "-" * 75)
    print("3. GESTURE REJECTION GATES EVALUATION")
    print("-" * 75)

    g1 = gates["gate1_presence"]
    print(f"  Gate 1 — Dual-Hand Presence (< 0.20):       {'[PASSED]' if g1['passed'] else '[FAILED]'} (Result: {g1['status']}, Reason: {g1['reason']})")

    g2 = gates["gate2_motion"]
    print(f"  Gate 2 — Stationary Motion (< 0.05 units/s):{'[PASSED]' if g2['passed'] else '[FAILED]'} (Result: {g2['status']}, Reason: {g2['reason']})")

    g3 = gates["gate3_confidence"]
    print(f"  Gate 3 — Low Confidence Threshold:          {'[PASSED]' if g3['passed'] else '[FAILED]'} (Result: {g3['status']}, Reason: {g3['reason']})")

    g4 = gates["gate4_temporal_debounce"]
    print(f"  Gate 4 — Temporal Debouncer (Streak >= 2):  {'[PASSED]' if g4['passed'] else '[FAILED]'}")
    for s in g4["steps"]:
        conf_str = f"({s['confirmed']})" if s.get("confirmed") else ""
        print(f"            Step {s['step']}: Streak {s['count']} -> Confirmed: {s['isConfirmed']} {conf_str}")

    # Check overall criteria
    parity_passed = (
        parity["maxLogitDiff"] < 1e-4
        and parity["maxProbDiff"] < 1e-4
        and parity["predAgreementPercent"] == 100.0
        and parity["top3AgreementPercent"] == 100.0
    )
    gates_passed = (
        g1["passed"]
        and g2["passed"]
        and g3["passed"]
        and g4["passed"]
    )
    all_passed = parity_passed and gates_passed

    # Save detailed report artifact
    report_path = REPORTS_DIR / "inference_parity_report.json"
    full_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_file": str(EXPORTED_JSON.name),
        "test_samples_count": parity["totalSamples"],
        "all_passed": all_passed,
        "parity_test_split": parity,
        "parity_full_dataset": parity_full,
        "rejection_gates": gates,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 75)
    print(f"OVERALL PHASE 7C VERIFICATION: {'100% PASSED' if all_passed else 'FAILED'}")
    print(f"Detailed report saved to: {report_path}")
    print("=" * 75)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
