import os
import glob
import csv
import json
import numpy as np

print("=== STEP 1: INSPECTING PHASE 8B.9 ATTEMPT LOG ===")
log_path = 'ai-training/reports/phase8b9/phase8b9_attempt_log.csv'
rows = []
with open(log_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f"Total trials in {log_path}: {len(rows)}")

print("\n--- Specific Inspection: TR-040, TR-041, TR-042 (21.good) ---")
for r in rows:
    if r['trial_id'] in ['TR-040', 'TR-041', 'TR-042']:
        print(f"[{r['trial_id']}] Class: {r['class_label']} (#{r['attempt_number']}) | Result: {r['recognition_correctness']}")
        print(f"    Top-1: {r['top1_prediction']} (conf: {r['top1_confidence']}) | Gate: {r['rejection_reason']}")
        print(f"    Debounced: {r['debounced_transcript']} | Hands: {r['hand_detection_status']} (L:{r['left_presence_ratio']}, R:{r['right_presence_ratio']})")
        print(f"    Top-3: {r['top3_candidates']}")
        print(f"    Latency: {r['recognition_latency_ms']}ms | Notes: {r['evaluator_notes']}\n")

print("\n--- All 21.good Trials (Class ID 13) ---")
for r in rows:
    if r['class_label'] == '21.good':
        print(f"[{r['trial_id']}] #{r['attempt_number']} | Top-1: {r['top1_prediction']} ({r['top1_confidence']}) | Gate: {r['rejection_reason']} | Result: {r['recognition_correctness']} | Confirmed: {r['debounced_transcript']} | Notes: {r['evaluator_notes']}")

print("\n--- All 34.stop Trials (Class ID 27) ---")
for r in rows:
    if r['class_label'] == '34.stop':
        print(f"[{r['trial_id']}] #{r['attempt_number']} | Top-1: {r['top1_prediction']} ({r['top1_confidence']}) | Gate: {r['rejection_reason']} | Result: {r['recognition_correctness']} | Confirmed: {r['debounced_transcript']} | Notes: {r['evaluator_notes']}")

print("\n--- All 45.hand Trials (Class ID 39) ---")
for r in rows:
    if r['class_label'] == '45.hand':
        print(f"[{r['trial_id']}] #{r['attempt_number']} | Top-1: {r['top1_prediction']} ({r['top1_confidence']}) | Gate: {r['rejection_reason']} | Result: {r['recognition_correctness']} | Confirmed: {r['debounced_transcript']} | Notes: {r['evaluator_notes']}")

print("\n=== STEP 2: CHECKING FOR RAW TRIAL ARTIFACTS IN REPOSITORY ===")
p8b9_files = glob.glob('ai-training/reports/phase8b9/**/*', recursive=True)
print("Files in ai-training/reports/phase8b9/:")
for f in p8b9_files:
    size = os.path.getsize(f)
    print(f"  {f} ({size} bytes)")

trial_dumps = (
    glob.glob('ai-training/**/TR-*.npy', recursive=True) +
    glob.glob('ai-training/**/TR-*.npz', recursive=True) +
    glob.glob('ai-training/**/phase8b9*frames*', recursive=True) +
    glob.glob('ai-training/**/phase8b9*landmarks*', recursive=True)
)
print(f"\nDiscovered trial frame/landmark dump files: {trial_dumps}")
