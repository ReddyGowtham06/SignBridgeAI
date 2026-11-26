import csv
import json

with open('ai-training/reports/phase8b9/phase8b9_attempt_log.csv', 'r', encoding='utf-8') as f:
    trials = list(csv.DictReader(f))

with open('ai-training/reports/phase8b9/phase8b9_full_benchmark_report.json', 'r', encoding='utf-8') as f:
    rep_json = json.load(f)

# Recompute aggregate metrics
total = len(trials)
correct = len([t for t in trials if t['recognition_correctness'] == 'correct'])
incorrect = len([t for t in trials if t['recognition_correctness'] == 'incorrect'])
rejected = len([t for t in trials if t['recognition_correctness'] == 'rejected'])
undetected = len([t for t in trials if t['recognition_correctness'] == 'undetected'])
invalid = len([t for t in trials if t['recognition_correctness'] == 'invalid'])
accepted = len([t for t in trials if t['rejection_reason'] == 'none'])

top3_overall = len([
    t for t in trials
    if t['class_label'] in [cand.split(':')[0] for cand in t['top3_candidates'].split('|')]
])

top3_accepted = len([
    t for t in trials
    if t['rejection_reason'] == 'none' and t['class_label'] in [cand.split(':')[0] for cand in t['top3_candidates'].split('|')]
])

rej_presence = len([t for t in trials if t['rejection_reason'] == 'presence_too_low'])
rej_motion = len([t for t in trials if t['rejection_reason'] == 'motion_stationary'])
rej_conf = len([t for t in trials if t['rejection_reason'] == 'confidence_low'])

latencies = [int(t['recognition_latency_ms']) for t in trials if t['recognition_latency_ms'] != '']
latencies.sort()
med_lat = latencies[len(latencies)//2]
p90_lat = latencies[int(len(latencies) * 0.90)]
p95_lat = latencies[int(len(latencies) * 0.95)]

e2e_acc = round(correct / total * 100, 2)
acc_acc = round(correct / accepted * 100, 2)
rej_rate = round(rejected / total * 100, 2)
top3_rate = round(top3_overall / total * 100, 2)

print("=== RECOMPUTED VS REPORTED ===")
print(f"Total: recomputed={total}, reported={rep_json['aggregate_metrics']['total_completed_attempts']}")
print(f"Correct: recomputed={correct}, reported={rep_json['aggregate_metrics']['correct_recognitions']}")
print(f"Incorrect: recomputed={incorrect}, reported={rep_json['aggregate_metrics']['incorrect_recognitions']}")
print(f"Rejected: recomputed={rejected}, reported={rep_json['aggregate_metrics']['rejected_attempts']}")
print(f"Accepted: recomputed={accepted}, reported={rep_json['aggregate_metrics']['accepted_attempts']}")
print(f"E2E Acc: recomputed={e2e_acc}%, reported={rep_json['aggregate_metrics']['end_to_end_accuracy_percent']}%")
print(f"Accepted Acc: recomputed={acc_acc}%, reported={rep_json['aggregate_metrics']['top1_accepted_accuracy_percent']}%")
print(f"Top-3 Acc: recomputed={top3_rate}%, reported={rep_json['aggregate_metrics']['top3_overall_accuracy_percent']}%")
print(f"Rejection Rate: recomputed={rej_rate}%, reported={rep_json['aggregate_metrics']['rejection_rate_percent']}%")
print(f"Median Latency: recomputed={med_lat} ms, reported={rep_json['aggregate_metrics']['median_latency_ms']} ms")
print(f"P90 Latency: recomputed={p90_lat} ms, reported={rep_json['aggregate_metrics']['p90_latency_ms']} ms")
print(f"P95 Latency: recomputed={p95_lat} ms, reported={rep_json['aggregate_metrics']['p95_latency_ms']} ms")
print(f"Rejections breakdown: presence={rej_presence}, motion={rej_motion}, conf={rej_conf}")
print(f"Reported rejections: {rep_json['rejection_breakdown']}")

# Check per-class metrics consistency
with open('ai-training/reports/phase8b9/phase8b9_per_class_metrics.csv', 'r', encoding='utf-8') as f:
    pc_rows = list(csv.DictReader(f))

mismatches = []
for cid in range(49):
    c_trials = [t for t in trials if int(t['class_id']) == cid]
    c_corr = len([t for t in c_trials if t['recognition_correctness'] == 'correct'])
    c_inc = len([t for t in c_trials if t['recognition_correctness'] == 'incorrect'])
    c_rej = len([t for t in c_trials if t['recognition_correctness'] == 'rejected'])
    c_acc = len([t for t in c_trials if t['rejection_reason'] == 'none'])
    c_e2e = round(c_corr / len(c_trials) * 100, 2)
    c_top1_acc = round(c_corr / c_acc * 100, 2) if c_acc > 0 else 0.0

    pc = pc_rows[cid]
    if int(pc['correct']) != c_corr:
        mismatches.append((cid, 'correct', pc['correct'], c_corr))
    if int(pc['incorrect']) != c_inc:
        mismatches.append((cid, 'incorrect', pc['incorrect'], c_inc))
    if int(pc['rejected']) != c_rej:
        mismatches.append((cid, 'rejected', pc['rejected'], c_rej))
    if float(pc['end_to_end_acc_pct']) != c_e2e:
        mismatches.append((cid, 'e2e_acc', pc['end_to_end_acc_pct'], c_e2e))

print(f"Per-class mismatches count: {len(mismatches)}")
if mismatches:
    print("Mismatches found:", mismatches[:5])
else:
    print("ALL 49 CLASSES MATCH RAW DATA WITH ZERO DISCREPANCIES!")

# Check confusion matrix consistency
with open('ai-training/reports/phase8b9/phase8b9_confusion_matrix.csv', 'r', encoding='utf-8') as f:
    cm_rows = list(csv.reader(f))

cm_header = cm_rows[0][1:]
cm_matrix = [[int(x) for x in row[1:]] for row in cm_rows[1:]]

cm_mismatches = 0
for t in trials:
    gt_id = int(t['class_id'])
    pred_lbl = t['top1_prediction']
    pred_id = cm_header.index(pred_lbl)
    # The script generated matrix had cell increments
# Recompute full matrix from trials
recomputed_cm = [[0 for _ in range(49)] for _ in range(49)]
for t in trials:
    gt_id = int(t['class_id'])
    pred_lbl = t['top1_prediction']
    pred_id = cm_header.index(pred_lbl)
    recomputed_cm[gt_id][pred_id] += 1

diffs = 0
for i in range(49):
    for j in range(49):
        if cm_matrix[i][j] != recomputed_cm[i][j]:
            diffs += 1
print(f"Confusion matrix differences: {diffs}")
