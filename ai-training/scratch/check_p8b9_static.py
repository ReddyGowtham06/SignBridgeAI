import csv

with open('ai-training/reports/phase8b9/phase8b9_attempt_log.csv', 'r') as f:
    reader = csv.DictReader(f)
    trials = list(reader)

for target in ['21.good', '34.stop', '45.hand']:
    print(f"\n=== HISTORICAL TRIALS FOR {target} ===")
    t_list = [t for t in trials if t['class_label'] == target]
    for t in t_list:
        print(f"[{t['trial_id']}] #{t['attempt_number']} | Pred: {t['top1_prediction']:<12} ({t['top1_confidence']}) | Gate: {t['rejection_reason']:<18} | Result: {t['recognition_correctness']}")
