import csv

with open('ai-training/reports/phase8b9/phase8b9_attempt_log.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

def inspect_classes(cids):
    for cid in cids:
        print(f"=== Class {cid} ===")
        for r in rows:
            if int(r['class_id']) == cid:
                print(f"  Att {r['attempt_number']} ({r['trial_id']}): outcome={r['recognition_correctness']} rej={r['rejection_reason']} top1={r['top1_prediction']} (conf={r['top1_confidence']}) top3={r['top3_candidates']} det={r['hand_detection_status']} L={r['left_presence_ratio']} R={r['right_presence_ratio']} notes={r['evaluator_notes']}")

inspect_classes([13, 27, 39, 8, 36, 41, 46, 7, 11, 17, 33, 38, 47])
