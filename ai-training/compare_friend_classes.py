import json
from pathlib import Path
import numpy as np

classes_to_compare = ["17.friend", "1.bird", "19.girl", "37.teacher", "45.hand", "6.fish", "24.help"]
raw_dir = Path("ai-training/data/raw_landmarks")

with open("ai-training/reports/evaluation_softmax_regression.json") as f:
    eval_data = json.load(f)

print(f"{'Class':<15} {'F1-Score':<10} {'Recall':<10} {'LeftPres':<10} {'RightPres':<10} {'BothPres':<10} {'Slot0Consec':<12} {'Slot1Consec':<12}")
print("-" * 90)

for cname in classes_to_compare:
    c_eval = eval_data["per_class"].get(cname, {})
    f1 = c_eval.get("f1_score", 0.0)
    rec = c_eval.get("recall", 0.0)

    cdir = raw_dir / cname
    files = list(cdir.glob("*.npz"))

    l_list, r_list, both_list = [], [], []
    c0_list, c1_list = [], []

    for f in files:
        d = np.load(f)
        mask = d["presence_mask"]
        n = len(mask)
        if n == 0: continue
        l_list.append(np.mean(mask[:, 0]))
        r_list.append(np.mean(mask[:, 1]))
        both_list.append(np.mean(mask[:, 0] & mask[:, 1]))
        c0_list.append(np.sum(mask[:-1, 0] & mask[1:, 0]) / n)
        c1_list.append(np.sum(mask[:-1, 1] & mask[1:, 1]) / n)

    avg_l = np.mean(l_list) if l_list else 0
    avg_r = np.mean(r_list) if r_list else 0
    avg_both = np.mean(both_list) if both_list else 0
    avg_c0 = np.mean(c0_list) if c0_list else 0
    avg_c1 = np.mean(c1_list) if c1_list else 0

    print(f"{cname:<15} {f1:<10.3f} {rec:<10.3f} {avg_l:<10.3f} {avg_r:<10.3f} {avg_both:<10.3f} {avg_c0:<12.3f} {avg_c1:<12.3f}")

