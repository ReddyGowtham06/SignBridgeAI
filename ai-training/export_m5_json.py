import json
import numpy as np

# Load M1 to get exact labels and id_to_label
with open("frontend/public/models/sign_classifier.json", "r", encoding="utf-8") as f:
    m1_data = json.load(f)

# Load M5 npz
m5 = np.load("ai-training/models/phase8b6b/m5_small_mlp.npz")

m5_json = {
    "model_type": "m5_small_mlp",
    "version": "9.1.0-phase9a1",
    "feature_dim": 758,
    "hidden_dim": 128,
    "num_classes": 49,
    "classes": [int(x) for x in m5["classes"]],
    "labels": m1_data["labels"],
    "id_to_label": m1_data["id_to_label"],
    "scaler_mean": [float(x) for x in m5["scaler_mean"]],
    "scaler_scale": [float(x) for x in m5["scaler_scale"]],
    "W1": [[float(y) for y in row] for row in m5["W1"]],
    "b1": [float(x) for x in m5["b1"]],
    "W2": [[float(y) for y in row] for row in m5["W2"]],
    "b2": [float(x) for x in m5["b2"]],
}

# Save in ai-training and frontend/public/models/m5_evaluator.json
p1 = "ai-training/models/phase8b6b/m5_evaluator.json"
p2 = "frontend/public/models/m5_evaluator.json"

with open(p1, "w", encoding="utf-8") as f:
    json.dump(m5_json, f)
print(f"Exported M5 JSON to {p1}")

with open(p2, "w", encoding="utf-8") as f:
    json.dump(m5_json, f)
print(f"Exported M5 JSON to {p2}")
