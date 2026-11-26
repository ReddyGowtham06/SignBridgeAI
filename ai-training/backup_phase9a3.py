import hashlib
import shutil
import os

model_path = 'frontend/public/models/sign_classifier.json'
with open(model_path, 'rb') as f:
    h = hashlib.sha256(f.read()).hexdigest().lower()

print('Model SHA-256:', h)
expected_hash = 'c8068a3fedd85645d736d16ebde450bd442970784adba40fdacb8970a4535274'
if h != expected_hash:
    raise ValueError(f'FATAL: Expected hash {expected_hash}, got {h}')
print('VERIFIED: Production model hash matches expected M1 champion.')

backup_dir = 'ai-training/reports/phase9a3/backups'
os.makedirs(backup_dir, exist_ok=True)

targets = [
    ('frontend/src/utils/signClassifier.ts', 'signClassifier.ts.bak_20261125_phase9a3'),
    ('frontend/src/pages/app/SignToTextPage.tsx', 'SignToTextPage.tsx.bak_20261125_phase9a3'),
    ('frontend/src/utils/adaptiveMotionGate.ts', 'adaptiveMotionGate.ts.bak_20261125_phase9a3'),
]

for src, name in targets:
    dst = os.path.join(backup_dir, name)
    shutil.copy2(src, dst)
    print(f'Backup created: {dst} ({os.path.getsize(dst)} bytes)')

