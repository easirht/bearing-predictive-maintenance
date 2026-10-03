import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import FILES_BY_SIZE, load_signal
from src.features import extract_all_features

ENV = ["env_bpfo", "env_bpfi", "env_bsf"]
N = 150  # windows per (size, load, class)

frames = []
for size, by_load in FILES_BY_SIZE.items():
    for hp, files in by_load.items():
        for label, fname in files.items():
            df = extract_all_features(load_signal(fname), hp)
            df["label"] = label
            frames.append(df.sample(n=min(N, len(df)), random_state=0))
data = pd.concat(frames, ignore_index=True)

model = RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(data[ENV], data["label"])
joblib.dump({"model": model, "features": ENV}, ROOT / "app" / "model.joblib")
print("model saved, classes:", list(model.classes_))

# small sample signals for the dashboard demo (0.007 in faults, 1 HP, 2 s each)
samples = {label: load_signal(fname)[:24000].astype(np.float32)
           for label, fname in FILES_BY_SIZE["0.007"][1].items()}
np.savez_compressed(ROOT / "app" / "samples.npz", **samples)
print("samples saved")