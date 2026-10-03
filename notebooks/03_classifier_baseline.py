import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from src.data_loader import FILES_BY_LOAD, load_signal
from src.features import extract_features

N_PER_CLASS = 200  # windows per class per load (balanced)

# 1. Build one feature table for all loads
frames = []
for hp, files in FILES_BY_LOAD.items():
    for label, fname in files.items():
        df = extract_features(load_signal(fname))
        df = df.sample(n=min(N_PER_CLASS, len(df)), random_state=0)
        df["label"] = label
        df["load"] = hp
        frames.append(df)
data = pd.concat(frames, ignore_index=True)

FEATURES = ["rms", "peak", "p2p", "kurtosis", "skewness", "crest_factor"]
classes = ["Normal", "Inner race", "Ball", "Outer race"]

models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=0),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10)),
}

# 2. Leave-one-load-out: train on 3 loads, test on the unseen 4th
rows = []
all_true, all_pred = {m: [] for m in models}, {m: [] for m in models}
for test_hp in sorted(data["load"].unique()):
    train = data[data["load"] != test_hp]
    test = data[data["load"] == test_hp]
    for name, model in models.items():
        model.fit(train[FEATURES], train["label"])
        pred = model.predict(test[FEATURES])
        rows.append({
            "test_load_HP": test_hp,
            "model": name,
            "accuracy": accuracy_score(test["label"], pred),
            "macro_f1": f1_score(test["label"], pred, average="macro"),
        })
        all_true[name] += list(test["label"])
        all_pred[name] += list(pred)

res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))
print()
print(res.groupby("model")[["accuracy", "macro_f1"]].mean().round(3))

results = ROOT / "results"
results.mkdir(exist_ok=True)
res.to_csv(results / "03_baseline_results.csv", index=False)

# 3. Confusion matrix for the best model (pooled over the 4 folds)
best = res.groupby("model")["macro_f1"].mean().idxmax()
cm = confusion_matrix(all_true[best], all_pred[best], labels=classes)
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"{best}: leave-one-load-out (pooled)")
fig.tight_layout()
fig.savefig(results / "03_confusion_matrix.png", dpi=150)
plt.show()