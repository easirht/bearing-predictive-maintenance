import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from src.data_loader import FILES_BY_SIZE, load_signal
from src.features import extract_features

N = 200  # windows per class per load
FEATURES = ["rms", "peak", "p2p", "kurtosis", "skewness", "crest_factor"]
classes = ["Normal", "Inner race", "Ball", "Outer race"]


def build(size, part):
    """part = 'train' or 'test'. Normal files are shared between sizes, so
    Normal uses the first half of the recording for train, second half for test."""
    frames = []
    for hp, files in FILES_BY_SIZE[size].items():
        for label, fname in files.items():
            sig = load_signal(fname)
            if label == "Normal":
                half = len(sig) // 2
                sig = sig[:half] if part == "train" else sig[half:]
            df = extract_features(sig)
            df = df.sample(n=min(N, len(df)), random_state=0)
            df["label"] = label
            df["load"] = hp
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=0),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10)),
}

rows, preds = [], {}
for tr, te in [("0.007", "0.014"), ("0.014", "0.007")]:
    train, test = build(tr, "train"), build(te, "test")
    for name, model in models.items():
        model.fit(train[FEATURES], train["label"])
        p = model.predict(test[FEATURES])
        rows.append({
            "train_size": tr, "test_size": te, "model": name,
            "accuracy": accuracy_score(test["label"], p),
            "macro_f1": f1_score(test["label"], p, average="macro"),
        })
        preds[(tr, te, name)] = (list(test["label"]), list(p))

res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))

results = ROOT / "results"
results.mkdir(exist_ok=True)
res.to_csv(results / "04_cross_size_results.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (tr, te) in zip(axes, [("0.007", "0.014"), ("0.014", "0.007")]):
    y, p = preds[(tr, te, "RandomForest")]
    ConfusionMatrixDisplay(confusion_matrix(y, p, labels=classes),
                           display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"RandomForest: train {tr} in -> test {te} in")
    ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(results / "04_cross_size_confusion.png", dpi=150)
plt.show()