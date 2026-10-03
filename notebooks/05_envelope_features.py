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
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import FILES_BY_SIZE, load_signal
from src.features import extract_all_features

N = 100  # windows per class per load
TIME = ["rms", "peak", "p2p", "kurtosis", "skewness", "crest_factor"]
ENV = ["env_bpfo", "env_bpfi", "env_bsf"]
SETS = {"time": TIME, "envelope": ENV, "time+envelope": TIME + ENV}


def build(size, part):
    frames = []
    for hp, files in FILES_BY_SIZE[size].items():
        for label, fname in files.items():
            sig = load_signal(fname)
            if label == "Normal":  # same file in both sizes: split it in halves
                half = len(sig) // 2
                sig = sig[:half] if part == "train" else sig[half:]
            df = extract_all_features(sig, hp)
            df = df.sample(n=min(N, len(df)), random_state=0)
            df["label"] = label
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=0),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10)),
    }


rows = []
for tr, te in [("0.007", "0.014"), ("0.014", "0.007")]:
    train, test = build(tr, "train"), build(te, "test")
    for set_name, cols in SETS.items():
        for name, model in make_models().items():
            model.fit(train[cols], train["label"])
            p = model.predict(test[cols])
            rows.append({
                "direction": f"{tr} -> {te}", "features": set_name, "model": name,
                "accuracy": accuracy_score(test["label"], p),
                "macro_f1": f1_score(test["label"], p, average="macro"),
            })

res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))

results = ROOT / "results"
results.mkdir(exist_ok=True)
res.to_csv(results / "05_envelope_results.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (direction, g) in zip(axes, res.groupby("direction")):
    g.pivot(index="features", columns="model", values="macro_f1").loc[list(SETS)].plot.bar(ax=ax, rot=0)
    ax.set_title(f"Train/test fault size: {direction} in")
    ax.set_ylabel("macro F1")
    ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(results / "05_feature_set_comparison.png", dpi=150)
plt.show()