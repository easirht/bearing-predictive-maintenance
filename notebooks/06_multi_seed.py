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

SEEDS = range(10)
N = 80  # windows per class per load, re-sampled for every seed
TIME = ["rms", "peak", "p2p", "kurtosis", "skewness", "crest_factor"]
ENV = ["env_bpfo", "env_bpfi", "env_bsf"]
SETS = {"time": TIME, "envelope": ENV, "time+envelope": TIME + ENV}
DIRECTIONS = [("0.007", "0.014"), ("0.014", "0.007")]

_cache = {}


def all_windows(size, part):
    """Extract features once per (size, part); subsampling happens later."""
    if (size, part) in _cache:
        return _cache[(size, part)]
    frames = []
    for hp, files in FILES_BY_SIZE[size].items():
        for label, fname in files.items():
            sig = load_signal(fname)
            if label == "Normal":  # same file in both sizes: split in halves
                half = len(sig) // 2
                sig = sig[:half] if part == "train" else sig[half:]
            df = extract_all_features(sig, hp)
            df["label"] = label
            df["load"] = hp
            frames.append(df)
    _cache[(size, part)] = pd.concat(frames, ignore_index=True)
    return _cache[(size, part)]


def subsample(df, seed):
    parts = [g.sample(n=min(N, len(g)), random_state=seed)
             for _, g in df.groupby(["label", "load"])]
    return pd.concat(parts, ignore_index=True)


def make_models(seed):
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10)),
    }


rows = []
for tr, te in DIRECTIONS:
    train_all, test_all = all_windows(tr, "train"), all_windows(te, "test")
    for seed in SEEDS:
        train, test = subsample(train_all, seed), subsample(test_all, seed)
        for set_name, cols in SETS.items():
            for name, model in make_models(seed).items():
                model.fit(train[cols], train["label"])
                p = model.predict(test[cols])
                rows.append({
                    "direction": f"{tr} -> {te}", "features": set_name,
                    "model": name, "seed": seed,
                    "accuracy": accuracy_score(test["label"], p),
                    "macro_f1": f1_score(test["label"], p, average="macro"),
                })
    print(f"done {tr} -> {te}")

res = pd.DataFrame(rows)
summ = (res.groupby(["direction", "features", "model"])[["accuracy", "macro_f1"]]
        .agg(["mean", "std"]).round(3))
print(summ.to_string())

results = ROOT / "results"
results.mkdir(exist_ok=True)
res.to_csv(results / "06_multi_seed_raw.csv", index=False)
summ.to_csv(results / "06_multi_seed_summary.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (direction, g) in zip(axes, res.groupby("direction")):
    stats = g.groupby(["features", "model"])["macro_f1"].agg(["mean", "std"]).reset_index()
    mean = stats.pivot(index="features", columns="model", values="mean").loc[list(SETS)]
    std = stats.pivot(index="features", columns="model", values="std").loc[list(SETS)]
    mean.plot.bar(ax=ax, yerr=std, rot=0, capsize=3)
    ax.set_title(f"Train -> test fault size: {direction} in")
    ax.set_ylabel("macro F1 (mean ± std, 10 seeds)")
    ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(results / "06_multi_seed_comparison.png", dpi=150)
plt.show()