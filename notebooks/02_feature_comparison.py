import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import FILES, load_signal
from src.features import extract_features

frames = []
for label, fname in FILES.items():
    df = extract_features(load_signal(fname))
    df["label"] = label
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
print(data.groupby("label").size().rename("n_windows"))
print()
print(data.groupby("label").mean().round(3))

results = ROOT / "results"
results.mkdir(exist_ok=True)
data.to_csv(results / "02_features.csv", index=False)

order = list(FILES.keys())
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, feat in zip(axes, ["rms", "kurtosis", "crest_factor"]):
    data.boxplot(column=feat, by="label", ax=ax)
    ax.set_title(feat)
    ax.set_xlabel("")
fig.suptitle("Time-domain features by bearing condition")
fig.tight_layout()
fig.savefig(results / "02_feature_boxplots.png", dpi=150)
plt.show()