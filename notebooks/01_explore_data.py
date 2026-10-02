import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import FILES, FS, load_signal

N = 2048  # first 2048 samples (~0.17 s)
t = np.arange(N) / FS

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, sharey=True)
for ax, (label, fname) in zip(axes, FILES.items()):
    sig = load_signal(fname)
    print(f"{label}: {len(sig)} samples, RMS = {np.sqrt(np.mean(sig**2)):.4f}")
    ax.plot(t, sig[:N], linewidth=0.8)
    ax.set_ylabel("Accel (g)")
    ax.set_title(label, loc="left", fontsize=10)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("CWRU bearing vibration (12k Drive End, 0 HP, 0.007 in fault)")
fig.tight_layout()

out = Path(__file__).resolve().parent.parent / "results" / "01_raw_signals.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
plt.show()