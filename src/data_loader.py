from pathlib import Path
from scipy.io import loadmat

FS = 12000  # sampling rate (Hz) for CWRU 12k drive end data

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FILES = {
    "Normal": "97.mat",
    "Inner race": "105.mat",
    "Ball": "118.mat",
    "Outer race": "130.mat",
}


def load_signal(filename):
    """Load the drive-end accelerometer signal from a CWRU .mat file."""
    d = loadmat(DATA_DIR / filename)
    key = [k for k in d if k.endswith("DE_time")][0]
    return d[key].flatten()