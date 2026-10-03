from pathlib import Path
from scipy.io import loadmat

FS = 12000  # sampling rate (Hz) for CWRU 12k drive end data

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# file numbers per load (HP): 0.007 in faults, 12k drive end
FILES_BY_LOAD = {
    0: {"Normal": "97.mat", "Inner race": "105.mat", "Ball": "118.mat", "Outer race": "130.mat"},
    1: {"Normal": "98.mat", "Inner race": "106.mat", "Ball": "119.mat", "Outer race": "131.mat"},
    2: {"Normal": "99.mat", "Inner race": "107.mat", "Ball": "120.mat", "Outer race": "132.mat"},
    3: {"Normal": "100.mat", "Inner race": "108.mat", "Ball": "121.mat", "Outer race": "133.mat"},
}

# kept for the earlier scripts (0 HP only)
FILES = FILES_BY_LOAD[0]


def load_signal(filename):
    """Load the drive-end accelerometer signal from a CWRU .mat file."""
    d = loadmat(DATA_DIR / filename)
    key = [k for k in d if k.endswith("DE_time")][0]
    return d[key].flatten()


# fault size (inch) -> load (HP) -> class -> file
FILES_BY_SIZE = {
    "0.007": FILES_BY_LOAD,
    "0.014": {
        0: {"Normal": "97.mat", "Inner race": "169.mat", "Ball": "185.mat", "Outer race": "197.mat"},
        1: {"Normal": "98.mat", "Inner race": "170.mat", "Ball": "186.mat", "Outer race": "198.mat"},
        2: {"Normal": "99.mat", "Inner race": "171.mat", "Ball": "187.mat", "Outer race": "199.mat"},
        3: {"Normal": "100.mat", "Inner race": "172.mat", "Ball": "188.mat", "Outer race": "200.mat"},
    },
}