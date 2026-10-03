import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def make_windows(sig, win=2048, overlap=0.5):
    """Cut a 1-D signal into fixed-length windows with overlap."""
    step = int(win * (1 - overlap))
    n = (len(sig) - win) // step + 1
    return np.stack([sig[i * step:i * step + win] for i in range(n)])


def time_features(w):
    """Time-domain features of one window."""
    rms = np.sqrt(np.mean(w ** 2))
    peak = np.max(np.abs(w))
    return {
        "rms": rms,
        "peak": peak,
        "p2p": np.ptp(w),
        "kurtosis": kurtosis(w),
        "skewness": skew(w),
        "crest_factor": peak / rms,
    }


def extract_features(sig, win=2048, overlap=0.5):
    """Return a DataFrame with one row of features per window."""
    windows = make_windows(sig, win, overlap)
    return pd.DataFrame([time_features(w) for w in windows])