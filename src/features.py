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
from scipy.signal import butter, sosfiltfilt, hilbert

FS = 12000
RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
# SKF 6205-2RS fault orders (multiples of shaft frequency)
ORDERS = {"bpfo": 3.5848, "bpfi": 5.4152, "bsf": 4.7135}

_SOS = butter(4, [2000, 5500], btype="bandpass", fs=FS, output="sos")


def envelope_spectrum(w):
    """Band-pass around the resonance band, take the envelope, return its power spectrum."""
    x = sosfiltfilt(_SOS, w - np.mean(w))
    env = np.abs(hilbert(x))
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    freqs = np.fft.rfftfreq(len(env), 1 / FS)
    return freqs, spec


def envelope_features(w, fr, tol=0.03):
    """Relative envelope energy at BPFO / BPFI / BSF (1x and 2x), normalised by total
    envelope energy in 5-600 Hz, so the value does not depend on vibration amplitude."""
    freqs, spec = envelope_spectrum(w)
    total = spec[(freqs >= 5) & (freqs <= 600)].sum() + 1e-12
    out = {}
    for name, order in ORDERS.items():
        e = 0.0
        for h in (1, 2):
            f0 = order * fr * h
            m = (freqs >= f0 * (1 - tol)) & (freqs <= f0 * (1 + tol))
            e += spec[m].sum()
        out["env_" + name] = np.log10(e / total + 1e-6)
    return out


def extract_all_features(sig, hp, win=4096, overlap=0.75):
    """Time-domain + envelope features. hp = motor load, used to get shaft speed."""
    fr = RPM_BY_LOAD[hp] / 60
    rows = []
    for w in make_windows(sig, win, overlap):
        d = time_features(w)
        d.update(envelope_features(w, fr))
        rows.append(d)
    return pd.DataFrame(rows)