import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import loadmat

from src.features import FS, ORDERS, RPM_BY_LOAD, envelope_spectrum, extract_all_features, make_windows

APP = Path(__file__).resolve().parent
LOG = APP / "alerts_log.csv"

st.set_page_config(page_title="Bearing Fault Monitor", layout="wide")
st.title("Bearing Fault Monitor")
st.caption("Envelope-spectrum features + Random Forest, trained on CWRU data (demo).")


@st.cache_resource
def load_model():
    bundle = joblib.load(APP / "model.joblib")
    return bundle["model"], bundle["features"]


model, FEATS = load_model()

# ---- sidebar ----
st.sidebar.header("Input")
source = st.sidebar.radio("Signal source", ["Sample signal", "Upload CWRU .mat file"])
hp = st.sidebar.selectbox("Motor load (HP)", [0, 1, 2, 3], index=1)
threshold = st.sidebar.slider("Alert if share of faulty windows is above", 0.1, 0.9, 0.5, 0.05)

sig = None
if source == "Sample signal":
    z = np.load(APP / "samples.npz")
    choice = st.sidebar.selectbox("Sample", list(z.files))
    sig = z[choice].astype(float)
    st.sidebar.caption("Samples are 1 HP, 0.007 in faults. Set load = 1 for correct shaft speed.")
else:
    up = st.sidebar.file_uploader("CWRU .mat file (12k drive end)", type=["mat"])
    if up is not None:
        d = loadmat(up)
        keys = [k for k in d if k.endswith("DE_time")]
        if keys:
            sig = d[keys[0]].flatten()
        else:
            st.error("No *_DE_time variable found in this file.")

if sig is None:
    st.info("Choose a sample or upload a file to start.")
    st.stop()

# ---- diagnosis ----
sig = sig[:60000]  # keep the app fast
feats = extract_all_features(sig, hp)
pred = model.predict(feats[FEATS])
counts = pd.Series(pred).value_counts()
verdict = counts.idxmax()
fault_share = float((pred != "Normal").mean())

c1, c2, c3 = st.columns(3)
c1.metric("Most frequent diagnosis", verdict)
c2.metric("Windows analysed", len(pred))
c3.metric("Share of faulty windows", f"{fault_share:.0%}")

if fault_share > threshold:
    st.error(f"ALERT: fault suspected ({verdict}). {fault_share:.0%} of windows are faulty.")
    row = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"),
                         "diagnosis": verdict, "fault_share": round(fault_share, 3)}])
    row.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
else:
    st.success("No fault alert. Bearing looks normal.")

# ---- plots ----
fr = RPM_BY_LOAD[hp] / 60
w = make_windows(sig, 4096, 0.75)[0]
freqs, spec = envelope_spectrum(w)
m = freqs <= 600

left, right = st.columns(2)
with left:
    st.subheader("Vibration (first 4096 samples)")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(len(w)) / FS, w, linewidth=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accel (g)")
    fig.tight_layout()
    st.pyplot(fig)
with right:
    st.subheader("Envelope spectrum")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(freqs[m], spec[m], linewidth=0.8)
    for name, order in ORDERS.items():
        ax.axvline(order * fr, linestyle="--", linewidth=0.9, label=name.upper(), color={"bpfo": "r", "bpfi": "g", "bsf": "orange"}[name])
    ax.set_xlabel("Frequency (Hz)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

st.subheader("Window predictions")
st.bar_chart(counts)

if LOG.exists():
    with st.expander("Alert log"):
        st.dataframe(pd.read_csv(LOG).tail(20))