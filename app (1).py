# app.py — Streamlit PhysioNet single-record inference (AE + CNN, multi-lead pipeline)

import os
import time
import random
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import wfdb
from wfdb import processing
from scipy.signal import butter, filtfilt, resample_poly
from tensorflow import keras

# -------------------------
# CONFIG (edit if needed)
# -------------------------
CONFIG = {
    "WORKDIR": "./physionet_data",
    "MODELS_DIR": "./models",                         # put autoencoder.keras and cnn_2lead.keras here
    "AUTOENC_PATH": "./models/autoencoder.keras",
    "CLF_PATH": "./models/cnn_2lead.keras",
    "AE_THRESHOLD_PATH": "./models/ae_threshold.npy", # Path to save/load AE threshold

    # MUST match training:
    "TARGET_FS": 250,                 # sampling frequency used during training
    "BANDPASS": (0.5, 30.0),          # bandpass used during training
    "WIN_SEC": 2.0,                   # window size used during training
    "FRAME_LEN": int(2.0 * 250),      # WIN_SEC * TARGET_FS (kept for reference)

    "NUM_LEADS": 3,                   # number of leads used during training
    "PAD_VALUE": 0.0,                 # padding value used during training
    "LEAD_PRIORITY": ["MLII","V5","V1","V2","V3","V4","V6","I","II","III","aVR","aVL","aVF"],

    "SEED": 42,
    "EPS_STD": 1e-8,
}

random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])

# Ensure work directory and model directory exist
os.makedirs(CONFIG["WORKDIR"], exist_ok=True)
os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)

# -------------------------
# Data Loading and Preprocessing Functions
# -------------------------

# Safer bandpass() - adapted from notebook fix
def bandpass(sig, fs, lo, hi, order=4):
    nyq = fs / 2.0
    lo_n = max(1e-6, float(lo) / nyq)
    hi_n = min(0.999, float(hi) / nyq)
    if not (0 < lo_n < hi_n < 1):
        # fallback: clamp to a safe band
        lo_n, hi_n = 0.004, 0.499
    b, a = butter(order, [lo_n, hi_n], btype='band')
    return filtfilt(b, a, sig, method="gust")

# Preprocessing for a single lead - adapted from notebook fix
def preprocess_single_lead(x, fs_src, fs_tgt, bp):
    """
    Resample and bandpass filter a single lead signal.
    """
    # Resample
    from math import gcd
    up, dn = int(fs_tgt), int(fs_src)
    g = gcd(up, dn)
    up //= g
    dn //= g
    x_rs = resample_poly(x, up, dn)

    # Bandpass filter
    x_bp = bandpass(x_rs, fs_tgt, bp[0], bp[1])

    # Standardize (handle potential zero std)
    mean_x = np.mean(x_bp)
    std_x = np.std(x_bp)
    if std_x < CONFIG["EPS_STD"]:
        return (x_bp - mean_x).astype(np.float32)  # remove mean if std too small
    return ((x_bp - mean_x) / std_x).astype(np.float32)

# Helper to pick lead indices based on priority
def pick_lead_indices(sig_names, wanted, k):
    idx = []
    for name in wanted:
        if name in sig_names and len(idx) < k:
            idx.append(sig_names.index(name))
    # Fill up remaining indices if needed from available leads
    for i, _ in enumerate(sig_names):
        if len(idx) >= k:
            break
        if i not in idx:
            idx.append(i)
    return idx[:k]

# Load record segments - adapted from notebook, including padding fix
def load_record_segments_multi(path_noext, fs_tgt=250, win_sec=2.0, mode="binary"):
    """
    Loads a PhysioNet record, preprocesses selected leads, and segments it around
    annotations or detected R-peaks. Handles padding.

    Returns:
        (segments, labels)
        segments: (N, segment_len, num_leads), labels: (N,)
    """
    try:
        rec = wfdb.rdrecord(path_noext)
        fs_src = float(rec.fs)
        sig = rec.p_signal.astype(np.float64)
        sig_names = list(rec.sig_name)
    except Exception as e:
        st.error(f"Error reading record {os.path.basename(path_noext)}: {e}")
        return np.empty((0, int(win_sec * fs_tgt), CONFIG["NUM_LEADS"])), np.empty((0,))

    K = CONFIG["NUM_LEADS"]
    # Select leads based on priority, handling cases with fewer leads
    use_idx = pick_lead_indices(sig_names, CONFIG["LEAD_PRIORITY"], min(K, sig.shape[1]))

    if not use_idx:
        st.warning(f"Could not find any leads from priority list in record {os.path.basename(path_noext)}.")
        return np.empty((0, int(win_sec * fs_tgt), K)), np.empty((0,))

    # Preprocess chosen leads
    chosen = [preprocess_single_lead(sig[:, i], fs_src, fs_tgt, CONFIG["BANDPASS"]) for i in use_idx]

    # Stack preprocessed leads
    Xstack = np.stack(chosen, axis=0)   # (k, T_resampled)

    # Pad with zeros if fewer than K leads were available
    if len(use_idx) < K:
        pad_shape = (K - len(use_idx), Xstack.shape[1])
        pad = np.full(pad_shape, CONFIG["PAD_VALUE"], dtype=Xstack.dtype)
        Xstack = np.concatenate([Xstack, pad], axis=0)

    Xstack = Xstack.T  # (T, K)

    # Get annotations or detect R-peaks
    try:
        ann = wfdb.rdann(path_noext, 'atr')
        # Adjust annotation samples to target frequency
        annsamp = (np.asarray(ann.sample, dtype=int) * (fs_tgt / fs_src)).astype(int)
        anntype = list(ann.symbol)
    except Exception:
        # Use gqrs_detect on the first available lead for R-peak detection
        r_locs = processing.gqrs_detect(sig=Xstack[:, 0], fs=fs_tgt)
        annsamp = np.asarray(r_locs, dtype=int)
        anntype = ['N'] * len(annsamp)  # Assume normal if no labels

    half = int(win_sec * fs_tgt / 2)
    segment_len = 2 * half
    Ttot = Xstack.shape[0]

    Xs, ys = [], []
    # Extract segments around annotation/R-peak samples
    for s, sy in zip(annsamp, anntype):
        l = s - half
        r = s + half

        # Calculate required padding
        pad_l = max(0, -l)
        pad_r = max(0, r - Ttot)

        # Adjust start index based on padding
        l_padded = l + pad_l

        # Pad if necessary
        if pad_l > 0 or pad_r > 0:
            padded_Xstack = np.pad(Xstack, ((pad_l, pad_r), (0, 0)),
                                   mode='constant', constant_values=CONFIG["PAD_VALUE"])
            seg = padded_Xstack[l_padded: l_padded + segment_len, :]
        else:
            if l < 0 or r > Ttot:
                continue
            seg = Xstack[l:r, :]

        if seg.shape[0] != segment_len:
            continue
        if np.any(np.isnan(seg)) or np.any(np.isinf(seg)):
            continue

        Xs.append(seg.astype(np.float32))
        # Map symbol to label (assuming binary mode 0=Normal, 1=Abnormal)
        if mode == "binary":
            NORMAL_SYMS = set(['N','L','R','e','j'])
            PVC_SYMS = set(['V','E'])
            ys.append(0 if sy in NORMAL_SYMS else 1)
        else:
            ys.append(-1)  # placeholder for other modes

    if not Xs:
        return np.empty((0, segment_len, K)), np.empty((0,))

    return np.stack(Xs), np.asarray(ys, dtype=int)

# Helper function for mapping symbols (multi-class extension if needed)
NORMAL_SYMS = set(['N','L','R','e','j'])
PVC_SYMS    = set(['V','E'])
APB_SYMS    = set(['A','a','J','S'])
FUSION_SYMS = set(['F'])

def map_symbol(s, mode="binary"):
    if mode == "binary":
        return 0 if s in NORMAL_SYMS else 1
    if s in NORMAL_SYMS: return 0
    if s in PVC_SYMS:    return 1
    if s in APB_SYMS:    return 2
    if s in FUSION_SYMS: return 3
    return 4

# Helper to check if a channel is good
def is_channel_good(x, eps=1e-8):
    return np.all(np.isfinite(x)) and (np.std(x) > eps)

# Helper to find the pair with lowest correlation
def pair_with_lowest_corr(leads_2d):
    # leads_2d: (L, T)
    L, T = leads_2d.shape
    best = (0, 1, 2.0)  # Initialize with a high correlation value
    for i in range(L):
        for j in range(i + 1, L):
            xi, xj = leads_2d[i], leads_2d[j]
            if is_channel_good(xi) and is_channel_good(xj):
                r = np.corrcoef(xi, xj)[0, 1]
                a = abs(r) if np.isfinite(r) else 2.0
            else:
                a = 2.0
            if a < best[2]:
                best = (i, j, a)
    return best  # (i, j, |corr|)

# -------------------------
# Streamlit App
# -------------------------

st.set_page_config(page_title="ECG Anomaly Detection and Classification (AE + CNN)", page_icon="🫀", layout="centered")
st.title("ECG Anomaly Detection and Classification (AE + CNN)")
st.write("Test PhysioNet records using trained Autoencoder and CNN models.")

# Load Models — use Streamlit caching to avoid reloading models on every interaction
@st.cache_resource
def load_models(ae_path, clf_path):
    ae_model = None
    clf_model = None

    if os.path.exists(ae_path):
        try:
            ae_model = keras.models.load_model(ae_path)
            st.success(f"Autoencoder model loaded from {ae_path}")
        except Exception as e:
            st.error(f"Error loading Autoencoder: {e}")
    else:
        st.warning(f"Autoencoder model file not found at {ae_path}")

    if os.path.exists(clf_path):
        try:
            clf_model = keras.models.load_model(clf_path)
            st.success(f"CNN classifier model loaded from {clf_path}")
        except Exception as e:
            st.error(f"Error loading CNN classifier: {e}")
    else:
        st.warning(f"CNN classifier model file not found at {clf_path}")

    return ae_model, clf_model

# Load the models when the app starts
ae_model, clf_model = load_models(CONFIG["AUTOENC_PATH"], CONFIG["CLF_PATH"])

# Load AE Threshold — use Streamlit caching
@st.cache_data
def load_ae_threshold(threshold_path):
    if not os.path.exists(threshold_path):
        return None
    try:
        threshold = np.load(threshold_path)
        return float(threshold)
    except Exception as e:
        st.error(f"Error loading AE threshold: {e}")
        return None

ae_threshold = load_ae_threshold(CONFIG["AE_THRESHOLD_PATH"])

# User Input
st.sidebar.header("Settings")
record_id_input = st.sidebar.text_input("Enter PhysioNet Record ID (e.g., 100)", "100")
db_name = st.sidebar.selectbox("Database", ["mitdb", "incartdb"], index=0)

# Prediction
if st.button("Analyze Record"):
    if ae_model is None or clf_model is None:
        st.error("Models are not loaded. Put files in ./models (autoencoder.keras, cnn_2lead.keras).")
        st.stop()
    if ae_threshold is None:
        st.error("AE threshold not loaded (models/ae_threshold.npy).")
        st.stop()

    st.info(f"Analyzing record: {record_id_input}...")
    start_time = time.time()

    DATA_DIR = os.path.join(CONFIG["WORKDIR"], db_name)
    os.makedirs(DATA_DIR, exist_ok=True)
    record_path_noext = os.path.join(DATA_DIR, record_id_input)

    # 1. Download record if not exists
    try:
        if not os.path.exists(f"{record_path_noext}.hea"):
            st.write(f"Record {record_id_input} not found locally. Downloading from PhysioNet...")
            wfdb.dl_database(db_name, dl_dir=DATA_DIR, records=[record_id_input], raise_overwrite_error=False)
            st.success(f"Downloaded record {record_id_input}.")
        else:
            st.write(f"Record {record_id_input} found locally.")
    except Exception as e:
        st.error(f"Error downloading record {record_id_input}: {e}")
        st.stop()

    # 2. Load and preprocess segments (multi-lead with padding)
    X_record, y_record_true = load_record_segments_multi(
        record_path_noext,
        fs_tgt=CONFIG["TARGET_FS"],
        win_sec=CONFIG["WIN_SEC"],
        mode="binary"
    )

    if X_record.size == 0:
        st.warning(f"No valid segments found for record {record_id_input}.")
    else:
        st.write(f"Found {len(X_record)} segments. Performing predictions...")

        # Autoencoder prediction (anomaly detection)
        ae_rec = ae_model.predict(X_record, verbose=0)
        ae_err = np.mean((ae_rec - X_record) ** 2, axis=(1, 2))
        ae_pred_anomalies = (ae_err > ae_threshold).astype(int)
        ae_anomaly_rate = ae_pred_anomalies.mean()

        # CNN classifier prediction
        cnn_log = clf_model.predict(X_record, verbose=0)
        cnn_pred_classes = np.argmax(cnn_log, axis=1)
        cnn_abnormal_rate = (cnn_pred_classes == 1).mean()  # Assuming 1 is the abnormal class

        end_time = time.time()

        # 4. Display Results
        st.subheader("Analysis Results")
        st.write(f"Record ID: **{record_id_input}**")
        st.write(f"Total Segments Analyzed: **{len(X_record)}**")
        st.write(f"Analysis Time: **{end_time - start_time:.2f} seconds**")

        st.subheader("Autoencoder Anomaly Detection")
        st.write(f"Proportion of segments detected as anomalies (AE Error > {ae_threshold:.4f}): **{ae_anomaly_rate:.3f}**")

        st.subheader("CNN Classification")
        st.write(f"Proportion of segments classified as abnormal: **{cnn_abnormal_rate:.3f}**")

        # Display sample segments with predictions
        st.subheader("Sample Segment Predictions")
        num_plots = min(4, len(X_record))
        if num_plots > 0:
            segment_len = int(CONFIG["WIN_SEC"] * CONFIG["TARGET_FS"])
            t = np.linspace(-CONFIG["WIN_SEC"]/2, CONFIG["WIN_SEC"]/2, segment_len)
            sample_indices = np.random.choice(len(X_record), num_plots, replace=False)
            for j in sample_indices:
                fig, ax = plt.subplots(figsize=(10, 3))
                for ch in range(X_record.shape[2]):
                    ax.plot(t, X_record[j, :, ch] + 3 * ch)
                ax.set_title(f"Segment {j} | AE Err={ae_err[j]:.4f} (Anomaly: {ae_pred_anomalies[j]}) | CNN Pred Class: {cnn_pred_classes[j]}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude (offset per lead)")
                st.pyplot(fig)
                plt.close(fig)

# Sidebar instructions (kept as UI text, not code blocks)
st.sidebar.subheader("How to Run")
st.sidebar.write("1) pip install streamlit wfdb numpy scipy tensorflow")
st.sidebar.write("2) Put models in ./models: autoencoder.keras, cnn_2lead.keras, ae_threshold.npy")
st.sidebar.write("3) streamlit run app.py")

st.sidebar.subheader("Note on Trained Models")
st.sidebar.write(f"This app assumes {CONFIG['NUM_LEADS']} leads (padded if needed), fs={CONFIG['TARGET_FS']} Hz, window={CONFIG['WIN_SEC']} s.")
