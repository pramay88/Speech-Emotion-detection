"""
STEP 2 — Feature Extraction
Run: python feature_extraction.py

Extracts audio features from RAVDESS .wav files and saves them as
numpy arrays ready for model training.

Features extracted per file:
  - MFCC            : 40 coefficients (captures vocal tract shape)
  - Mel spectrogram : 128 bands       (frequency-time representation)
  - Chroma          : 12 pitch classes (harmonic content)
  - ZCR             : 1 value         (signal noisiness proxy)
  - RMS energy      : 1 value         (loudness)

All features are padded/truncated to a fixed time length (max_len=128 frames)
so the model sees tensors of shape (128, 182) — time × features.
"""

import os
import glob
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# ── Config 
DATA_DIR   = "data/ravdess"
SAVE_DIR   = "data/features"
SR         = 22050   # sample rate
MAX_LEN    = 128     # time frames (≈ 2.7 seconds @ hop_length=512)
N_MFCC     = 40
N_MELS     = 128
N_CHROMA   = 12
HOP_LENGTH = 512

EMOTION_MAP = {
    "01": "neutral",  "02": "calm",     "03": "happy",    "04": "sad",
    "05": "angry",    "06": "fearful",  "07": "disgust",  "08": "surprised"
}
LABEL_TO_INT = {v: i for i, v in enumerate(sorted(EMOTION_MAP.values()))}

# ── Helpers ───────────────────────────────────────────────────────────────────

def pad_or_truncate(arr, max_len):
    """arr shape: (time, features)  →  (max_len, features)"""
    if arr.shape[0] < max_len:
        pad = np.zeros((max_len - arr.shape[0], arr.shape[1]))
        arr = np.vstack([arr, pad])
    else:
        arr = arr[:max_len]
    return arr


def augment(y, sr):
    """Return list of (possibly augmented) signals: original + 2 augmentations."""
    variants = [y]

    # Time stretch
    for rate in [0.8, 1.2]:
        variants.append(librosa.effects.time_stretch(y, rate=rate))

    # Pitch shift
    for n_steps in [-2, 2]:
        variants.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps))

    # Noise
    noise = 0.005 * np.random.randn(len(y))
    variants.append(y + noise)

    return variants


def extract_features(y, sr):
    """Return feature matrix of shape (time, features)."""
    # MFCC: (n_mfcc, time) → transpose → (time, n_mfcc)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  hop_length=HOP_LENGTH).T

    # Delta and delta-delta MFCC for temporal dynamics
    mfcc_d  = librosa.feature.delta(mfcc, axis=0)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2, axis=0)

    # Mel spectrogram: (n_mels, time) → transpose
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                          hop_length=HOP_LENGTH)
    mel = librosa.power_to_db(mel, ref=np.max).T

    # Chroma: (12, time) → transpose
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          hop_length=HOP_LENGTH).T

    # Zero-crossing rate: (1, time) → transpose
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH).T

    # RMS energy: (1, time) → transpose
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH).T

    # Concatenate along feature axis: time × (40 + 40 + 40 + 128 + 12 + 1 + 1) = time × 262
    # (trimmed to first n_mels block for balance — use full mfcc triplet + mel + chroma + zcr + rms)
    features = np.concatenate([mfcc, mfcc_d, mfcc_d2, mel, chroma, zcr, rms], axis=1)
    return features


def parse_label(filepath):
    """Extract emotion label from RAVDESS filename convention."""
    basename = os.path.basename(filepath)
    parts = basename.replace(".wav", "").split("-")
    emotion_code = parts[2]
    return EMOTION_MAP.get(emotion_code)


# ── Main 

def build_dataset(use_augmentation=True):
    os.makedirs(SAVE_DIR, exist_ok=True)

    all_files = glob.glob(f"{DATA_DIR}/**/*.wav", recursive=True)
    if not all_files:
        raise FileNotFoundError(
            f"No .wav files found in {DATA_DIR}. Run download_data.py first."
        )

    X, y, meta = [], [], []

    for fpath in tqdm(all_files, desc="Extracting features"):
        label = parse_label(fpath)
        if label is None:
            continue

        try:
            audio, sr = librosa.load(fpath, sr=SR)
            # Trim leading/trailing silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
        except Exception as e:
            print(f"  [!] Skipped {fpath}: {e}")
            continue

        signals = augment(audio, sr) if use_augmentation else [audio]

        for sig in signals:
            feats = extract_features(sig, sr)
            feats = pad_or_truncate(feats, MAX_LEN)
            X.append(feats)
            y.append(LABEL_TO_INT[label])
            meta.append({"file": fpath, "emotion": label})

    X = np.array(X, dtype=np.float32)   # (N, MAX_LEN, n_features)
    y = np.array(y, dtype=np.int32)

    np.save(f"{SAVE_DIR}/X.npy", X)
    np.save(f"{SAVE_DIR}/y.npy", y)
    pd.DataFrame(meta).to_csv(f"{SAVE_DIR}/meta.csv", index=False)

    print(f"\n[✓] Saved X: {X.shape}, y: {y.shape}")
    print(f"    Features per frame: {X.shape[2]}")
    print(f"    Label mapping: {LABEL_TO_INT}")
    return X, y


if __name__ == "__main__":
    build_dataset(use_augmentation=True)
