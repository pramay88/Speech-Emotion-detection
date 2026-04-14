"""
STEP 4 — Inference
Usage:
    from predict import predict_emotion
    result = predict_emotion("path/to/audio.wav")
    print(result)   # → {'emotion': 'happy', 'confidence': 0.87, 'all_probs': {...}}

Or run standalone:
    python predict.py path/to/audio.wav
"""

import sys
import os
import pickle
import numpy as np
import librosa
import tensorflow as tf

# ── Config (must match feature_extraction.py) ─────────────────────────────────
SR         = 22050
MAX_LEN    = 128
HOP_LENGTH = 512
N_MFCC     = 40
N_MELS     = 128
MODEL_PATH  = "models/best_model.keras"
SCALER_PATH = "models/scaler.pkl"

EMOTION_LABELS = sorted([
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised"
])

# ── Load model once ────────────────────────────────────────────────────────────
_model  = None
_scaler = None

def _load_model():
    global _model, _scaler
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    if _scaler is None:
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)


# ── Feature extraction (mirrors feature_extraction.py) ────────────────────────
def _extract(y, sr):
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH).T
    mfcc_d = librosa.feature.delta(mfcc, axis=0)
    mfcc_d2= librosa.feature.delta(mfcc, order=2, axis=0)
    mel    = librosa.power_to_db(
                 librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                                hop_length=HOP_LENGTH), ref=np.max).T
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH).T
    zcr    = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH).T
    rms    = librosa.feature.rms(y=y, hop_length=HOP_LENGTH).T
    features = np.concatenate([mfcc, mfcc_d, mfcc_d2, mel, chroma, zcr, rms], axis=1)

    # Pad or truncate to MAX_LEN
    if features.shape[0] < MAX_LEN:
        pad = np.zeros((MAX_LEN - features.shape[0], features.shape[1]))
        features = np.vstack([features, pad])
    else:
        features = features[:MAX_LEN]
    return features


def predict_emotion(audio_path: str) -> dict:
    """
    Predict emotion from a .wav file.

    Returns:
        {
            'emotion'    : str,           # top predicted emotion
            'confidence' : float,         # 0.0–1.0
            'all_probs'  : dict[str, float]
        }
    """
    _load_model()

    y, sr = librosa.load(audio_path, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=20)

    feats = _extract(y, sr)                          # (MAX_LEN, F)
    N, F = feats.shape
    feats_flat = feats.reshape(-1, F)
    feats_flat = _scaler.transform(feats_flat)
    feats = feats_flat.reshape(1, N, F)              # (1, MAX_LEN, F)

    probs = _model.predict(feats, verbose=0)[0]      # (n_classes,)
    top_idx = int(np.argmax(probs))

    return {
        "emotion":    EMOTION_LABELS[top_idx],
        "confidence": float(probs[top_idx]),
        "all_probs":  {label: float(p) for label, p in zip(EMOTION_LABELS, probs)}
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio.wav>")
        sys.exit(1)

    path = sys.argv[1]
    result = predict_emotion(path)
    print(f"\nFile      : {path}")
    print(f"Emotion   : {result['emotion'].upper()}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print("\nAll probabilities:")
    for emo, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"  {emo:<12} {prob*100:5.1f}%  {bar}")
