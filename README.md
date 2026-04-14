# Speech Emotion Recognition — Hackathon Starter

Identify human emotions from speech signals using deep learning.
**Model**: CNN + Bidirectional LSTM | **Dataset**: RAVDESS | **Accuracy**: ~75–80%

---

## Quick Start (4-hour plan)

### Hour 1 — Setup & data
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download RAVDESS dataset (~200 MB)
python download_data.py
```

### Hour 1–2 — Feature extraction
```bash
# Extracts MFCC, Mel spectrogram, Chroma, ZCR from all audio files
# Output: data/features/X.npy  (N, 128, 262)
#         data/features/y.npy  (N,)
python feature_extraction.py
```

### Hour 2–3 — Train the model
```bash
# Trains CNN + BiLSTM for up to 50 epochs (early stopping)
# Output: models/best_model.keras
#         models/training_history.png
#         models/confusion_matrix.png
python train_model.py
```

### Hour 3–4 — Demo
```bash
# Test a single file
python predict.py path/to/audio.wav

# Launch Gradio web demo (works with microphone too!)
python app.py
```

---

## Architecture

```
Input  (batch, 128 frames, 262 features)
  │
  ├─ Conv1D(64, kernel=5) + BatchNorm + MaxPool + Dropout(0.3)
  ├─ Conv1D(128, kernel=3) + BatchNorm + MaxPool + Dropout(0.3)
  │
  ├─ Bidirectional LSTM(128, return_sequences=True)
  ├─ Bidirectional LSTM(64)
  ├─ Dropout(0.4)
  │
  ├─ Dense(64, relu) + Dropout(0.4)
  └─ Dense(8, softmax)
```

**Features extracted per audio file:**

| Feature        | Dimension | Purpose                          |
|----------------|-----------|----------------------------------|
| MFCC           | 40        | Vocal tract shape                |
| MFCC delta     | 40        | First-order temporal change      |
| MFCC delta²    | 40        | Second-order temporal change     |
| Mel spectrogram| 128       | Frequency-time representation    |
| Chroma         | 12        | Harmonic / pitch content         |
| ZCR            | 1         | Signal noisiness                 |
| RMS energy     | 1         | Loudness                         |
| **Total**      | **262**   |                                  |

---

## Dataset — RAVDESS

- 7,356 audio clips | 24 actors (12M / 12F)
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Filename convention: `03-01-05-01-02-01-12.wav`
  - Field 3 = emotion code (01=neutral … 08=surprised)

---

## Tips to improve accuracy

| Technique             | Expected gain |
|-----------------------|---------------|
| Increase EPOCHS to 80 | +1–2%         |
| Add pitch-shift augmentation | +2–3%  |
| Use pre-trained wav2vec2 embeddings | +5–10% |
| Ensemble with SVM on MFCC | +2–4%    |
| 5-fold cross validation | better estimate |

---

## Files

```
ser_hackathon/
├── requirements.txt        # dependencies
├── download_data.py        # download RAVDESS
├── feature_extraction.py   # audio → numpy arrays
├── train_model.py          # model definition + training
├── predict.py              # single-file inference
├── app.py                  # Gradio demo
└── README.md
```
