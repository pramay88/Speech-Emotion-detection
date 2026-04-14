"""
STEP 3 — Train Model
Run: python train_model.py

Architecture: CNN → LSTM → Dense (softmax)
  Input  : (batch, 128 frames, 262 features)
  CNN    : 2x [Conv1D + BatchNorm + MaxPool + Dropout]   → local pattern detection
  LSTM   : 128 units, Bidirectional                       → temporal dynamics
  Dense  : 64 units + Dropout(0.4)
  Output : 8 classes (softmax)

Expected performance on RAVDESS (with augmentation):
  Validation accuracy: ~70 to 80%  (state-of-art is ~85%)
  Training time      : ~15 to 25 min on CPU, ~5 min on GPU
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers

# ── Config 
FEATURE_DIR = "data/features"
MODEL_DIR   = "models"
EPOCHS      = 50
BATCH_SIZE  = 32
LEARNING_RATE = 1e-3

EMOTION_LABELS = sorted([
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised"
])

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data

def load_data():
    X = np.load(f"{FEATURE_DIR}/X.npy")
    y = np.load(f"{FEATURE_DIR}/y.npy")
    print(f"[✓] Loaded X: {X.shape}, y: {y.shape}")

    # Normalize each feature column across the dataset
    N, T, F = X.shape
    X_flat = X.reshape(-1, F)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(N, T, F)

    # Save scaler for inference
    import pickle
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return X, y


# ── Model definition ──────────────────────────────────────────────────────────

def build_model(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)   # (T, F)

    # ── CNN blocks for local feature extraction
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    # ── Bidirectional LSTM for temporal dynamics
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.4)(x)

    # ── Dense head
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ── Training

def train():
    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}  Val: {X_val.shape}")

    model = build_model(input_shape=(X.shape[1], X.shape[2]),
                        n_classes=len(EMOTION_LABELS))
    model.summary()

    cbs = [
        callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                monitor="val_accuracy"),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=4,
                                    monitor="val_loss", verbose=1),
        callbacks.ModelCheckpoint(f"{MODEL_DIR}/best_model.keras",
                                  save_best_only=True, monitor="val_accuracy"),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=1
    )

    # ── Evaluation
    model.load_weights(f"{MODEL_DIR}/best_model.keras")
    y_pred = np.argmax(model.predict(X_val), axis=1)

    print("\n── Classification Report ──────────────────────────────────")
    print(classification_report(y_val, y_pred, target_names=EMOTION_LABELS))

    # ── Plots
    plot_history(history)
    plot_confusion_matrix(y_val, y_pred)
    print(f"\n[✓] Model saved to {MODEL_DIR}/best_model.keras")


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].set_xlabel("Epoch")

    axes[1].plot(history.history["accuracy"], label="train acc")
    axes[1].plot(history.history["val_accuracy"], label="val acc")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/training_history.png", dpi=120)
    print(f"[✓] Training plot saved to {MODEL_DIR}/training_history.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=ax)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion matrix — validation set")
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/confusion_matrix.png", dpi=120)
    print(f"[✓] Confusion matrix saved to {MODEL_DIR}/confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    train()
