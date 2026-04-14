"""
STEP 5 — Launch Demo
Run: python app.py

Opens a Gradio web UI where you can:
  • Upload a .wav file and see the predicted emotion
  • Record from your microphone live
  • See a bar chart of all emotion probabilities
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tempfile
import soundfile as sf
matplotlib.use("Agg")
import librosa

from predict import predict_emotion, EMOTION_LABELS

# ── Emotion → emoji ──────────────────────────────────────────────────────────
EMOJI = {
    "angry":     "😠", "calm":      "😌",
    "disgust":   "🤢", "fearful":   "😨",
    "happy":     "😄", "neutral":   "😐",
    "sad":       "😢", "surprised": "😲",
}

COLORS = {
    "angry": "#E24B4A", "calm": "#1D9E75", "disgust": "#BA7517",
    "fearful": "#D4537E", "happy": "#EF9F27", "neutral": "#888780",
    "sad": "#378ADD", "surprised": "#7F77DD"
}


def _to_audio_path(audio_input):
    """Normalize Gradio audio payload to a local .wav filepath."""
    if audio_input is None:
        return None

    if isinstance(audio_input, str):
        return audio_input

    if isinstance(audio_input, dict):
        path = audio_input.get("path")
        if path:
            return path

    if isinstance(audio_input, tuple) and len(audio_input) == 2:
        sr, data = audio_input
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        sf.write(tmp.name, data, sr)
        return tmp.name

    if isinstance(audio_input, np.ndarray):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        sf.write(tmp.name, audio_input, 22050)
        return tmp.name

    raise ValueError(f"Unsupported audio input type: {type(audio_input).__name__}")


def analyze(audio_input):
    print("DEBUG: analyze called")

    if audio_input is None:
        return "Upload a .wav file to start.", None

    try:
        audio_path = _to_audio_path(audio_input)
        print("DEBUG: audio path:", audio_path)

        result = predict_emotion(audio_path)
        print("DEBUG: result:", result)

        emotion    = result["emotion"]
        confidence = result["confidence"]
        all_probs  = result["all_probs"]

        label = (
            f"## {EMOJI[emotion]} {emotion.upper()}\n\n"
            f"**Confidence:** {confidence*100:.1f}%"
        )

        # --- chart ---
        fig, ax = plt.subplots(figsize=(6, 3.5))

        # Light background
        fig.patch.set_facecolor("#ffffff")   # outer background (white)
        ax.set_facecolor("#f7f7f7")          # slightly soft gray for plot area

        labels = sorted(all_probs.keys())
        values = [all_probs[l] * 100 for l in labels]
        bar_colors = [COLORS[l] for l in labels]

        ax.barh(labels, values, color=bar_colors)

        # Improve readability on light theme
        ax.set_title("Emotion probabilities", color="#333333")
        ax.tick_params(colors="#333333")

        # Optional: subtle grid
        ax.grid(axis="x", color="#dddddd", linestyle="--", linewidth=0.5)

        return label, fig

    except Exception as e:
        print("ERROR:", e)
        return f"Error: {e}", plt.figure()  # ✅ ALWAYS return 2 values
    


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Speech Emotion Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 Speech Emotion Recognition")
    gr.Markdown(
        "Upload a `.wav` file or record from your microphone. "
        "The model predicts one of **8 emotions**: "
        "angry, calm, disgust, fearful, happy, neutral, sad, surprised."
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                format="wav",
                label="Audio input"
            )
            analyze_btn = gr.Button("Analyze emotion", variant="primary")

        with gr.Column(scale=1):
            emotion_out = gr.Markdown(label="Result")
            chart_out   = gr.Plot(label="Probability breakdown")

    analyze_btn.click(
        fn=analyze,
        inputs=audio_input,
        outputs=[emotion_out, chart_out]
    )

    gr.Examples(
        examples=[],   # add local .wav paths here if desired
        inputs=audio_input
    )

if __name__ == "__main__":
    demo.launch(share=False)
