"""
STEP 1 — Download RAVDESS dataset
Run: python download_data.py

RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
- 7356 audio files, 24 actors (12 male, 12 female)
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised

ALTERNATIVE (if download fails):
  pip install kaggle
  kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio
"""

import os
import zipfile
import requests
from tqdm import tqdm

RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
DATA_DIR = "data/ravdess"

def download_ravdess():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = "data/ravdess.zip"

    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
        print(f"[✓] Dataset already exists at {DATA_DIR}/")
        return

    print(f"Downloading RAVDESS dataset (~200MB)...")
    response = requests.get(RAVDESS_URL, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    os.remove(zip_path)
    print(f"[✓] Dataset ready at {DATA_DIR}/")

def verify_dataset():
    """Count audio files and summarise emotion distribution."""
    import glob
    files = glob.glob(f"{DATA_DIR}/**/*.wav", recursive=True)
    print(f"\n[✓] Found {len(files)} audio files")

    emotion_map = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry",   "06": "fearful", "07": "disgust", "08": "surprised"
    }
    counts = {}
    for f in files:
        label = os.path.basename(f).split("-")[2]
        emo = emotion_map.get(label, "unknown")
        counts[emo] = counts.get(emo, 0) + 1

    print("\nEmotion distribution:")
    for emo, n in sorted(counts.items()):
        bar = "█" * (n // 10)
        print(f"  {emo:<12} {n:>4}  {bar}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download_ravdess()
    verify_dataset()
