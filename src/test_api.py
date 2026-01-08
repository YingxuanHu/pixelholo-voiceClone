import requests
import os

# --- CONFIGURATION ---
URL = "http://localhost:8000/generate"

# 1. The Text you want to speak
TEXT = "I am Alvin, This is a test. I am checking if the accent comes through clearly."

# 2. Your Trained Model (Select Epoch 30 or 40)
MODEL_PATH = "outputs/training/alvin1_accent_final/epoch_2nd_00014.pth"

# 3. The "Fuel" for the Accent (Your best wav file)
# This is crucial. The model copies the style from this specific file.
REF_WAV = "data/alvin1/processed_wavs/alvin1_0012.wav" 

# 4. The Boosters
payload = {
    "text": TEXT,
    "model_path": MODEL_PATH,
    "ref_wav_path": REF_WAV,
    
    # ACCENT CONTROLS
    "alpha": 0.05,  # Voice Timbre (Lower = more like you)
    "beta": 0.8,   # Accent/Rhythm (Lower = forced accent)
    "diffusion_steps": 30, # Quality (10 is fast, 20 is high quality)
    "embedding_scale": 1.2
}

print(f"Sending request to {URL}...")
try:
    response = requests.post(URL, json=payload)
    
    if response.status_code == 200:
        output_file = "output_test.wav"
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Success! Audio saved to: {output_file}")
    else:
        print(f"Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"Failed to connect: {e}")