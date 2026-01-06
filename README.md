# Voice Clone 5090

Local StyleTTS2 fine-tuning + FastAPI inference pipeline.

## Prereqs
- System deps: `espeak-ng` and `ffmpeg` (phonemizer + audio tooling).
- GPU: NVIDIA RTX 5090 (recommended, but CUDA-capable GPU works).

## Optional venv (recommended)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout
- `data/` dataset outputs per speaker
- `models/` checkpoints and logs
- `src/` core scripts
- `config.py` shared settings

## Usage
```bash
# 1) Preprocess video into chunks + metadata.csv
python src/preprocess.py --video /path/to/user.mp4 --name name1

# 2) Fine-tune StyleTTS2
# Clone StyleTTS2 into ./lib/StyleTTS2 before training.
# Download the LibriTTS pretrained checkpoint + config into the repo:
# mkdir -p lib/StyleTTS2/Models/LibriTTS
# wget -O lib/StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth \
#   https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth
# wget -O lib/StyleTTS2/Models/LibriTTS/config.yml \
#   https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/config.yml
python src/train.py --dataset_path ./data/name1

# 3) Run the API (set default model path)
export STYLE_TTS2_MODEL=/path/to/your/model.pth
uvicorn src.inference:app --reload

# 4) Request synthesis
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, I am now digital."}'
```

## Notes
- `src/train.py` patches a StyleTTS2 config and launches the finetune script.
- `src/inference.py` caches the model in memory for low latency.
