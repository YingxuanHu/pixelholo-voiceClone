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

## CUDA libs for faster-whisper (if using pip cuDNN/cuBLAS)
If `faster-whisper` fails to load cuDNN, point `LD_LIBRARY_PATH` at the pip-installed libs:
```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/nvidia/cudnn/lib:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/nvidia/cublas/lib
```

## Project layout
- `data/` dataset outputs per speaker
- `outputs/training/` checkpoints and logs
- `outputs/inference/` generated audio files
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
# If you hit CUDA OOM, start with smaller settings:
# python src/train.py --dataset_path ./data/name1 --batch_size 2 --max_len 200
# Accent overfit run (new output dir, higher epochs, lower batch):
# python src/train.py --dataset_path ./data/name1 --output_dir ./outputs/training/name1_accent --epochs 50 --batch_size 4 --max_len 500

# 3) Run the API (set default model path)
export STYLE_TTS2_MODEL=/path/to/your/model.pth
export STYLE_TTS2_CONFIG=/path/to/config_ft.yml
export STYLE_TTS2_REF_WAV=/path/to/reference.wav
uvicorn src.inference:app --reload

# 4) Request synthesis
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, I am now digital.","ref_wav_path":"/path/to/reference.wav"}'
```

## Notes
- `src/train.py` patches a StyleTTS2 config and launches the finetune script.
- `src/inference.py` caches the model in memory for low latency.
