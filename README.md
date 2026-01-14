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
To make it automatic per project, add it to the venv activation script:
```bash
cat >> .venv/bin/activate <<'EOF'
VENV_SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$VENV_SITE_PACKAGES/nvidia/cudnn/lib:$VENV_SITE_PACKAGES/nvidia/cublas/lib"
EOF
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
# Defaults: denoise + VAD + stricter text filtering (language=en).
# Optional preprocessing tweaks:
# python src/preprocess.py --video /path/to/user.mp4 --name name1 --no-denoise
# python src/preprocess.py --video /path/to/user.mp4 --name name1 --min_words 2 --min_avg_logprob -1.0

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
# Recommended: auto-tune inference + select best epoch (all checkpoints) + build lexicon
# python src/train.py --dataset_path ./data/name1 \
#   --auto_tune_profile --tune_ref_wav ./data/name1/processed_wavs/name1_0001.wav \
#   --auto_select_epoch --select_ref_wav ./data/name1/processed_wavs/name1_0001.wav \
#   --auto_build_lexicon --lexicon_lang en-ca
# If you omit the reference wav flags, train.py auto-picks a clean, low-pitch, longer clip from processed_wavs.
# Thorough selection/tuning is now the default when auto-tuning/selecting.
# To force a faster (single-ref/single-text) pass:
# python src/train.py --dataset_path ./data/name1 \
#   --auto_tune_profile --tune_quick \
#   --auto_select_epoch --select_quick \
#   --auto_build_lexicon --lexicon_lang en-ca

# 3) One-step inference (no server)
python src/speak.py --profile name1 --text "Hello, this is a test."

# 4) Optional: run the API
# export STYLE_TTS2_MODEL=/path/to/your/model.pth
# export STYLE_TTS2_CONFIG=/path/to/config_ft.yml
# export STYLE_TTS2_REF_WAV=/path/to/reference.wav
# uvicorn src.inference:app --reload
# curl -X POST "http://localhost:8000/generate" \
#   -H "Content-Type: application/json" \
#   -d '{"text":"Hello, I am now digital.","ref_wav_path":"/path/to/reference.wav"}'
```

## Flag reference (common)
### Preprocess (`src/preprocess.py`)
- `--video`: input video path.
- `--name`: profile name (writes into `data/<name>/`).
- `--language`: Whisper language (default `en`).
- `--denoise/--no-denoise`: enable/disable ffmpeg denoise pass.
- `--min_words`: drop segments with fewer words (default 4).
- `--min_speech_ratio`: drop clips with too much silence (default 0.6).
- `--min_avg_logprob`: drop low-confidence segments (default -0.5).
- `--max_no_speech_prob`: drop likely non-speech segments (default 0.4).
- `--merge_gap_sec`: merge adjacent segments within this gap.
- `--legacy_split`: use silence splitting before transcription.
- `--quiet`: reduce logs.
- Long segments are split and re-transcribed into smaller clips.

### Train (`src/train.py`)
- `--dataset_path`: profile dataset (e.g., `./data/name1`).
- `--output_dir`: override checkpoint output folder.
- `--epochs`, `--batch_size`, `--max_len`, `--grad_accum_steps`: training knobs.
- `--max_text_chars`, `--max_text_words`: drop overly long transcripts to avoid BERT limits.
- Early stop: training writes `epoch_stats.json` (val/dur/f0) and stops early when it hits the sweet-spot range or overfits. Adjust thresholds in `outputs/training/<profile>/config_ft.yml` under `early_stop`.
- `--auto_tune_profile`: writes `profile.json` (auto alpha/beta/steps/scale/f0).
- `--auto_select_epoch`: writes `best_epoch.txt` + `epoch_scores.json` (evaluates all epochs unless `--select_limit` is set).
- `--auto_build_lexicon`: writes `data/<name>/lexicon.json`.
- `--lexicon_lang`: language code for lexicon generation (defaults to `en-ca`).
- `--tune_ref_wav` / `--select_ref_wav`: reference wav for tuning/selection.
- `--tune_ref_dir` / `--select_ref_dir`: directory of reference wavs (prefers clean, low-pitch clips).
- `--tune_ref_count` / `--select_ref_count`: how many refs to evaluate (default 1).
- `--tune_probe_texts` / `--select_probe_texts`: text file with one prompt per line.
- `--tune_thorough` / `--select_thorough`: use multiple refs + multiple prompts (slower, more stable).
- `--tune_quick` / `--select_quick`: force single-ref/single-text quick pass.
- `--use_resemblyzer`: re-rank top-quality epochs by speaker similarity.
- `--quality_top_n`: number of top-quality epochs to compare (default 5).
- `--identity_margin`: prefer earlier epoch within this similarity margin.
- `--tune_target_f0_hz`: target median F0 for deeper/shallower voices.
- If no reference wav is provided, the script auto-picks one from `processed_wavs`.
- If both auto-select and auto-tune are enabled, auto-tune uses the chosen best epoch.

### Inference (`src/speak.py`)
- `--profile`: use `outputs/training/<profile>` + `data/<profile>`.
- `--text`: text to synthesize.
- `--ref_wav`: override reference wav.
- `--phonemizer_lang`: accent locale (e.g., `en-ca`, `en-us`, `en-gb`, `en-au`, `en-in`).
- `--lexicon_path`: per-word pronunciation overrides (defaults to `data/<profile>/lexicon.json` if present).
- `--max_chunk_chars` / `--max_chunk_words`: auto-split long text to avoid BERT token limits.
- `--pause_ms`: silence inserted between chunks.
- `--pitch_shift`: semitone shift post-process (negative = deeper voice).
- `--seed`: deterministic generation (default 1234); use `--no_seed` to disable.
API requests can also pass `max_chunk_chars`, `max_chunk_words`, and `pause_ms` to control chunking.

## Notes
- `src/train.py` patches a StyleTTS2 config and launches the finetune script.
- Training now defaults `spect_params.f_max` to 8000 Hz when missing, to reduce high-frequency sharpness.
- `src/inference.py` caches the model in memory for low latency.
- `src/auto_tune_profile.py` writes `profile.json` next to a checkpoint with tuned defaults.
- Auto-tuning adapts alpha/beta if pitch correlation is low or timbre similarity is too high.
- It also reduces embedding/diffusion when centroid_ratio is too bright (sharpness).
- If `profile.json` or `f0_scale.txt` exist next to a model, inference will use them automatically.
- `src/auto_select_epoch.py` scores checkpoints and writes `best_epoch.txt`.
- Selection prefers earlier epochs within a score margin (disable with `--no_prefer_earlier`).
- If `epoch_stats.json` exists, selection also penalizes epochs outside the sweet-spot loss ranges.
- Optional: add `--use_resemblyzer` to rank top-quality epochs by speaker similarity (requires `pip install resemblyzer`).
- Epoch scoring now also penalizes noisy outputs (flatness/ZCR) and low harmonicity.
- `src/auto_tune_profile.py` now also writes `f0_scale.txt` for downstream tools.
- `src/build_lexicon.py` generates `lexicon.json` from metadata.

## Accent overrides (optional)
- Set `phonemizer_lang` per request or in `profile.json` (examples: `en`, `en-gb`, `en-au`, `en-in`).
- Add a lexicon at `data/<profile>/lexicon.json` or pass `lexicon_path`:
  ```json
  {
    "water": "w ɔː t əɹ",
    "about": "əˈbæʊt"
  }
  ```
  Values should be espeak-style phonemes (same format produced by the phonemizer).
- Generate a default lexicon from your metadata:
  ```bash
  python src/build_lexicon.py --profile name1 --lang en-ca
  ```
