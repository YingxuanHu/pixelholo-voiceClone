from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

RAW_VIDEOS_DIRNAME = "raw_videos"
PROCESSED_WAVS_DIRNAME = "processed_wavs"
METADATA_FILENAME = "metadata.csv"

DEFAULT_SAMPLE_RATE = 44100
TARGET_LUFS = -23.0

MIN_CHUNK_SEC = 2.0
MAX_CHUNK_SEC = 10.0
SILENCE_MIN_LEN_MS = 500
SILENCE_THRESH_DB = -40
KEEP_SILENCE_MS = 200

DEFAULT_MODEL_SIZE = "large-v3"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"

DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LEN = 500
DEFAULT_FP16 = True
DEFAULT_EPOCHS = 10

STYLE_TTS2_DIR = BASE_DIR / "lib" / "StyleTTS2"


def dataset_root(speaker_name: str) -> Path:
    return DATA_DIR / speaker_name


def raw_videos_dir(speaker_name: str) -> Path:
    return dataset_root(speaker_name) / RAW_VIDEOS_DIRNAME


def processed_wavs_dir(speaker_name: str) -> Path:
    return dataset_root(speaker_name) / PROCESSED_WAVS_DIRNAME


def metadata_path(speaker_name: str) -> Path:
    return dataset_root(speaker_name) / METADATA_FILENAME
