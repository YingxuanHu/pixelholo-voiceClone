from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

RAW_VIDEOS_DIRNAME = "raw_videos"
PROCESSED_WAVS_DIRNAME = "processed_wavs"
METADATA_FILENAME = "metadata.csv"

DEFAULT_SAMPLE_RATE = 24000
DEFAULT_F_MAX = 8000
TARGET_LUFS = -23.0

MIN_CHUNK_SEC = 2.0
MAX_CHUNK_SEC = 10.0
SILENCE_MIN_LEN_MS = 500
SILENCE_THRESH_DB = -40
KEEP_SILENCE_MS = 200

DEFAULT_MODEL_SIZE = "large-v3"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_LANGUAGE = "en"
DEFAULT_VAD_FILTER = True
DEFAULT_MIN_AVG_LOGPROB = -0.5
DEFAULT_MAX_NO_SPEECH_PROB = 0.4
DEFAULT_MIN_WORDS = 4
DEFAULT_MERGE_GAP_SEC = 0.2
DEFAULT_DENOISE = True
DEFAULT_MIN_CHUNK_DBFS = -35.0
DEFAULT_MAX_CLIP_DBFS = None
DEFAULT_MIN_SPEECH_RATIO = 0.6

DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_LEN = 400
DEFAULT_FP16 = True
DEFAULT_EPOCHS = 25

STYLE_TTS2_DIR = BASE_DIR / "lib" / "StyleTTS2"


def dataset_root(speaker_name: str) -> Path:
    return DATA_DIR / speaker_name


def raw_videos_dir(speaker_name: str) -> Path:
    return dataset_root(speaker_name) / RAW_VIDEOS_DIRNAME


def processed_wavs_dir(speaker_name: str) -> Path:
    return dataset_root(speaker_name) / PROCESSED_WAVS_DIRNAME


def metadata_path(speaker_name: str) -> Path:
    return dataset_root(speaker_name) / METADATA_FILENAME
