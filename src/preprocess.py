import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import split_on_silence
from faster_whisper import WhisperModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import (  # noqa: E402
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_SIZE,
    DEFAULT_SAMPLE_RATE,
    KEEP_SILENCE_MS,
    MAX_CHUNK_SEC,
    MIN_CHUNK_SEC,
    SILENCE_MIN_LEN_MS,
    SILENCE_THRESH_DB,
    TARGET_LUFS,
    metadata_path,
    processed_wavs_dir,
    raw_videos_dir,
)


def _run_ffmpeg_extract(video_path: Path, wav_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        "-af",
        f"loudnorm=I={TARGET_LUFS}:TP=-2:LRA=7",
        str(wav_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")


def _split_to_max_len(segment: AudioSegment, max_len_ms: int) -> list[AudioSegment]:
    chunks = []
    start = 0
    while start < len(segment):
        end = min(start + max_len_ms, len(segment))
        chunks.append(segment[start:end])
        start = end
    return chunks


def _sanitize_text(text: str) -> str:
    text = " ".join(text.split())
    return text.replace("|", " ").strip()


def process_video(
    video_path: Path,
    speaker_name: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
) -> Path:
    dataset_root = raw_videos_dir(speaker_name).parent
    raw_dir = raw_videos_dir(speaker_name)
    wavs_dir = processed_wavs_dir(speaker_name)
    meta_path = metadata_path(speaker_name)

    raw_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir.mkdir(parents=True, exist_ok=True)

    copied_video = raw_dir / video_path.name
    if video_path.resolve() != copied_video.resolve():
        shutil.copy2(video_path, copied_video)

    extracted_wav = dataset_root / f"{speaker_name}_full.wav"
    _run_ffmpeg_extract(copied_video, extracted_wav)

    audio = AudioSegment.from_wav(extracted_wav)
    if audio.frame_rate != DEFAULT_SAMPLE_RATE:
        audio = audio.set_frame_rate(DEFAULT_SAMPLE_RATE)
    if audio.channels != 1:
        audio = audio.set_channels(1)

    silence_thresh = audio.dBFS + SILENCE_THRESH_DB if audio.dBFS != float("-inf") else SILENCE_THRESH_DB
    chunks = split_on_silence(
        audio,
        min_silence_len=SILENCE_MIN_LEN_MS,
        silence_thresh=silence_thresh,
        keep_silence=KEEP_SILENCE_MS,
    )

    min_len_ms = int(MIN_CHUNK_SEC * 1000)
    max_len_ms = int(MAX_CHUNK_SEC * 1000)

    filtered_chunks: list[AudioSegment] = []
    for chunk in chunks:
        if len(chunk) < min_len_ms:
            continue
        if len(chunk) > max_len_ms:
            for sub_chunk in _split_to_max_len(chunk, max_len_ms):
                if len(sub_chunk) >= min_len_ms:
                    filtered_chunks.append(sub_chunk)
        else:
            filtered_chunks.append(chunk)

    if not filtered_chunks:
        raise RuntimeError("No valid audio chunks found after silence splitting.")

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    with meta_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        for idx, chunk in enumerate(filtered_chunks, start=1):
            chunk_name = f"{speaker_name}_{idx:04d}.wav"
            chunk_path = wavs_dir / chunk_name
            chunk.export(chunk_path, format="wav")

            segments, _info = model.transcribe(str(chunk_path))
            text = " ".join(segment.text.strip() for segment in segments)
            text = _sanitize_text(text)
            if not text:
                chunk_path.unlink(missing_ok=True)
                continue

            writer.writerow([chunk_name, text, speaker_name])

    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a video into StyleTTS2-ready chunks.")
    parser.add_argument("--video", required=True, type=Path, help="Path to input video (.mp4)")
    parser.add_argument("--name", required=True, help="Speaker name")
    parser.add_argument("--model_size", default=DEFAULT_MODEL_SIZE, help="faster-whisper model size")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device for faster-whisper")
    parser.add_argument("--compute_type", default=DEFAULT_COMPUTE_TYPE, help="Compute type for faster-whisper")

    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    meta_path = process_video(
        video_path=args.video,
        speaker_name=args.name,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )
    print(f"Metadata written to {meta_path}")


if __name__ == "__main__":
    main()
