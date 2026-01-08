import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from pydub import AudioSegment
from pydub.silence import split_on_silence
from faster_whisper import WhisperModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import (  # noqa: E402
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_DENOISE,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CLIP_DBFS,
    DEFAULT_MAX_NO_SPEECH_PROB,
    DEFAULT_MERGE_GAP_SEC,
    DEFAULT_MODEL_SIZE,
    DEFAULT_MIN_AVG_LOGPROB,
    DEFAULT_MIN_CHUNK_DBFS,
    DEFAULT_MIN_WORDS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VAD_FILTER,
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


def _run_ffmpeg_extract(video_path: Path, wav_path: Path, denoise: bool) -> None:
    filters = []
    if denoise:
        filters.extend(["highpass=f=80", "lowpass=f=12000", "afftdn"])
    filters.append(f"loudnorm=I={TARGET_LUFS}:TP=-2:LRA=7")
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
        ",".join(filters),
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


def _segment_text_is_valid(
    segment,
    min_avg_logprob: float,
    max_no_speech_prob: float,
    min_words: int,
) -> str | None:
    text = _sanitize_text(segment.text)
    if not text:
        return None
    avg_logprob = getattr(segment, "avg_logprob", None)
    if avg_logprob is not None and avg_logprob < min_avg_logprob:
        return None
    no_speech_prob = getattr(segment, "no_speech_prob", None)
    if no_speech_prob is not None and no_speech_prob > max_no_speech_prob:
        return None
    if min_words and len(text.split()) < min_words:
        return None
    return text


def _merge_segments(
    segments: list[dict[str, float | str]],
    merge_gap_sec: float,
    max_len_sec: float,
) -> list[dict[str, float | str]]:
    merged: list[dict[str, float | str]] = []
    current: dict[str, float | str] | None = None

    for seg in segments:
        if current is None:
            current = dict(seg)
            continue
        gap = float(seg["start"]) - float(current["end"])
        merged_len = float(seg["end"]) - float(current["start"])
        if gap <= merge_gap_sec and merged_len <= max_len_sec:
            current["end"] = seg["end"]
            current["text"] = f"{current['text']} {seg['text']}"
        else:
            merged.append(current)
            current = dict(seg)
    if current is not None:
        merged.append(current)
    return merged


def _transcribe_full_audio(
    model: WhisperModel,
    wav_path: Path,
    language: str,
    vad_filter: bool,
    min_avg_logprob: float,
    max_no_speech_prob: float,
    min_words: int,
    merge_gap_sec: float,
    min_len_sec: float,
    max_len_sec: float,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, float | str]]:
    segments, _info = model.transcribe(
        str(wav_path),
        language=language,
        vad_filter=vad_filter,
    )

    collected: list[dict[str, float | str]] = []
    for segment in segments:
        text = _segment_text_is_valid(segment, min_avg_logprob, max_no_speech_prob, min_words)
        if not text:
            continue
        if segment.end <= segment.start:
            continue
        collected.append({"start": segment.start, "end": segment.end, "text": text})

    collected.sort(key=lambda item: float(item["start"]))
    merged = _merge_segments(collected, merge_gap_sec, max_len_sec)

    filtered: list[dict[str, float | str]] = []
    for seg in merged:
        duration = float(seg["end"]) - float(seg["start"])
        if duration < min_len_sec:
            continue
        if duration > max_len_sec:
            continue
        filtered.append(seg)

    if log:
        log(
            f"Segments: raw={len(collected)} merged={len(merged)} kept={len(filtered)} "
            f"(min_len={min_len_sec}s max_len={max_len_sec}s)"
        )

    return filtered


def _chunk_is_usable(
    chunk: AudioSegment,
    min_chunk_dbfs: float | None,
    max_clip_dbfs: float | None,
) -> bool:
    if min_chunk_dbfs is not None:
        if chunk.dBFS == float("-inf") or chunk.dBFS < min_chunk_dbfs:
            return False
    if max_clip_dbfs is not None and chunk.max_dBFS > max_clip_dbfs:
        return False
    return True


def process_video(
    video_path: Path,
    speaker_name: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    language: str = DEFAULT_LANGUAGE,
    vad_filter: bool = DEFAULT_VAD_FILTER,
    min_avg_logprob: float = DEFAULT_MIN_AVG_LOGPROB,
    max_no_speech_prob: float = DEFAULT_MAX_NO_SPEECH_PROB,
    min_words: int = DEFAULT_MIN_WORDS,
    merge_gap_sec: float = DEFAULT_MERGE_GAP_SEC,
    denoise: bool = DEFAULT_DENOISE,
    min_chunk_dbfs: float | None = DEFAULT_MIN_CHUNK_DBFS,
    max_clip_dbfs: float | None = DEFAULT_MAX_CLIP_DBFS,
    legacy_split: bool = False,
    quiet: bool = False,
) -> Path:
    def _log(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    dataset_root = raw_videos_dir(speaker_name).parent
    raw_dir = raw_videos_dir(speaker_name)
    wavs_dir = processed_wavs_dir(speaker_name)
    meta_path = metadata_path(speaker_name)

    raw_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir.mkdir(parents=True, exist_ok=True)

    copied_video = raw_dir / video_path.name
    if video_path.resolve() != copied_video.resolve():
        shutil.copy2(video_path, copied_video)

    _log(f"Extracting audio from {copied_video.name} (denoise={denoise})...")
    extracted_wav = dataset_root / f"{speaker_name}_full.wav"
    _run_ffmpeg_extract(copied_video, extracted_wav, denoise=denoise)

    audio = AudioSegment.from_wav(extracted_wav)
    if audio.frame_rate != DEFAULT_SAMPLE_RATE:
        audio = audio.set_frame_rate(DEFAULT_SAMPLE_RATE)
    if audio.channels != 1:
        audio = audio.set_channels(1)
    _log(
        f"Loaded audio: {audio.duration_seconds:.1f}s, {audio.frame_rate}Hz, "
        f"{audio.channels}ch, dBFS={audio.dBFS:.1f}"
    )

    min_len_ms = int(MIN_CHUNK_SEC * 1000)
    max_len_ms = int(MAX_CHUNK_SEC * 1000)
    min_len_sec = MIN_CHUNK_SEC
    max_len_sec = MAX_CHUNK_SEC

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    with meta_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        chunk_index = 1
        if not legacy_split:
            _log(
                f"Transcribing full audio (lang={language}, vad={vad_filter}, "
                f"min_words={min_words}, min_avg_logprob={min_avg_logprob})..."
            )
            segments = _transcribe_full_audio(
                model=model,
                wav_path=extracted_wav,
                language=language,
                vad_filter=vad_filter,
                min_avg_logprob=min_avg_logprob,
                max_no_speech_prob=max_no_speech_prob,
                min_words=min_words,
                merge_gap_sec=merge_gap_sec,
                min_len_sec=min_len_sec,
                max_len_sec=max_len_sec,
                log=_log,
            )
        else:
            segments = []

        if segments:
            _log(f"Exporting {len(segments)} segments...")
            for seg in segments:
                start_ms = int(float(seg["start"]) * 1000)
                end_ms = int(float(seg["end"]) * 1000)
                if end_ms <= start_ms:
                    continue
                chunk = audio[start_ms:end_ms]
                if not _chunk_is_usable(chunk, min_chunk_dbfs, max_clip_dbfs):
                    continue
                chunk_name = f"{speaker_name}_{chunk_index:04d}.wav"
                chunk_path = wavs_dir / chunk_name
                chunk.export(chunk_path, format="wav")
                writer.writerow([chunk_name, seg["text"], speaker_name])
                if chunk_index == 1 or chunk_index % 10 == 0 or chunk_index == len(segments):
                    _log(f"Wrote {chunk_name} ({chunk_index}/{len(segments)})")
                chunk_index += 1
        else:
            _log("Whisper segmentation yielded no usable segments. Falling back to silence split...")
            silence_thresh = (
                audio.dBFS + SILENCE_THRESH_DB if audio.dBFS != float("-inf") else SILENCE_THRESH_DB
            )
            chunks = split_on_silence(
                audio,
                min_silence_len=SILENCE_MIN_LEN_MS,
                silence_thresh=silence_thresh,
                keep_silence=KEEP_SILENCE_MS,
            )

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

            _log(f"Transcribing {len(filtered_chunks)} chunks...")
            for chunk in filtered_chunks:
                if not _chunk_is_usable(chunk, min_chunk_dbfs, max_clip_dbfs):
                    continue
                chunk_name = f"{speaker_name}_{chunk_index:04d}.wav"
                chunk_path = wavs_dir / chunk_name
                chunk.export(chunk_path, format="wav")

                segs, _info = model.transcribe(
                    str(chunk_path),
                    language=language,
                    vad_filter=vad_filter,
                )
                text_parts = []
                for segment in segs:
                    text = _segment_text_is_valid(
                        segment, min_avg_logprob, max_no_speech_prob, min_words
                    )
                    if text:
                        text_parts.append(text)
                text = _sanitize_text(" ".join(text_parts))
                if not text:
                    chunk_path.unlink(missing_ok=True)
                    continue

                writer.writerow([chunk_name, text, speaker_name])
                if chunk_index == 1 or chunk_index % 10 == 0 or chunk_index == len(filtered_chunks):
                    _log(f"Wrote {chunk_name} ({chunk_index}/{len(filtered_chunks)})")
                chunk_index += 1

        if chunk_index == 1:
            raise RuntimeError("No valid audio chunks found after filtering.")

    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a video into StyleTTS2-ready chunks.")
    parser.add_argument("--video", required=True, type=Path, help="Path to input video (.mp4)")
    parser.add_argument("--name", required=True, help="Speaker name")
    parser.add_argument("--model_size", default=DEFAULT_MODEL_SIZE, help="faster-whisper model size")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device for faster-whisper")
    parser.add_argument("--compute_type", default=DEFAULT_COMPUTE_TYPE, help="Compute type for faster-whisper")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Whisper language code")
    parser.add_argument("--no_vad", action="store_true", help="Disable whisper VAD filtering")
    parser.add_argument("--min_avg_logprob", type=float, default=DEFAULT_MIN_AVG_LOGPROB)
    parser.add_argument("--max_no_speech_prob", type=float, default=DEFAULT_MAX_NO_SPEECH_PROB)
    parser.add_argument("--min_words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--merge_gap_sec", type=float, default=DEFAULT_MERGE_GAP_SEC)
    parser.add_argument("--denoise", action="store_true", default=DEFAULT_DENOISE)
    parser.add_argument("--min_chunk_dbfs", type=float, default=DEFAULT_MIN_CHUNK_DBFS)
    parser.add_argument("--max_clip_dbfs", type=float, default=DEFAULT_MAX_CLIP_DBFS)
    parser.add_argument("--legacy_split", action="store_true", help="Use silence split before transcription")
    parser.add_argument("--quiet", action="store_true", help="Reduce preprocessing logs")

    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    meta_path = process_video(
        video_path=args.video,
        speaker_name=args.name,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        vad_filter=not args.no_vad,
        min_avg_logprob=args.min_avg_logprob,
        max_no_speech_prob=args.max_no_speech_prob,
        min_words=args.min_words,
        merge_gap_sec=args.merge_gap_sec,
        denoise=args.denoise,
        min_chunk_dbfs=args.min_chunk_dbfs,
        max_clip_dbfs=args.max_clip_dbfs,
        legacy_split=args.legacy_split,
        quiet=args.quiet,
    )
    print(f"Metadata written to {meta_path}")


if __name__ == "__main__":
    main()
