import argparse
import json
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from inference import StyleTTS2RepoEngine


def _find_latest_checkpoint(training_dir: Path) -> Path | None:
    checkpoints = sorted(training_dir.glob("epoch_2nd_*.pth"))
    return checkpoints[-1] if checkpoints else None


def _find_best_checkpoint(training_dir: Path) -> Path | None:
    best_path = training_dir / "best_epoch.txt"
    if best_path.exists():
        content = best_path.read_text().strip()
        if content:
            candidate = Path(content)
            if candidate.exists():
                return candidate
    return None


def _pick_reference_wav(profile_dir: Path) -> Path | None:
    wavs = sorted((profile_dir / "processed_wavs").glob("*.wav"))
    if not wavs:
        return None
    try:
        import librosa
        import numpy as np
    except Exception:
        return wavs[0]

    best: tuple[float, Path] | None = None
    for path in wavs:
        try:
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio, _ = librosa.effects.trim(audio, top_db=35)
            if len(audio) < sr:
                continue
            rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
            if rms < 0.01:
                continue
            f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
            f0 = f0[np.isfinite(f0)]
            if f0.size < 5:
                continue
            f0_median = float(np.median(f0))
            f0_std = float(np.std(f0))
            flat = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
            score = f0_median + (flat * 300.0) + (f0_std / 50.0) - (rms * 20.0)
            if best is None or score < best[0]:
                best = (score, path)
        except Exception:
            continue

    return best[1] if best else wavs[0]


def _load_profile_defaults(base_dir: Path) -> dict:
    for filename in ("profile.json", "inference_defaults.json"):
        candidate = base_dir / filename
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text())
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return {}


def _load_lexicon(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"lexicon_path not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Lexicon must be a JSON object.")
    cleaned = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str):
            cleaned[key.lower()] = value.strip()
    return cleaned or None


def _load_f0_scale(model_path: Path) -> float | None:
    candidate = model_path.parent / "f0_scale.txt"
    if candidate.exists():
        try:
            return float(candidate.read_text().strip())
        except ValueError:
            return None
    return None


def _split_text(text: str, max_chars: int, max_words: int) -> list[str]:
    if not text:
        return []
    sentences = [s.strip() for s in re.findall(r"[^.!?]+[.!?]?", text) if s.strip()]
    chunks: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars and len(sentence.split()) <= max_words:
            chunks.append(sentence)
            continue
        words = sentence.split()
        current: list[str] = []
        for word in words:
            current.append(word)
            if len(" ".join(current)) >= max_chars or len(current) >= max_words:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
    return chunks


def _apply_pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    if semitones == 0:
        return audio
    if shutil.which("rubberband"):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = Path(tmpdir) / "in.wav"
                out_path = Path(tmpdir) / "out.wav"
                sf.write(in_path, audio, sample_rate)
                result = subprocess.run(
                    ["rubberband", "-p", str(semitones), str(in_path), str(out_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or "rubberband failed")
                shifted, out_sr = sf.read(out_path)
                if shifted.ndim > 1:
                    shifted = shifted.mean(axis=1)
                if out_sr != sample_rate:
                    try:
                        import librosa
                    except Exception as exc:
                        raise RuntimeError("Missing librosa for pitch resampling.") from exc
                    shifted = librosa.resample(shifted, orig_sr=out_sr, target_sr=sample_rate)
                return np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass
    try:
        import librosa
    except Exception as exc:
        raise RuntimeError("Missing librosa for pitch shifting.") from exc
    shifted = librosa.effects.pitch_shift(audio.astype(np.float32), sr=sample_rate, n_steps=semitones)
    return np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0)


def _apply_de_esser(audio: np.ndarray, sample_rate: int, cutoff_hz: float, order: int) -> np.ndarray:
    if cutoff_hz <= 0:
        return audio
    nyquist = sample_rate * 0.5
    if cutoff_hz >= nyquist:
        return audio
    try:
        from scipy.signal import butter, sosfilt
    except Exception as exc:
        raise RuntimeError("Missing scipy for de-esser filtering.") from exc
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    filtered = sosfilt(sos, audio.astype(np.float32))
    return np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)


def _smart_vad_trim(
    audio: np.ndarray,
    sample_rate: int,
    top_db: float = 30.0,
    frame_length: int = 1024,
    hop_length: int = 256,
    pad_ms: float = 50.0,
) -> np.ndarray:
    try:
        import librosa
    except Exception:
        return audio
    if audio.size == 0:
        return audio
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.power_to_db(rms * rms, ref=np.max)
    non_silent = np.flatnonzero(rms_db > -top_db)
    if non_silent.size == 0:
        return audio
    start = librosa.frames_to_samples(non_silent[0], hop_length=hop_length)
    end = librosa.frames_to_samples(non_silent[-1], hop_length=hop_length)
    pad = int(sample_rate * (pad_ms / 1000.0))
    start = max(0, start - pad)
    end = min(audio.size, end + pad)
    return audio[start:end]


def _soft_clip(audio: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    if audio.size == 0:
        return audio
    audio = audio.astype(np.float32, copy=False)
    max_val = float(np.max(np.abs(audio)))
    if max_val <= threshold:
        return audio
    return np.tanh(audio / threshold) * threshold


def _remove_dc(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio
    audio = audio.astype(np.float32, copy=False)
    return audio - float(np.mean(audio))


def _apply_crossfade(
    chunks: list[np.ndarray],
    sample_rate: int,
    crossfade_ms: float,
    fade_edges_ms: float = 8.0,
) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)
    cross_len = int(sample_rate * (crossfade_ms / 1000.0))
    fade_len = int(sample_rate * (fade_edges_ms / 1000.0))

    def _edge_fade(audio: np.ndarray) -> np.ndarray:
        if fade_len <= 1:
            return audio
        effective_len = fade_len
        if effective_len * 2 > audio.size:
            effective_len = max(1, audio.size // 2)
        audio = audio.astype(np.float32, copy=False)
        fade_in = np.linspace(0.0, 1.0, effective_len, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, effective_len, dtype=np.float32)
        audio[:effective_len] *= fade_in
        audio[-effective_len:] *= fade_out
        return audio

    if cross_len < 2:
        return np.concatenate([_edge_fade(c) for c in chunks])

    faded = _edge_fade(chunks[0].astype(np.float32, copy=False))
    t = np.linspace(0.0, 1.0, cross_len, dtype=np.float32)
    fade_in = np.sin(t * (np.pi / 2.0))
    fade_out = np.cos(t * (np.pi / 2.0))
    for nxt in chunks[1:]:
        nxt = _edge_fade(nxt.astype(np.float32, copy=False))
        if faded.size < cross_len or nxt.size < cross_len:
            faded = np.concatenate([faded, nxt])
            continue
        tail = faded[-cross_len:] * fade_out
        head = nxt[:cross_len] * fade_in
        blended = tail + head
        faded = np.concatenate([faded[:-cross_len], blended, nxt[cross_len:]])
    return faded


def _fade_edges(audio: np.ndarray, sample_rate: int, fade_ms: float = 5.0) -> np.ndarray:
    if audio.size == 0:
        return audio
    fade_len = int(sample_rate * (fade_ms / 1000.0))
    if fade_len <= 1:
        return audio
    if fade_len * 2 > audio.size:
        fade_len = max(1, audio.size // 2)
    audio = audio.astype(np.float32, copy=False)
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out
    return audio


def main() -> None:
    parser = argparse.ArgumentParser(description="One-step local inference (no server).")
    parser.add_argument("--profile", help="Profile name (uses outputs/training/<profile>).")
    parser.add_argument("--model_path", type=Path, help="Override model checkpoint path.")
    parser.add_argument("--config_path", type=Path, help="Override config path.")
    parser.add_argument("--ref_wav", type=Path, help="Reference wav path.")
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--out", type=Path, help="Output wav file.")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--diffusion_steps", type=int)
    parser.add_argument("--embedding_scale", type=float)
    parser.add_argument("--f0_scale", type=float)
    parser.add_argument("--phonemizer_lang", type=str)
    parser.add_argument("--lexicon_path", type=Path)
    parser.add_argument(
        "--max_chunk_chars",
        type=int,
        default=None,
        help="Split long text into chunks of this many characters.",
    )
    parser.add_argument(
        "--max_chunk_words",
        type=int,
        default=None,
        help="Split long text into chunks of this many words.",
    )
    parser.add_argument(
        "--pause_ms",
        type=int,
        default=None,
        help="Silence between chunks (ms).",
    )
    parser.add_argument(
        "--crossfade_ms",
        type=float,
        default=8.0,
        help="Crossfade between chunks to avoid clicks (ms).",
    )
    parser.add_argument(
        "--pitch_shift",
        type=float,
        default=None,
        help="Pitch shift in semitones (negative for deeper voice).",
    )
    parser.add_argument(
        "--de_esser_cutoff",
        type=float,
        default=None,
        help="Apply low-pass de-esser at this cutoff Hz (0 disables).",
    )
    parser.add_argument(
        "--de_esser_order",
        type=int,
        default=None,
        help="Filter order for de-esser (higher = stronger).",
    )
    parser.add_argument(
        "--pad_text",
        action="store_true",
        help="Wrap each chunk with a pause token to push artifacts into silence.",
    )
    parser.add_argument(
        "--pad_text_token",
        type=str,
        default=None,
        help="Token used for pad_text (default: '...').",
    )
    parser.add_argument(
        "--smart_trim_db",
        type=float,
        default=None,
        help="Trim chunk using RMS VAD (dB); 0 disables.",
    )
    parser.add_argument(
        "--smart_trim_pad_ms",
        type=float,
        default=None,
        help="Padding added around VAD trim (ms).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument("--no_seed", action="store_true", help="Disable deterministic seeding.")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    training_dir = None
    profile_dir = None

    if args.profile:
        training_dir = project_root / "outputs" / "training" / args.profile
        profile_dir = project_root / "data" / args.profile
    elif args.model_path:
        training_dir = args.model_path.parent

    defaults = {}
    if training_dir is not None:
        defaults = _load_profile_defaults(training_dir)
        if args.profile and not defaults:
            raise FileNotFoundError(
                f"profile.json not found in {training_dir}. Run auto_tune_profile to generate it."
            )

    model_path = args.model_path
    if model_path is None:
        candidate = defaults.get("model_path")
        if candidate:
            model_path = Path(candidate)
    if model_path is None and training_dir is not None:
        model_path = _find_best_checkpoint(training_dir) or _find_latest_checkpoint(training_dir)

    if model_path is None or not model_path.exists():
        raise FileNotFoundError("Model checkpoint not found. Provide --model_path or --profile.")

    config_path = args.config_path
    if config_path is None:
        candidate = defaults.get("config_path")
        if candidate:
            config_path = Path(candidate)
    if config_path is None:
        candidate = model_path.parent / "config_ft.yml"
        if candidate.exists():
            config_path = candidate

    if config_path is None or not config_path.exists():
        raise FileNotFoundError("Config not found. Provide --config_path or ensure config_ft.yml exists.")

    if not defaults:
        defaults = _load_profile_defaults(model_path.parent)
    ref_wav = args.ref_wav
    if ref_wav is None:
        candidate = defaults.get("ref_wav_path")
        if candidate:
            ref_wav = Path(candidate)
    if ref_wav is None and profile_dir is not None:
        ref_wav = _pick_reference_wav(profile_dir)

    if ref_wav is None or not ref_wav.exists():
        raise FileNotFoundError("Reference wav not found. Provide --ref_wav.")
    f0_scale = args.f0_scale
    if f0_scale is None:
        f0_scale = defaults.get("f0_scale")
    if f0_scale is None:
        f0_scale = _load_f0_scale(model_path) or 1.0

    alpha = args.alpha if args.alpha is not None else defaults.get("alpha", 0.1)
    beta = args.beta if args.beta is not None else defaults.get("beta", 0.1)
    diffusion_steps = (
        args.diffusion_steps
        if args.diffusion_steps is not None
        else defaults.get("diffusion_steps", 30)
    )
    embedding_scale = (
        args.embedding_scale
        if args.embedding_scale is not None
        else defaults.get("embedding_scale", 1.5)
    )
    phonemizer_lang = (
        args.phonemizer_lang
        if args.phonemizer_lang is not None
        else defaults.get("phonemizer_lang")
    )
    lexicon_path = (
        args.lexicon_path
        if args.lexicon_path is not None
        else defaults.get("lexicon_path")
    )
    if lexicon_path is None and profile_dir is not None:
        candidate = profile_dir / "lexicon.json"
        if candidate.exists():
            lexicon_path = candidate
    lexicon = _load_lexicon(Path(lexicon_path)) if lexicon_path else None

    max_chunk_chars = args.max_chunk_chars
    if max_chunk_chars is None:
        max_chunk_chars = defaults.get("max_chunk_chars", 180)
    max_chunk_words = args.max_chunk_words
    if max_chunk_words is None:
        max_chunk_words = defaults.get("max_chunk_words", 45)
    pause_ms = args.pause_ms
    if pause_ms is None:
        pause_ms = defaults.get("pause_ms", 180)
    crossfade_ms = args.crossfade_ms
    if crossfade_ms is None:
        crossfade_ms = defaults.get("crossfade_ms", 8.0)
    pitch_shift = args.pitch_shift
    if pitch_shift is None:
        pitch_shift = defaults.get("pitch_shift", 0.0)
    de_esser_cutoff = args.de_esser_cutoff
    if de_esser_cutoff is None:
        de_esser_cutoff = defaults.get("de_esser_cutoff", 0.0)
    de_esser_order = args.de_esser_order
    if de_esser_order is None:
        de_esser_order = defaults.get("de_esser_order", 2)
    pad_text = args.pad_text
    if not pad_text:
        pad_text = bool(defaults.get("pad_text", True))
    pad_text_token = args.pad_text_token
    if pad_text_token is None:
        pad_text_token = defaults.get("pad_text_token", "...")
    smart_trim_db = args.smart_trim_db
    if smart_trim_db is None:
        smart_trim_db = defaults.get("smart_trim_db", 30.0)
    smart_trim_pad_ms = args.smart_trim_pad_ms
    if smart_trim_pad_ms is None:
        smart_trim_pad_ms = defaults.get("smart_trim_pad_ms", 50.0)

    out_path = args.out
    if out_path is None:
        base_out_dir = project_root / "outputs" / "inference"
        profile_out_dir = base_out_dir / (args.profile or "manual")
        profile_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = profile_out_dir / f"{model_path.stem}_speak.wav"

    if not args.no_seed and args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    infer_start = time.perf_counter()
    engine = StyleTTS2RepoEngine(model_path=model_path, config_path=config_path)
    chunks = _split_text(args.text, max_chunk_chars, max_chunk_words)
    if not chunks:
        raise ValueError("No text to synthesize.")

    audio_parts: list[np.ndarray] = []
    pause = np.zeros(int(engine.sample_rate * (pause_ms / 1000.0)), dtype=np.float32)
    chunk_seed = None if args.no_seed else args.seed
    for idx, chunk in enumerate(chunks):
        if pad_text:
            chunk = f"{pad_text_token} {chunk} {pad_text_token}"
        audio = engine.generate(
            chunk,
            ref_wav_path=ref_wav,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
            f0_scale=f0_scale,
            phonemizer_lang=phonemizer_lang,
            lexicon=lexicon,
            seed=chunk_seed,
        )
        if smart_trim_db and smart_trim_db > 0:
            audio = _smart_vad_trim(
                audio,
                engine.sample_rate,
                top_db=float(smart_trim_db),
                pad_ms=float(smart_trim_pad_ms),
            )
        audio = _remove_dc(audio.astype(np.float32, copy=False))
        audio = _fade_edges(audio, engine.sample_rate, fade_ms=5.0)
        audio_parts.append(audio)
        if idx < len(chunks) - 1:
            audio_parts.append(pause)

    audio = _apply_crossfade(audio_parts, engine.sample_rate, crossfade_ms, fade_edges_ms=8.0)
    if pitch_shift:
        audio = _apply_pitch_shift(audio, engine.sample_rate, pitch_shift)
    if de_esser_cutoff:
        audio = _apply_de_esser(audio, engine.sample_rate, de_esser_cutoff, int(de_esser_order))
    audio = _soft_clip(audio)

    sf.write(out_path, audio, engine.sample_rate)
    infer_elapsed = time.perf_counter() - infer_start
    print(f"Saved: {out_path}")
    print(f"Inference time: {infer_elapsed:.2f}s")


if __name__ == "__main__":
    main()
