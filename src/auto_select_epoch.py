import argparse
import json
import re
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
import yaml

from inference import StyleTTS2RepoEngine

_EPOCH_RE = re.compile(r"epoch_2nd_(\d+)", re.IGNORECASE)

DEFAULT_PROBE_TEXTS = [
    "Hello, this is a quick pitch calibration sample.",
    "The quick brown fox jumps over the lazy dog.",
    "We repaired the loose screw and the wobbly leg.",
    "Please call me tomorrow morning at nine.",
    "I can't make the meeting today, sorry about that.",
]


def _load_mono(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


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


def _trim(audio: np.ndarray) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(audio, top_db=35)
    return trimmed if trimmed.size else audio


def _median_f0(audio: np.ndarray, sr: int) -> float:
    f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
    f0 = f0[np.isfinite(f0)]
    if f0.size == 0:
        raise ValueError("Unable to estimate F0 from audio.")
    return float(np.median(f0))


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def _harmonic_ratio(audio: np.ndarray) -> float:
    if not audio.size:
        return 0.0
    harmonic = librosa.effects.harmonic(audio)
    return _rms(harmonic) / (_rms(audio) + 1e-6)


def _hf_ratio(audio: np.ndarray, sr: int, low_hz: float = 6000.0, high_hz: float = 10000.0) -> float:
    if not audio.size:
        return 0.0
    n_fft = 1024
    hop = 256
    spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total = spec[freqs >= 80.0].sum()
    if total <= 0:
        return 0.0
    band = spec[(freqs >= low_hz) & (freqs <= high_hz)].sum()
    return float(band / total)


def _mfcc_dtw_distance(ref_audio: np.ndarray, out_audio: np.ndarray, sr: int) -> float:
    mfcc_ref = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=13)
    mfcc_out = librosa.feature.mfcc(y=out_audio, sr=sr, n_mfcc=13)
    if mfcc_ref.size == 0 or mfcc_out.size == 0:
        return float("inf")
    _cost, wp = librosa.sequence.dtw(mfcc_ref.T, mfcc_out.T, metric="euclidean")
    path = np.asarray(wp)
    if path.size == 0:
        return float("inf")
    diffs = mfcc_ref[:, path[:, 0]] - mfcc_out[:, path[:, 1]]
    dist = np.linalg.norm(diffs, axis=0)
    return float(np.mean(dist))


def _f0_contour(audio: np.ndarray, sr: int) -> np.ndarray:
    f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
    f0 = f0[np.isfinite(f0)]
    return f0


def _f0_correlation(ref_f0: np.ndarray, out_f0: np.ndarray) -> float:
    if ref_f0.size < 5 or out_f0.size < 5:
        return 0.0
    size = min(ref_f0.size, out_f0.size)
    ref = np.log(ref_f0[:size] + 1e-6)
    out = np.log(out_f0[:size] + 1e-6)
    ref = (ref - ref.mean()) / (ref.std() + 1e-6)
    out = (out - out.mean()) / (out.std() + 1e-6)
    return float(np.mean(ref * out))


def _normalize_wer_text(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [w for w in words if w]


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = _normalize_wer_text(reference)
    hyp_words = _normalize_wer_text(hypothesis)
    if not ref_words:
        return 0.0
    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return float(dp[-1][-1]) / max(1, len(ref_words))


def _transcribe_audio(model, audio: np.ndarray, sr: int, language: str | None) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        sf.write(handle.name, audio, sr)
        segments, _info = model.transcribe(handle.name, language=language)
        text = " ".join(seg.text.strip() for seg in segments if seg.text)
        return text.strip()


def _metrics(ref_audio: np.ndarray, out_audio: np.ndarray, sr: int) -> dict[str, float]:
    ref_trim = _trim(ref_audio)
    out_trim = _trim(out_audio)

    min_len = min(len(ref_trim), len(out_trim))
    if min_len == 0:
        raise ValueError("Empty audio after trimming.")
    ref_trim = ref_trim[:min_len]
    out_trim = out_trim[:min_len]

    f0_ref = _median_f0(ref_trim, sr)
    f0_out = _median_f0(out_trim, sr)
    pitch_ratio = f0_out / f0_ref

    mfcc_ref = librosa.feature.mfcc(y=ref_trim, sr=sr, n_mfcc=13)
    mfcc_out = librosa.feature.mfcc(y=out_trim, sr=sr, n_mfcc=13)
    v_ref = np.mean(mfcc_ref, axis=1)
    v_out = np.mean(mfcc_out, axis=1)
    denom = (np.linalg.norm(v_ref) * np.linalg.norm(v_out)) or 1.0
    mfcc_cos = float(np.dot(v_ref, v_out) / denom)

    flat_ref = float(np.mean(librosa.feature.spectral_flatness(y=ref_trim)))
    flat_out = float(np.mean(librosa.feature.spectral_flatness(y=out_trim)))
    flat_diff = abs(flat_out - flat_ref)

    rms_ref = float(np.sqrt(np.mean(np.square(ref_trim))))
    rms_out = float(np.sqrt(np.mean(np.square(out_trim))))
    rms_ratio = rms_out / rms_ref if rms_ref > 0 else 1.0

    centroid_ref = float(np.mean(librosa.feature.spectral_centroid(y=ref_trim, sr=sr)))
    centroid_out = float(np.mean(librosa.feature.spectral_centroid(y=out_trim, sr=sr)))
    centroid_ratio = centroid_out / centroid_ref if centroid_ref > 0 else 1.0

    f0_corr = _f0_correlation(_f0_contour(ref_trim, sr), _f0_contour(out_trim, sr))
    mfcc_dtw = _mfcc_dtw_distance(ref_trim, out_trim, sr)
    flat_out = float(np.mean(librosa.feature.spectral_flatness(y=out_trim)))
    zcr_out = float(np.mean(librosa.feature.zero_crossing_rate(out_trim)))
    harm_ratio = _harmonic_ratio(out_trim)
    ref_dur = len(ref_trim) / sr
    out_dur = len(out_trim) / sr
    dur_ratio = out_dur / ref_dur if ref_dur > 0 else 1.0
    hf_ratio = _hf_ratio(out_trim, sr)

    return {
        "pitch_ratio": pitch_ratio,
        "mfcc_cos": mfcc_cos,
        "flat_diff": flat_diff,
        "rms_ratio": rms_ratio,
        "centroid_ratio": centroid_ratio,
        "f0_corr": f0_corr,
        "mfcc_dtw": mfcc_dtw,
        "flat_out": flat_out,
        "zcr_out": zcr_out,
        "harm_ratio": harm_ratio,
        "out_dur": out_dur,
        "dur_ratio": dur_ratio,
        "hf_ratio": hf_ratio,
    }


def _score(metrics: dict[str, float], wer_weight: float = 0.0) -> float:
    pitch_err = abs(np.log(metrics["pitch_ratio"])) if metrics["pitch_ratio"] > 0 else 1.0
    rms_err = abs(np.log(metrics["rms_ratio"])) if metrics["rms_ratio"] > 0 else 1.0
    centroid_err = abs(np.log(metrics["centroid_ratio"])) if metrics["centroid_ratio"] > 0 else 1.0
    mcd = metrics.get("mfcc_dtw", 0.0)
    flat_out = metrics.get("flat_out", 0.0)
    zcr_out = metrics.get("zcr_out", 0.0)
    harm_ratio = metrics.get("harm_ratio", 0.0)
    out_dur = metrics.get("out_dur", 0.0)
    dur_ratio = metrics.get("dur_ratio", 1.0)
    hf_ratio = metrics.get("hf_ratio", 0.0)
    dur_penalty = 0.0
    if out_dur < 0.5:
        dur_penalty = (0.5 - out_dur) * 4.0
    dur_ratio_penalty = 0.0
    if dur_ratio < 0.8:
        dur_ratio_penalty = (0.8 - dur_ratio) * 2.0
    elif dur_ratio > 1.25:
        dur_ratio_penalty = (dur_ratio - 1.25) * 2.0
    wer = metrics.get("wer")
    return (
        pitch_err
        + metrics["flat_diff"] * 2.0
        + rms_err * 0.2
        + centroid_err * 0.3
        + (mcd * 0.02)
        + (flat_out * 1.2)
        + (zcr_out * 0.3)
        + (1.0 - harm_ratio) * 0.6
        + dur_penalty
        + dur_ratio_penalty
        + (hf_ratio * 2.0)
        + ((wer * wer_weight) if isinstance(wer, (int, float)) else 0.0)
        - metrics["mfcc_cos"] * 0.1
        - metrics["f0_corr"] * 0.1
    )


def _stats_penalty(stats: dict[str, float]) -> float:
    def _range_penalty(value: float | None, low: float, high: float, weight: float, hard_low: float | None = None) -> float:
        if value is None or not np.isfinite(value):
            return 0.0
        penalty = 0.0
        if value < low:
            penalty += (low - value) * weight
        elif value > high:
            penalty += (value - high) * weight
        if hard_low is not None and value < hard_low:
            penalty += (hard_low - value) * weight * 2.5
        return penalty

    return (
        _range_penalty(stats.get("val_loss"), 0.45, 0.55, 2.0, hard_low=0.4)
        + _range_penalty(stats.get("dur_loss"), 1.4, 1.6, 1.2, hard_low=1.0)
        + _range_penalty(stats.get("f0_loss"), 1.0, 1.3, 1.2, hard_low=0.8)
    )


def _load_profile_defaults(training_dir: Path) -> dict:
    candidate = training_dir / "profile.json"
    if candidate.exists():
        try:
            data = json.loads(candidate.read_text())
        except json.JSONDecodeError:
            return {}
        if isinstance(data, dict):
            return data
    return {}


def _load_epoch_stats(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    text = path.read_text().strip()
    if not text:
        return {}
    entries = []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        entries = parsed
    elif isinstance(parsed, dict):
        entries = [parsed]
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    stats = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        try:
            stats[int(epoch)] = entry
        except (ValueError, TypeError):
            continue
    return stats


def _load_f0_scale(training_dir: Path, override: float | None) -> float:
    if override is not None:
        return override
    candidate = training_dir / "f0_scale.txt"
    if candidate.exists():
        try:
            return float(candidate.read_text().strip())
        except ValueError:
            return 1.0
    return 1.0


def _epoch_index(checkpoint: str) -> int | None:
    match = _EPOCH_RE.search(checkpoint)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _load_sample_rate(config_path: Path) -> int:
    try:
        config = yaml.safe_load(config_path.read_text())
    except Exception:
        return 24000
    if not isinstance(config, dict):
        return 24000
    preprocess = config.get("preprocess_params", {})
    if isinstance(preprocess, dict):
        return int(preprocess.get("sr", 24000))
    return 24000


def _load_texts(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"probe_texts not found: {path}")
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line]


def _rank_reference_wavs(ref_dir: Path) -> list[Path]:
    wavs = sorted(ref_dir.glob("*.wav"))
    if not wavs:
        return []
    scored: list[tuple[float, Path]] = []
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
            scored.append((score, path))
        except Exception:
            continue
    scored.sort(key=lambda item: item[0])
    return [path for _score, path in scored] or wavs


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick best sounding epoch by audio similarity.")
    parser.add_argument("--training_dir", type=Path, required=True)
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--ref_wav", type=Path, help="Primary reference wav.")
    parser.add_argument("--ref_dir", type=Path, help="Directory of reference wavs.")
    parser.add_argument("--ref_count", type=int, default=1, help="Number of references to use.")
    parser.add_argument(
        "--text",
        default="Hello, this is a quick pitch calibration sample.",
        help="Text prompt for the probe generation.",
    )
    parser.add_argument("--probe_texts", type=Path, help="Text file with one prompt per line.")
    parser.add_argument("--thorough", action="store_true", help="Use multiple refs + probe texts.")
    parser.add_argument("--limit", type=int, help="Only check last N checkpoints.")
    parser.add_argument("--f0_scale", type=float, help="Override f0_scale.")
    parser.add_argument("--alpha", type=float, help="Override alpha.")
    parser.add_argument("--beta", type=float, help="Override beta.")
    parser.add_argument("--diffusion_steps", type=int, help="Override diffusion steps.")
    parser.add_argument("--embedding_scale", type=float, help="Override embedding scale.")
    parser.add_argument("--save_best", action="store_true", help="Save best sample wav.")
    parser.add_argument("--use_resemblyzer", action="store_true", help="Include speaker similarity score.")
    parser.add_argument("--use_wer", action="store_true", help="Include transcription WER in scoring.")
    parser.add_argument("--wer_weight", type=float, default=0.8, help="Penalty weight for WER.")
    parser.add_argument("--wer_model_size", type=str, default="tiny.en", help="faster-whisper model size.")
    parser.add_argument("--wer_device", type=str, help="Device for WER transcription (cuda/cpu).")
    parser.add_argument(
        "--wer_compute_type",
        type=str,
        default="float16",
        help="Compute type for WER transcription.",
    )
    parser.add_argument("--wer_language", type=str, help="Language code for WER transcription.")
    parser.add_argument(
        "--quality_top_n",
        type=int,
        default=5,
        help="Only run speaker similarity on the top N quality checkpoints.",
    )
    parser.add_argument(
        "--identity_margin",
        type=float,
        default=0.02,
        help="Prefer earlier epochs within this identity margin of the best.",
    )
    parser.add_argument(
        "--score_margin",
        type=float,
        default=0.15,
        help="Pick earliest epoch within this score margin of the best.",
    )
    parser.add_argument(
        "--no_prefer_earlier",
        action="store_false",
        dest="prefer_earlier",
        help="Disable early-epoch preference.",
    )
    parser.add_argument(
        "--report_top",
        type=int,
        default=5,
        help="Print the top N checkpoints by score.",
    )
    parser.add_argument("--phonemizer_lang", type=str, help="Override phonemizer language.")
    parser.add_argument("--lexicon_path", type=Path, help="Lexicon JSON for phoneme overrides.")
    parser.add_argument(
        "--stats_min_epoch",
        type=int,
        default=0,
        help="Ignore epoch_stats entries below this epoch.",
    )
    parser.add_argument(
        "--overfit_floor_factor",
        type=float,
        default=0.8,
        help="Reject epochs with val_loss below median * factor.",
    )
    parser.set_defaults(prefer_earlier=True)

    args = parser.parse_args()

    training_dir = args.training_dir.resolve()
    config_path = args.config_path.resolve()
    ref_wav = args.ref_wav.resolve() if args.ref_wav else None

    checkpoints = sorted(training_dir.glob("epoch_2nd_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {training_dir}")
    if args.limit and args.limit > 0:
        checkpoints = checkpoints[-args.limit :]

    profile = _load_profile_defaults(training_dir)
    alpha = args.alpha if args.alpha is not None else profile.get("alpha", 0.1)
    beta = args.beta if args.beta is not None else profile.get("beta", 0.1)
    diffusion_steps = (
        args.diffusion_steps if args.diffusion_steps is not None else profile.get("diffusion_steps", 10)
    )
    embedding_scale = (
        args.embedding_scale if args.embedding_scale is not None else profile.get("embedding_scale", 1.5)
    )
    if args.f0_scale is not None:
        f0_scale = args.f0_scale
    else:
        f0_scale = profile.get("f0_scale")
        if f0_scale is None:
            f0_scale = _load_f0_scale(training_dir, None)
    phonemizer_lang = args.phonemizer_lang or profile.get("phonemizer_lang")
    lexicon_path = args.lexicon_path or profile.get("lexicon_path")
    lexicon = _load_lexicon(Path(lexicon_path)) if lexicon_path else None

    speaker_encoder = None
    preprocess_wav = None
    ref_embed_cache: dict[str, np.ndarray] = {}
    if args.use_resemblyzer:
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav as _preprocess_wav
        except Exception as exc:
            raise RuntimeError("Resemblyzer not installed. Run `pip install resemblyzer`.") from exc
        speaker_encoder = VoiceEncoder()
        preprocess_wav = _preprocess_wav

    texts = _load_texts(args.probe_texts)
    if not texts:
        texts = DEFAULT_PROBE_TEXTS if args.thorough else [args.text]
    if args.text and args.text not in texts:
        texts.append(args.text)

    ref_wavs: list[Path] = []
    if ref_wav:
        ref_wavs.append(ref_wav)
    target_refs = max(args.ref_count, 3) if args.thorough else args.ref_count

    ref_dir = args.ref_dir
    if ref_dir is None:
        candidate = training_dir.parents[2] / "data" / training_dir.name / "processed_wavs"
        if candidate.exists():
            ref_dir = candidate
    if ref_dir:
        ranked = _rank_reference_wavs(ref_dir.resolve())
        for path in ranked:
            if path not in ref_wavs:
                ref_wavs.append(path)
            if len(ref_wavs) >= target_refs:
                break

    if not ref_wavs:
        raise FileNotFoundError("No reference wavs found. Provide --ref_wav or --ref_dir.")

    epoch_stats = _load_epoch_stats(training_dir / "epoch_stats.json")
    if args.stats_min_epoch > 0:
        epoch_stats = {k: v for k, v in epoch_stats.items() if k >= args.stats_min_epoch}
    if epoch_stats:
        val_losses = [
            v.get("val_loss")
            for v in epoch_stats.values()
            if isinstance(v.get("val_loss"), (int, float))
        ]
        val_losses = [v for v in val_losses if np.isfinite(v)]
        if val_losses:
            median_val = float(np.median(val_losses))
            overfit_floor = median_val * args.overfit_floor_factor
            allowed_epochs = {
                ep
                for ep, stats in epoch_stats.items()
                if isinstance(stats.get("val_loss"), (int, float))
                and np.isfinite(stats.get("val_loss"))
                and stats["val_loss"] >= overfit_floor
            }
            if allowed_epochs:
                filtered = [
                    ckpt
                    for ckpt in checkpoints
                    if (epoch := _epoch_index(str(ckpt))) is not None and epoch in allowed_epochs
                ]
                if filtered:
                    print(
                        f"Filtered checkpoints below overfit floor {overfit_floor:.3f} "
                        f"(kept {len(filtered)} of {len(checkpoints)})"
                    )
                    checkpoints = filtered
    results = []
    store_audio = args.save_best or args.use_resemblyzer
    best_audio_by_ckpt = {} if store_audio else None
    best_ref_by_ckpt = {} if store_audio else None
    sample_rate = _load_sample_rate(config_path)

    wer_model = None
    wer_language = args.wer_language
    if args.use_wer:
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise RuntimeError("faster-whisper not installed. Run `pip install faster-whisper`.") from exc
        wer_device = args.wer_device or ("cuda" if torch.cuda.is_available() else "cpu")
        wer_compute_type = args.wer_compute_type or ("float16" if wer_device == "cuda" else "int8")
        wer_model = WhisperModel(args.wer_model_size, device=wer_device, compute_type=wer_compute_type)

    print(f"Evaluating {len(checkpoints)} checkpoints...")
    print(f"alpha={alpha} beta={beta} diffusion={diffusion_steps} embedding={embedding_scale} f0_scale={f0_scale}")
    print(f"Using {len(ref_wavs)} reference wav(s) and {len(texts)} prompt(s).")

    total_ckpts = len(checkpoints)
    for idx, ckpt in enumerate(checkpoints, start=1):
        print(f"[{idx}/{total_ckpts}] Scoring {ckpt.name}...")
        engine = StyleTTS2RepoEngine(model_path=ckpt, config_path=config_path)
        all_metrics: dict[str, list[float]] = {
            "pitch_ratio": [],
            "mfcc_cos": [],
            "flat_diff": [],
            "rms_ratio": [],
            "centroid_ratio": [],
            "f0_corr": [],
            "mfcc_dtw": [],
            "flat_out": [],
            "zcr_out": [],
            "harm_ratio": [],
            "out_dur": [],
            "dur_ratio": [],
            "hf_ratio": [],
            "wer": [],
        }
        failures = 0
        best_sample = None
        best_sample_score = None
        best_sample_ref = None
        sample_scores: list[float] = []
        for ref_path in ref_wavs:
            ref_audio = _load_mono(ref_path, engine.sample_rate)
            for text in texts:
                try:
                    audio = engine.generate(
                        text,
                        ref_wav_path=ref_path,
                        alpha=alpha,
                        beta=beta,
                        diffusion_steps=diffusion_steps,
                        embedding_scale=embedding_scale,
                        f0_scale=f0_scale,
                        phonemizer_lang=phonemizer_lang,
                        lexicon=lexicon,
                    )
                    metrics = _metrics(ref_audio, audio, engine.sample_rate)
                    if wer_model is not None:
                        hypothesis = _transcribe_audio(wer_model, audio, engine.sample_rate, wer_language)
                        metrics["wer"] = _word_error_rate(text, hypothesis)
                    score = _score(metrics, wer_weight=args.wer_weight if wer_model is not None else 0.0)
                    for key, value in metrics.items():
                        if np.isfinite(value):
                            all_metrics[key].append(float(value))
                    if best_sample_score is None or score < best_sample_score:
                        best_sample_score = score
                        best_sample = audio
                        best_sample_ref = ref_path
                    sample_scores.append(score)
                except Exception:
                    failures += 1
                    continue

        summary = {}
        for key, values in all_metrics.items():
            summary[key] = float(np.median(values)) if values else 0.0
        epoch_index = _epoch_index(str(ckpt))
        stats = epoch_stats.get(epoch_index) if epoch_index is not None else None
        stats_penalty = _stats_penalty(stats) if stats else 0.0
        base_score = float(np.median(sample_scores)) if sample_scores else _score(summary)
        score = base_score + (failures * 0.5) + stats_penalty
        entry = {
            "checkpoint": str(ckpt),
            "score": score,
            "metrics": summary,
            "failures": failures,
            "epoch_index": epoch_index,
            "stats": stats or {},
            "stats_penalty": stats_penalty,
        }
        results.append(entry)
        if best_audio_by_ckpt is not None and best_sample is not None:
            best_audio_by_ckpt[str(ckpt)] = best_sample
            best_ref_by_ckpt[str(ckpt)] = best_sample_ref or ref_wavs[0]

        del engine
        torch.cuda.empty_cache()

    if not results:
        raise RuntimeError("No checkpoints evaluated.")

    results_sorted = sorted(results, key=lambda item: item["score"])
    best = results_sorted[0]
    if args.prefer_earlier and args.score_margin is not None and args.score_margin >= 0:
        cutoff = best["score"] + args.score_margin
        candidates = [item for item in results_sorted if item["score"] <= cutoff]
        if candidates:
            candidates.sort(
                key=lambda item: (
                    item["epoch_index"] is None,
                    item["epoch_index"] if item["epoch_index"] is not None else 0,
                )
            )
            best = candidates[0]

    if args.use_resemblyzer and speaker_encoder is not None and preprocess_wav is not None:
        top_n = max(1, args.quality_top_n)
        candidates = results_sorted[:top_n]
        for entry in candidates:
            ckpt = entry["checkpoint"]
            audio = best_audio_by_ckpt.get(ckpt) if best_audio_by_ckpt else None
            ref_path = best_ref_by_ckpt.get(ckpt) if best_ref_by_ckpt else None
            if audio is None or ref_path is None:
                entry["spk_sim"] = 0.0
                continue
            cache_key = str(ref_path)
            if cache_key in ref_embed_cache:
                ref_embed = ref_embed_cache[cache_key]
            else:
                ref_embed = speaker_encoder.embed_utterance(preprocess_wav(str(ref_path)))
                ref_embed_cache[cache_key] = ref_embed
            out_wav = preprocess_wav(audio.astype(np.float32), source_sr=sample_rate)
            out_embed = speaker_encoder.embed_utterance(out_wav)
            denom = (np.linalg.norm(ref_embed) * np.linalg.norm(out_embed)) or 1.0
            entry["spk_sim"] = float(np.dot(ref_embed, out_embed) / denom)

        candidates = [c for c in candidates if "spk_sim" in c]
        if candidates:
            best_sim = max(c["spk_sim"] for c in candidates)
            if args.prefer_earlier:
                margin = max(0.0, args.identity_margin)
                within = [c for c in candidates if c["spk_sim"] >= best_sim - margin]
                within.sort(
                    key=lambda item: (
                        item["epoch_index"] is None,
                        item["epoch_index"] if item["epoch_index"] is not None else 0,
                        item["score"],
                    )
                )
                best = within[0] if within else max(candidates, key=lambda c: c["spk_sim"])
            else:
                best = max(candidates, key=lambda c: c["spk_sim"])

    best_audio = None
    if best_audio_by_ckpt is not None:
        best_audio = best_audio_by_ckpt.get(best["checkpoint"])
        if best_audio is None:
            raise RuntimeError("No audio samples generated during epoch selection.")

    best_path = training_dir / "best_epoch.txt"
    best_path.write_text(best["checkpoint"] + "\n")

    results_path = training_dir / "epoch_scores.json"
    results_path.write_text(json.dumps(results, indent=2))

    print(f"Best checkpoint: {best['checkpoint']} (score {best['score']:.5f})")
    top_n = max(0, args.report_top)
    if top_n:
        print("Top checkpoints:")
        for entry in results_sorted[:top_n]:
            epoch_label = entry.get("epoch_index")
            epoch_display = f"{epoch_label:04d}" if isinstance(epoch_label, int) else "unknown"
            stats = entry.get("stats") or {}
            stats_str = ""
            if stats:
                val = stats.get("val_loss")
                dur = stats.get("dur_loss")
                f0 = stats.get("f0_loss")
                if (
                    isinstance(val, (int, float))
                    and isinstance(dur, (int, float))
                    and isinstance(f0, (int, float))
                    and np.isfinite(val)
                    and np.isfinite(dur)
                    and np.isfinite(f0)
                ):
                    stats_str = f"  val={val:.3f} dur={dur:.3f} f0={f0:.3f}"
            wer = entry.get("metrics", {}).get("wer")
            wer_str = f"  wer={wer:.3f}" if isinstance(wer, (int, float)) and np.isfinite(wer) else ""
            spk_sim = entry.get("spk_sim")
            if spk_sim is None:
                print(
                    f"  epoch {epoch_display}  score {entry['score']:.5f}{stats_str}{wer_str}  {entry['checkpoint']}"
                )
            else:
                print(
                    f"  epoch {epoch_display}  score {entry['score']:.5f}{stats_str}{wer_str}  spk {spk_sim:.4f}  "
                    f"{entry['checkpoint']}"
                )
    print(f"Wrote {best_path}")
    print(f"Wrote {results_path}")

    if args.save_best and best_audio is not None:
        out_dir = training_dir.parents[1] / "inference" / training_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(best['checkpoint']).stem}_best.wav"
        sf.write(out_path, best_audio, 24000)
        print(f"Saved best sample: {out_path}")


if __name__ == "__main__":
    main()
