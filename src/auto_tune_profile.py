import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from inference import StyleTTS2RepoEngine

DEFAULT_PROBE_TEXTS = [
    "Hello, this is a quick pitch calibration sample.",
    "The quick brown fox jumps over the lazy dog.",
    "We repaired the loose screw and the wobbly leg.",
    "Please call me tomorrow morning at nine.",
    "I can't make the meeting today, sorry about that.",
]


def _find_latest_checkpoint(training_dir: Path) -> Path | None:
    checkpoints = sorted(training_dir.glob("epoch_2nd_*.pth"))
    return checkpoints[-1] if checkpoints else None


def _pick_reference_wav(profile_dir: Path) -> Path | None:
    wavs = sorted((profile_dir / "processed_wavs").glob("*.wav"))
    return wavs[0] if wavs else None


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

    return {
        "pitch_ratio": pitch_ratio,
        "mfcc_cos": mfcc_cos,
        "flat_diff": flat_diff,
        "rms_ratio": rms_ratio,
        "centroid_ratio": centroid_ratio,
        "f0_corr": f0_corr,
    }


def _score(metrics: dict[str, float]) -> float:
    pitch_err = abs(np.log(metrics["pitch_ratio"])) if metrics["pitch_ratio"] > 0 else 1.0
    rms_err = abs(np.log(metrics["rms_ratio"])) if metrics["rms_ratio"] > 0 else 1.0
    centroid_err = abs(np.log(metrics["centroid_ratio"])) if metrics["centroid_ratio"] > 0 else 1.0
    return (
        pitch_err
        + metrics["flat_diff"] * 2.0
        + rms_err * 0.2
        + centroid_err * 0.3
        - metrics["mfcc_cos"] * 0.1
        - metrics["f0_corr"] * 0.1
    )


def _parse_list(value: str | None, default: list[float]) -> list[float]:
    if not value:
        return default
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return default
    return [int(x.strip()) for x in value.split(",") if x.strip()]


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
            f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
            f0 = f0[np.isfinite(f0)]
            if f0.size == 0:
                continue
            score = float(np.median(f0))
            scored.append((score, path))
        except Exception:
            continue
    scored.sort(key=lambda item: item[0])
    return [path for _score, path in scored] or wavs


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-tune inference params for a profile.")
    parser.add_argument("--profile", help="Profile name (uses outputs/training/<profile>).")
    parser.add_argument("--model_path", type=Path, help="Override model checkpoint path.")
    parser.add_argument("--config_path", type=Path, help="Override config path.")
    parser.add_argument("--ref_wav", type=Path, help="Reference wav for tuning.")
    parser.add_argument("--ref_dir", type=Path, help="Directory of reference wavs.")
    parser.add_argument("--ref_count", type=int, default=1, help="Number of references to use.")
    parser.add_argument(
        "--text",
        default="Hello, this is a quick pitch calibration sample.",
        help="Text prompt for the probe generation.",
    )
    parser.add_argument("--probe_texts", type=Path, help="Text file with one prompt per line.")
    parser.add_argument("--thorough", action="store_true", help="Use multiple refs + probe texts.")
    parser.add_argument("--alphas", help="Comma-separated alpha values.")
    parser.add_argument("--betas", help="Comma-separated beta values.")
    parser.add_argument("--diffusions", help="Comma-separated diffusion steps.")
    parser.add_argument("--embeddings", help="Comma-separated embedding scales.")
    parser.add_argument("--save_best", action="store_true", help="Save best sample wav.")
    parser.add_argument("--phonemizer_lang", type=str, help="Override phonemizer language.")
    parser.add_argument("--lexicon_path", type=Path, help="Lexicon JSON for phoneme overrides.")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    training_dir = None
    profile_dir = None

    if args.profile:
        training_dir = project_root / "outputs" / "training" / args.profile
        profile_dir = project_root / "data" / args.profile
    elif args.model_path:
        training_dir = args.model_path.parent

    model_path = args.model_path
    if model_path is None and training_dir is not None:
        model_path = _find_latest_checkpoint(training_dir)

    if model_path is None or not model_path.exists():
        raise FileNotFoundError("Model checkpoint not found. Provide --model_path or --profile.")

    config_path = args.config_path
    if config_path is None:
        candidate = model_path.parent / "config_ft.yml"
        if candidate.exists():
            config_path = candidate

    if config_path is None or not config_path.exists():
        raise FileNotFoundError("Config not found. Provide --config_path or ensure config_ft.yml exists.")

    ref_wav = args.ref_wav
    if ref_wav is None and profile_dir is not None:
        ref_wav = _pick_reference_wav(profile_dir)

    texts = _load_texts(args.probe_texts)
    if not texts:
        texts = DEFAULT_PROBE_TEXTS if args.thorough else [args.text]
    if args.text and args.text not in texts:
        texts.append(args.text)

    ref_wavs: list[Path] = []
    if ref_wav is not None:
        ref_wavs.append(ref_wav)
    target_refs = max(args.ref_count, 3) if args.thorough else args.ref_count

    ref_dir = args.ref_dir
    if ref_dir is None and profile_dir is not None:
        candidate = profile_dir / "processed_wavs"
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
        raise FileNotFoundError("Reference wav not found. Provide --ref_wav or --ref_dir.")

    alphas = _parse_list(args.alphas, [0.1, 0.3])
    betas = _parse_list(args.betas, [0.1, 0.3])
    diffusions = _parse_int_list(args.diffusions, [10, 25])
    embeddings = _parse_list(args.embeddings, [1.0, 1.3])

    print(f"Using model: {model_path}")
    print(f"Using config: {config_path}")
    print(f"Using {len(ref_wavs)} reference wav(s) and {len(texts)} prompt(s).")

    engine = StyleTTS2RepoEngine(model_path=model_path, config_path=config_path)
    sr = engine.sample_rate

    lexicon = _load_lexicon(args.lexicon_path)

    # First pass to estimate f0_scale.
    probe_text = texts[0]
    f0_scales = []
    for ref_path in ref_wavs:
        ref_audio = _load_mono(ref_path, sr)
        probe_audio = engine.generate(
            probe_text,
            ref_wav_path=ref_path,
            alpha=0.1,
            beta=0.1,
            diffusion_steps=10,
            embedding_scale=1.5,
            f0_scale=1.0,
            phonemizer_lang=args.phonemizer_lang,
            lexicon=lexicon,
        )
        f0_ref = _median_f0(_trim(ref_audio), sr)
        f0_out = _median_f0(_trim(probe_audio), sr)
        f0_scales.append(1.0 / (f0_out / f0_ref))

    f0_scale = float(np.median(f0_scales)) if f0_scales else 1.0

    print(f"Estimated f0_scale: {f0_scale:.4f}")

    best = None
    best_audio = None
    for alpha in alphas:
        for beta in betas:
            for diffusion_steps in diffusions:
                for embedding_scale in embeddings:
                    all_metrics: dict[str, list[float]] = {
                        "pitch_ratio": [],
                        "mfcc_cos": [],
                        "flat_diff": [],
                        "rms_ratio": [],
                        "centroid_ratio": [],
                        "f0_corr": [],
                    }
                    failures = 0
                    best_sample = None
                    best_sample_score = None
                    for ref_path in ref_wavs:
                        ref_audio = _load_mono(ref_path, sr)
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
                                    phonemizer_lang=args.phonemizer_lang,
                                    lexicon=lexicon,
                                )
                                metrics = _metrics(ref_audio, audio, sr)
                                score = _score(metrics)
                                for key, value in metrics.items():
                                    if np.isfinite(value):
                                        all_metrics[key].append(float(value))
                                if best_sample_score is None or score < best_sample_score:
                                    best_sample_score = score
                                    best_sample = audio
                            except Exception:
                                failures += 1
                                continue

                    summary = {}
                    for key, values in all_metrics.items():
                        summary[key] = float(np.mean(values)) if values else 0.0
                    score = _score(summary) + (failures * 0.5)
                    record = {
                        "alpha": alpha,
                        "beta": beta,
                        "diffusion_steps": diffusion_steps,
                        "embedding_scale": embedding_scale,
                        "f0_scale": f0_scale,
                        "metrics": summary,
                        "score": score,
                        "failures": failures,
                    }
                    if best is None or score < best["score"]:
                        best = record
                        best_audio = best_sample

    if best is None:
        raise RuntimeError("No candidates evaluated.")

    profile = {
        "model_path": str(model_path),
        "config_path": str(config_path),
        "ref_wav_path": str(ref_wavs[0]),
        "alpha": best["alpha"],
        "beta": best["beta"],
        "diffusion_steps": best["diffusion_steps"],
        "embedding_scale": best["embedding_scale"],
        "f0_scale": best["f0_scale"],
        "score": best["score"],
        "metrics": best["metrics"],
    }
    if args.phonemizer_lang:
        profile["phonemizer_lang"] = args.phonemizer_lang
    if args.lexicon_path:
        profile["lexicon_path"] = str(args.lexicon_path)

    profile_path = model_path.parent / "profile.json"
    profile_path.write_text(json.dumps(profile, indent=2))
    print(f"Saved profile defaults: {profile_path}")

    f0_path = model_path.parent / "f0_scale.txt"
    f0_path.write_text(f"{best['f0_scale']:.4f}\n")
    print(f"Saved f0_scale: {f0_path}")

    if args.save_best and best_audio is not None:
        out_dir = project_root / "outputs" / "inference" / model_path.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_path.stem}_best.wav"
        sf.write(out_path, best_audio, sr)
        print(f"Saved best sample: {out_path}")


if __name__ == "__main__":
    main()
