import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from inference import StyleTTS2RepoEngine


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
    mfcc_cos = float(np.dot(v_ref, v_out) / (np.linalg.norm(v_ref) * np.linalg.norm(v_out)))

    flat_ref = float(np.mean(librosa.feature.spectral_flatness(y=ref_trim)))
    flat_out = float(np.mean(librosa.feature.spectral_flatness(y=out_trim)))
    flat_diff = abs(flat_out - flat_ref)

    rms_ref = float(np.sqrt(np.mean(np.square(ref_trim))))
    rms_out = float(np.sqrt(np.mean(np.square(out_trim))))
    rms_ratio = rms_out / rms_ref if rms_ref > 0 else 1.0

    return {
        "pitch_ratio": pitch_ratio,
        "mfcc_cos": mfcc_cos,
        "flat_diff": flat_diff,
        "rms_ratio": rms_ratio,
    }


def _score(metrics: dict[str, float]) -> float:
    return (
        abs(metrics["pitch_ratio"] - 1.0)
        + metrics["flat_diff"] * 2.0
        + abs(metrics["rms_ratio"] - 1.0) * 0.2
        - metrics["mfcc_cos"] * 0.1
    )


def _parse_list(value: str | None, default: list[float]) -> list[float]:
    if not value:
        return default
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return default
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-tune inference params for a profile.")
    parser.add_argument("--profile", help="Profile name (uses outputs/training/<profile>).")
    parser.add_argument("--model_path", type=Path, help="Override model checkpoint path.")
    parser.add_argument("--config_path", type=Path, help="Override config path.")
    parser.add_argument("--ref_wav", type=Path, help="Reference wav for tuning.")
    parser.add_argument(
        "--text",
        default="Hello, this is a quick pitch calibration sample.",
        help="Text prompt for the probe generation.",
    )
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

    if ref_wav is None or not ref_wav.exists():
        raise FileNotFoundError("Reference wav not found. Provide --ref_wav.")

    alphas = _parse_list(args.alphas, [0.1, 0.3])
    betas = _parse_list(args.betas, [0.1, 0.3])
    diffusions = _parse_int_list(args.diffusions, [10, 25])
    embeddings = _parse_list(args.embeddings, [1.0, 1.3])

    print(f"Using model: {model_path}")
    print(f"Using config: {config_path}")
    print(f"Using ref wav: {ref_wav}")

    engine = StyleTTS2RepoEngine(model_path=model_path, config_path=config_path)
    sr = engine.sample_rate

    ref_audio = _load_mono(ref_wav, sr)
    lexicon = _load_lexicon(args.lexicon_path)

    # First pass to estimate f0_scale.
    probe_audio = engine.generate(
        args.text,
        ref_wav_path=ref_wav,
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
    f0_scale = 1.0 / (f0_out / f0_ref)

    print(f"Estimated f0_scale: {f0_scale:.4f}")

    best = None
    best_audio = None
    for alpha in alphas:
        for beta in betas:
            for diffusion_steps in diffusions:
                for embedding_scale in embeddings:
                    audio = engine.generate(
                        args.text,
                        ref_wav_path=ref_wav,
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
                    record = {
                        "alpha": alpha,
                        "beta": beta,
                        "diffusion_steps": diffusion_steps,
                        "embedding_scale": embedding_scale,
                        "f0_scale": f0_scale,
                        "metrics": metrics,
                        "score": score,
                    }
                    if best is None or score < best["score"]:
                        best = record
                        best_audio = audio

    if best is None:
        raise RuntimeError("No candidates evaluated.")

    profile = {
        "model_path": str(model_path),
        "config_path": str(config_path),
        "ref_wav_path": str(ref_wav),
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
        out_dir = project_root / "outputs" / "inference"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_path.stem}_best.wav"
        sf.write(out_path, best_audio, sr)
        print(f"Saved best sample: {out_path}")


if __name__ == "__main__":
    main()
