import argparse
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


def _median_f0(audio: np.ndarray, sr: int) -> float:
    trimmed, _ = librosa.effects.trim(audio, top_db=35)
    if trimmed.size == 0:
        trimmed = audio
    f0 = librosa.yin(trimmed, fmin=50, fmax=500, sr=sr)
    f0 = f0[np.isfinite(f0)]
    if f0.size == 0:
        raise ValueError("Unable to estimate F0 from audio.")
    return float(np.median(f0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-compute f0_scale for a trained profile.")
    parser.add_argument("--profile", help="Profile name (uses outputs/training/<profile>).")
    parser.add_argument("--model_path", type=Path, help="Override model checkpoint path.")
    parser.add_argument("--config_path", type=Path, help="Override config path.")
    parser.add_argument("--ref_wav", type=Path, help="Reference wav for pitch comparison.")
    parser.add_argument(
        "--text",
        default="Hello, this is a quick pitch calibration sample.",
        help="Text prompt for the probe generation.",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--diffusion_steps", type=int, default=10)
    parser.add_argument("--embedding_scale", type=float, default=1.5)
    parser.add_argument("--out_wav", type=Path, help="Where to save the probe output.")
    parser.add_argument("--scale_out", type=Path, help="Where to write f0_scale.txt.")

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

    out_wav = args.out_wav
    if out_wav is None:
        out_dir = project_root / "outputs" / "inference"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_wav = out_dir / f"{model_path.stem}_f0_probe.wav"

    scale_out = args.scale_out or (model_path.parent / "f0_scale.txt")

    print(f"Using model: {model_path}")
    print(f"Using config: {config_path}")
    print(f"Using ref wav: {ref_wav}")

    engine = StyleTTS2RepoEngine(model_path=model_path, config_path=config_path)
    audio = engine.generate(
        args.text,
        ref_wav_path=ref_wav,
        alpha=args.alpha,
        beta=args.beta,
        diffusion_steps=args.diffusion_steps,
        embedding_scale=args.embedding_scale,
        f0_scale=1.0,
    )

    sr = engine.sample_rate
    sf.write(out_wav, audio, sr)
    print(f"Probe audio saved: {out_wav}")

    ref_audio = _load_mono(ref_wav, sr)
    out_audio = _load_mono(out_wav, sr)
    f0_ref = _median_f0(ref_audio, sr)
    f0_out = _median_f0(out_audio, sr)
    ratio = f0_out / f0_ref
    f0_scale = 1.0 / ratio

    scale_out.write_text(f"{f0_scale:.4f}\n")

    print(f"ref median f0: {f0_ref:.2f} Hz")
    print(f"out median f0: {f0_out:.2f} Hz")
    print(f"pitch ratio (out/ref): {ratio:.3f}")
    print(f"recommended f0_scale: {f0_scale:.4f}")
    print(f"Saved f0_scale to: {scale_out}")


if __name__ == "__main__":
    main()
