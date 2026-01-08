import argparse
import json
from pathlib import Path

import soundfile as sf

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
    return wavs[0] if wavs else None


def _load_profile_defaults(model_path: Path) -> dict:
    for filename in ("profile.json", "inference_defaults.json"):
        candidate = model_path.parent / filename
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
        model_path = _find_best_checkpoint(training_dir) or _find_latest_checkpoint(training_dir)

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

    defaults = _load_profile_defaults(model_path)
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
        else defaults.get("diffusion_steps", 10)
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
    lexicon = _load_lexicon(Path(lexicon_path)) if lexicon_path else None

    out_path = args.out
    if out_path is None:
        out_dir = project_root / "outputs" / "inference"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_path.stem}_speak.wav"

    engine = StyleTTS2RepoEngine(model_path=model_path, config_path=config_path)
    audio = engine.generate(
        args.text,
        ref_wav_path=ref_wav,
        alpha=alpha,
        beta=beta,
        diffusion_steps=diffusion_steps,
        embedding_scale=embedding_scale,
        f0_scale=f0_scale,
        phonemizer_lang=phonemizer_lang,
        lexicon=lexicon,
    )

    sf.write(out_path, audio, engine.sample_rate)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
