import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from inference import StyleTTS2RepoEngine


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick best sounding epoch by audio similarity.")
    parser.add_argument("--training_dir", type=Path, required=True)
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--ref_wav", type=Path, required=True)
    parser.add_argument(
        "--text",
        default="Hello, this is a quick pitch calibration sample.",
        help="Text prompt for the probe generation.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Only check last N checkpoints.")
    parser.add_argument("--f0_scale", type=float, help="Override f0_scale.")
    parser.add_argument("--alpha", type=float, help="Override alpha.")
    parser.add_argument("--beta", type=float, help="Override beta.")
    parser.add_argument("--diffusion_steps", type=int, help="Override diffusion steps.")
    parser.add_argument("--embedding_scale", type=float, help="Override embedding scale.")
    parser.add_argument("--save_best", action="store_true", help="Save best sample wav.")
    parser.add_argument("--phonemizer_lang", type=str, help="Override phonemizer language.")
    parser.add_argument("--lexicon_path", type=Path, help="Lexicon JSON for phoneme overrides.")

    args = parser.parse_args()

    training_dir = args.training_dir.resolve()
    config_path = args.config_path.resolve()
    ref_wav = args.ref_wav.resolve()

    checkpoints = sorted(training_dir.glob("epoch_2nd_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {training_dir}")
    if args.limit:
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

    print(f"Evaluating {len(checkpoints)} checkpoints...")
    print(f"alpha={alpha} beta={beta} diffusion={diffusion_steps} embedding={embedding_scale} f0_scale={f0_scale}")

    ref_audio = _load_mono(ref_wav, 24000)

    best = None
    results = []

    for ckpt in checkpoints:
        engine = StyleTTS2RepoEngine(model_path=ckpt, config_path=config_path)
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
        metrics = _metrics(ref_audio, audio, engine.sample_rate)
        score = _score(metrics)
        entry = {
            "checkpoint": str(ckpt),
            "score": score,
            "metrics": metrics,
        }
        results.append(entry)
        if best is None or score < best["score"]:
            best = entry
            best_audio = audio

        del engine
        torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError("No checkpoints evaluated.")

    best_path = training_dir / "best_epoch.txt"
    best_path.write_text(best["checkpoint"] + "\n")

    results_path = training_dir / "epoch_scores.json"
    results_path.write_text(json.dumps(results, indent=2))

    print(f"Best checkpoint: {best['checkpoint']} (score {best['score']:.5f})")
    print(f"Wrote {best_path}")
    print(f"Wrote {results_path}")

    if args.save_best:
        out_dir = training_dir.parents[1] / "inference"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(best['checkpoint']).stem}_best.wav"
        sf.write(out_path, best_audio, 24000)
        print(f"Saved best sample: {out_path}")


if __name__ == "__main__":
    main()
