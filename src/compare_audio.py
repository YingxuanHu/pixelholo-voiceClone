import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _analyze(audio: np.ndarray, sr: int, top_db: int) -> dict[str, float]:
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    if trimmed.size == 0:
        trimmed = audio

    rms = float(np.sqrt(np.mean(np.square(trimmed))))
    peak = float(np.max(np.abs(trimmed))) if trimmed.size else 0.0

    f0 = librosa.yin(trimmed, fmin=50, fmax=500, sr=sr)
    f0 = f0[np.isfinite(f0)]
    f0_median = float(np.median(f0)) if f0.size else float("nan")
    f0_mean = float(np.mean(f0)) if f0.size else float("nan")

    centroid = librosa.feature.spectral_centroid(y=trimmed, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=trimmed)
    zcr = librosa.feature.zero_crossing_rate(trimmed)

    return {
        "sr": sr,
        "duration_s": float(len(audio) / sr),
        "trimmed_duration_s": float(len(trimmed) / sr),
        "rms": rms,
        "peak": peak,
        "f0_median": f0_median,
        "f0_mean": f0_mean,
        "centroid_mean": float(np.mean(centroid)) if centroid.size else float("nan"),
        "flatness_mean": float(np.mean(flatness)) if flatness.size else float("nan"),
        "zcr_mean": float(np.mean(zcr)) if zcr.size else float("nan"),
    }


def _compare(ref: np.ndarray, out: np.ndarray, sr: int) -> dict[str, float]:
    min_len = min(len(ref), len(out))
    if min_len == 0:
        return {"mfcc_cosine": float("nan"), "pitch_ratio": float("nan")}
    ref = ref[:min_len]
    out = out[:min_len]

    mfcc_ref = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=13)
    mfcc_out = librosa.feature.mfcc(y=out, sr=sr, n_mfcc=13)
    v_ref = np.mean(mfcc_ref, axis=1)
    v_out = np.mean(mfcc_out, axis=1)
    mfcc_cosine = float(np.dot(v_ref, v_out) / (np.linalg.norm(v_ref) * np.linalg.norm(v_out)))

    f0_ref = librosa.yin(ref, fmin=50, fmax=500, sr=sr)
    f0_out = librosa.yin(out, fmin=50, fmax=500, sr=sr)
    f0_ref = f0_ref[np.isfinite(f0_ref)]
    f0_out = f0_out[np.isfinite(f0_out)]
    if f0_ref.size and f0_out.size:
        pitch_ratio = float(np.median(f0_out) / np.median(f0_ref))
    else:
        pitch_ratio = float("nan")

    return {"mfcc_cosine": mfcc_cosine, "pitch_ratio": pitch_ratio}


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(f"{label}:")
    print(f"  sr: {metrics['sr']}")
    print(
        "  duration:",
        f"{metrics['duration_s']:.2f}s (trim {metrics['trimmed_duration_s']:.2f}s)",
    )
    print(f"  rms: {metrics['rms']:.6f}  peak: {metrics['peak']:.6f}")
    print(
        "  f0 median:",
        f"{metrics['f0_median']:.2f} Hz  mean {metrics['f0_mean']:.2f} Hz",
    )
    print(f"  centroid mean: {metrics['centroid_mean']:.1f} Hz")
    print(f"  flatness mean: {metrics['flatness_mean']:.6f}")
    print(f"  zcr mean: {metrics['zcr_mean']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare output wav against reference.")
    parser.add_argument("--ref", required=True, type=Path, help="Reference wav path")
    parser.add_argument("--out", required=True, type=Path, help="Generated wav path")
    parser.add_argument("--sr", type=int, default=24000, help="Target sample rate")
    parser.add_argument("--top_db", type=int, default=35, help="Trim threshold in dB")
    args = parser.parse_args()

    ref_audio, ref_sr = _load_mono(args.ref)
    out_audio, out_sr = _load_mono(args.out)

    if ref_sr != args.sr:
        ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=args.sr)
    if out_sr != args.sr:
        out_audio = librosa.resample(out_audio, orig_sr=out_sr, target_sr=args.sr)

    ref_metrics = _analyze(ref_audio, args.sr, args.top_db)
    out_metrics = _analyze(out_audio, args.sr, args.top_db)
    comp = _compare(ref_audio, out_audio, args.sr)

    _print_metrics("reference", ref_metrics)
    _print_metrics("output", out_metrics)
    print("comparison:")
    print(f"  mfcc cosine: {comp['mfcc_cosine']:.4f}")
    print(f"  pitch ratio (out/ref median): {comp['pitch_ratio']:.3f}")


if __name__ == "__main__":
    main()
