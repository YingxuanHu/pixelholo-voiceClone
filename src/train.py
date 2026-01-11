import argparse
import random
import subprocess
import sys
import warnings
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

warnings.filterwarnings(
    "ignore",
    message="`torch.nn.utils.weight_norm` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="dropout option adds dropout after all but last recurrent layer",
    category=UserWarning,
)

from config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_FP16,
    DEFAULT_MAX_LEN,
    METADATA_FILENAME,
    PROCESSED_WAVS_DIRNAME,
    STYLE_TTS2_DIR,
)


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(output_dir.glob("epoch_2nd_*.pth"))
    return checkpoints[-1] if checkpoints else None


def _auto_pick_ref_wav(dataset_root: Path) -> Path | None:
    wav_dir = dataset_root / PROCESSED_WAVS_DIRNAME
    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        return None

    try:
        import librosa
        import numpy as np
        import soundfile as sf
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
            f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
            f0 = f0[np.isfinite(f0)]
            if f0.size == 0:
                continue
            score = float(np.median(f0))
            if best is None or score < best[0]:
                best = (score, path)
        except Exception:
            continue

    return best[1] if best else wavs[0]


def _load_base_config(styletts2_dir: Path, base_config: Path | None) -> tuple[dict, Path]:
    if base_config:
        config_path = base_config
    else:
        candidates = [
            styletts2_dir / "Configs" / "config_ft.yml",
            styletts2_dir / "Configs" / "config.yml",
        ]
        config_path = next((p for p in candidates if p.exists()), None)
        if config_path is None:
            raise FileNotFoundError("No base config found in StyleTTS2 repo.")

    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)
    return config, config_path


def _write_train_val_lists(
    meta_path: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
) -> tuple[Path, Path, Path]:
    with meta_path.open("r") as handle:
        rows = [line.strip().split("|") for line in handle if line.strip()]

    random.Random(seed).shuffle(rows)
    split_idx = max(1, int(len(rows) * (1.0 - val_ratio)))
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    train_list = output_dir / "train_list.txt"
    val_list = output_dir / "val_list.txt"
    ood_list = output_dir / "ood_texts.txt"

    def _write_list(path: Path, entries: list[list[str]]) -> None:
        with path.open("w") as handle:
            for filename, text, _speaker in entries:
                handle.write(f"{filename}|{text}|0\n")

    _write_list(train_list, train_rows)
    _write_list(val_list, val_rows if val_rows else train_rows[:1])
    ood_lines = [text for _fname, text, _speaker in (train_rows or val_rows)]
    if not ood_lines:
        ood_lines = ["hello world"]
    with ood_list.open("w") as handle:
        for line in ood_lines:
            handle.write(f"{line}\n")

    return train_list, val_list, ood_list


def _patch_config(
    config: dict,
    dataset_root: Path,
    output_dir: Path,
    train_list: Path,
    val_list: Path,
    ood_list: Path,
    batch_size: int,
    max_len: int,
    fp16_run: bool,
    epochs: int,
    pretrained_model: str | None,
    max_steps: int | None,
    grad_accum_steps: int | None,
) -> dict:
    config["log_dir"] = str(output_dir)
    config["batch_size"] = batch_size
    config["max_len"] = max_len
    config["fp16_run"] = fp16_run
    config["epochs"] = epochs
    config["num_workers"] = config.get("num_workers", 0)
    config["val_num_workers"] = config.get("val_num_workers", 0)

    data_params = config.get("data_params", {})
    data_params["train_data"] = str(train_list)
    data_params["val_data"] = str(val_list)
    data_params["OOD_data"] = str(ood_list)
    data_params["root_path"] = str((dataset_root / PROCESSED_WAVS_DIRNAME).resolve())
    config["data_params"] = data_params

    preprocess_params = config.get("preprocess_params", {})
    if "sr" not in preprocess_params:
        preprocess_params["sr"] = 44100
    config["preprocess_params"] = preprocess_params

    if pretrained_model:
        config["pretrained_model"] = pretrained_model
    if max_steps is not None:
        config["max_steps"] = max_steps
    if grad_accum_steps is not None:
        config["grad_accum_steps"] = grad_accum_steps

    return config


def _resolve_entrypoint(styletts2_dir: Path, use_accelerate: bool) -> Path:
    if use_accelerate:
        entry = styletts2_dir / "train_finetune_accelerate.py"
        if entry.exists():
            return entry
    entry = styletts2_dir / "train_finetune.py"
    if entry.exists():
        return entry
    raise FileNotFoundError("No training entrypoint found in StyleTTS2 repo.")


def launch_training(
    styletts2_dir: Path,
    config_path: Path,
    use_accelerate: bool,
) -> None:
    entrypoint = _resolve_entrypoint(styletts2_dir, use_accelerate)
    config_path = config_path.resolve()

    if use_accelerate and entrypoint.name == "train_finetune_accelerate.py":
        command = ["accelerate", "launch", str(entrypoint), "-p", str(config_path)]
    else:
        command = [sys.executable, str(entrypoint), "-p", str(config_path)]

    subprocess.run(command, cwd=styletts2_dir, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune StyleTTS2 on a single-speaker dataset.")
    parser.add_argument("--dataset_path", required=True, type=Path, help="Path to dataset root")
    parser.add_argument("--styletts2_dir", type=Path, default=STYLE_TTS2_DIR, help="Path to StyleTTS2 repo")
    parser.add_argument("--base_config", type=Path, help="Base StyleTTS2 config to patch")
    parser.add_argument("--output_dir", type=Path, help="Output directory for checkpoints/logs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--fp16_run", type=bool, default=DEFAULT_FP16)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--pretrained_checkpoint", type=str, help="Path to LibriTTS checkpoint")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--use_accelerate", action="store_true")
    parser.add_argument("--max_steps", type=int, help="Stop after this many training steps")
    parser.add_argument("--grad_accum_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--dry_run", action="store_true", help="Only write config, do not start training")
    parser.add_argument(
        "--auto_tune_profile",
        action="store_true",
        help="After training, auto-tune inference defaults and write profile.json.",
    )
    parser.add_argument("--tune_ref_wav", type=Path, help="Reference wav for profile tuning")
    parser.add_argument("--tune_ref_dir", type=Path, help="Directory of reference wavs for tuning")
    parser.add_argument("--tune_ref_count", type=int, default=1, help="Number of references for tuning")
    parser.add_argument(
        "--tune_text",
        type=str,
        default="Hello, this is a quick pitch calibration sample.",
        help="Probe text for profile tuning",
    )
    parser.add_argument("--tune_probe_texts", type=Path, help="Text file with prompts for tuning")
    parser.add_argument("--tune_thorough", action="store_true", help="Use multiple refs/texts for tuning")
    parser.add_argument(
        "--tune_quick",
        action="store_true",
        help="Disable thorough tuning even when auto-tuning is enabled.",
    )
    parser.add_argument(
        "--auto_select_epoch",
        action="store_true",
        help="After training, score checkpoints and write best_epoch.txt.",
    )
    parser.add_argument("--select_ref_wav", type=Path, help="Reference wav for epoch selection")
    parser.add_argument("--select_ref_dir", type=Path, help="Directory of reference wavs for selection")
    parser.add_argument("--select_ref_count", type=int, default=1, help="Number of references for selection")
    parser.add_argument(
        "--select_text",
        type=str,
        default="Hello, this is a quick pitch calibration sample.",
        help="Probe text for epoch selection",
    )
    parser.add_argument("--select_probe_texts", type=Path, help="Text file with prompts for selection")
    parser.add_argument("--select_thorough", action="store_true", help="Use multiple refs/texts for selection")
    parser.add_argument(
        "--select_quick",
        action="store_true",
        help="Disable thorough selection even when auto-select is enabled.",
    )
    parser.add_argument("--select_limit", type=int, help="Check last N checkpoints")
    parser.add_argument(
        "--auto_build_lexicon",
        action="store_true",
        help="After training, generate data/<profile>/lexicon.json from metadata.csv.",
    )
    parser.add_argument("--lexicon_lang", type=str, default="en-us", help="Phonemizer language for lexicon")
    parser.add_argument("--lexicon_min_count", type=int, default=1, help="Minimum word count for lexicon")

    args = parser.parse_args()

    if args.tune_quick:
        args.tune_thorough = False
    elif args.auto_tune_profile:
        args.tune_thorough = True

    if args.select_quick:
        args.select_thorough = False
    elif args.auto_select_epoch:
        args.select_thorough = True

    if not args.styletts2_dir.exists():
        raise FileNotFoundError(
            f"StyleTTS2 repo not found at {args.styletts2_dir}. Clone it into lib/StyleTTS2."
        )

    args.dataset_path = args.dataset_path.resolve()
    meta_path = args.dataset_path / METADATA_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at {meta_path}")

    output_dir = args.output_dir or (PROJECT_ROOT / "outputs" / "training" / args.dataset_path.name)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_list, val_list, ood_list = _write_train_val_lists(
        meta_path=meta_path,
        output_dir=output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    config, base_config_path = _load_base_config(args.styletts2_dir, args.base_config)
    config = _patch_config(
        config=config,
        dataset_root=args.dataset_path,
        output_dir=output_dir,
        train_list=train_list,
        val_list=val_list,
        ood_list=ood_list,
        batch_size=args.batch_size,
        max_len=args.max_len,
        fp16_run=args.fp16_run,
        epochs=args.epochs,
        pretrained_model=args.pretrained_checkpoint,
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum_steps,
    )

    output_config_path = output_dir / "config_ft.yml"
    with output_config_path.open("w") as handle:
        yaml.dump(config, handle, default_flow_style=False)

    print(f"Base config: {base_config_path}")
    print(f"Patched config: {output_config_path}")

    if args.dry_run:
        return

    launch_training(
        styletts2_dir=args.styletts2_dir,
        config_path=output_config_path,
        use_accelerate=args.use_accelerate,
    )

    latest_ckpt = _latest_checkpoint(output_dir)
    if latest_ckpt is None:
        return

    auto_ref = None
    if args.auto_tune_profile or args.auto_select_epoch:
        auto_ref = _auto_pick_ref_wav(args.dataset_path)
        if auto_ref:
            print(f"Auto-picked reference wav: {auto_ref}")

    if args.auto_tune_profile:
        tune_script = PROJECT_ROOT / "src" / "auto_tune_profile.py"
        command = [
            sys.executable,
            str(tune_script),
            "--model_path",
            str(latest_ckpt),
            "--config_path",
            str(output_config_path),
            "--text",
            args.tune_text,
            "--save_best",
        ]
        if args.tune_probe_texts:
            command += ["--probe_texts", str(args.tune_probe_texts)]
        if args.tune_ref_dir:
            command += ["--ref_dir", str(args.tune_ref_dir)]
        if args.tune_ref_count:
            command += ["--ref_count", str(args.tune_ref_count)]
        if args.tune_thorough:
            command.append("--thorough")
        ref = args.tune_ref_wav or auto_ref
        if ref:
            command += ["--ref_wav", str(ref)]
        subprocess.run(command, check=True)

    if args.auto_select_epoch:
        select_script = PROJECT_ROOT / "src" / "auto_select_epoch.py"
        command = [
            sys.executable,
            str(select_script),
            "--training_dir",
            str(output_dir),
            "--config_path",
            str(output_config_path),
            "--text",
            args.select_text,
            "--save_best",
        ]
        if args.select_probe_texts:
            command += ["--probe_texts", str(args.select_probe_texts)]
        if args.select_ref_dir:
            command += ["--ref_dir", str(args.select_ref_dir)]
        if args.select_ref_count:
            command += ["--ref_count", str(args.select_ref_count)]
        if args.select_thorough:
            command.append("--thorough")
        if args.select_limit is not None:
            command += ["--limit", str(args.select_limit)]
        ref = args.select_ref_wav or args.tune_ref_wav or auto_ref
        if ref:
            command += ["--ref_wav", str(ref)]
        subprocess.run(command, check=True)

    if args.auto_build_lexicon:
        lex_script = PROJECT_ROOT / "src" / "build_lexicon.py"
        command = [
            sys.executable,
            str(lex_script),
            "--profile",
            args.dataset_path.name,
            "--lang",
            args.lexicon_lang,
            "--min_count",
            str(args.lexicon_min_count),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
