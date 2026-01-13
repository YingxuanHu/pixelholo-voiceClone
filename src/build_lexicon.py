import argparse
import json
import re
from collections import Counter
from pathlib import Path

import phonemizer


WORD_RE = re.compile(r"[A-Za-z']+")


def _extract_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def _load_metadata(metadata_path: Path) -> list[str]:
    lines = metadata_path.read_text().splitlines()
    texts = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        texts.append(parts[1].strip())
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a default lexicon.json for a profile.")
    parser.add_argument("--profile", help="Profile name (uses data/<profile>/metadata.csv).")
    parser.add_argument("--metadata", type=Path, help="Path to metadata.csv.")
    parser.add_argument("--lang", default="en-ca", help="Phonemizer language (default: en-ca).")
    parser.add_argument("--min_count", type=int, default=1, help="Only include words seen N times.")
    parser.add_argument("--output", type=Path, help="Output lexicon.json path.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if args.profile:
        metadata_path = project_root / "data" / args.profile / "metadata.csv"
        output_path = project_root / "data" / args.profile / "lexicon.json"
    elif args.metadata:
        metadata_path = args.metadata
        output_path = metadata_path.parent / "lexicon.json"
    else:
        raise SystemExit("Provide --profile or --metadata.")

    if args.output:
        output_path = args.output

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    texts = _load_metadata(metadata_path)
    counts = Counter()
    for text in texts:
        counts.update(_extract_words(text))

    words = [word for word, count in counts.items() if count >= args.min_count]
    words.sort()

    try:
        backend = phonemizer.backend.EspeakBackend(
            language=args.lang,
            preserve_punctuation=False,
            with_stress=True,
        )
    except RuntimeError:
        backend = phonemizer.backend.EspeakBackend(
            language="en-us",
            preserve_punctuation=False,
            with_stress=True,
        )

    lexicon = {}
    for word in words:
        phoneme = backend.phonemize([word])[0].strip()
        lexicon[word] = phoneme

    output_path.write_text(json.dumps(lexicon, indent=2, ensure_ascii=True))
    print(f"Wrote {output_path} ({len(lexicon)} words)")


if __name__ == "__main__":
    main()
