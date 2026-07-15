from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

VALID_LABEL = re.compile(r"^[A-Z0-9]{4}$")


def validate(labels_file: Path) -> tuple[list[str], Counter[str]]:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    for line_number, raw_line in enumerate(labels_file.read_text(encoding="utf-8").splitlines(), 1):
        try:
            relative_path, label = raw_line.rsplit("\t", 1)
        except ValueError:
            errors.append(f"line {line_number}: expected '<path>\\t<label>'")
            continue
        label = label.strip().upper()
        if not VALID_LABEL.fullmatch(label):
            errors.append(
                f"line {line_number}: label must contain exactly four A-Z/0-9 characters"
            )
        if not (labels_file.parent / relative_path).is_file():
            errors.append(f"line {line_number}: missing image '{relative_path}'")
        counts[label] += 1
    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate gateKeeper PaddleOCR label file")
    parser.add_argument("labels", type=Path)
    args = parser.parse_args()
    errors, counts = validate(args.labels)
    print(f"label counts: {dict(counts)}")
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
