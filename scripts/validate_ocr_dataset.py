from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from gatekeeper.domain.code_recipe import CodeRecipe
from gatekeeper.training.code_labels import validate_ocr_labels

VALID_LABEL = re.compile(r"^[A-Z0-9]{4}$")


def validate(
    labels_file: Path, allowed_codes: frozenset[str] | None = None
) -> tuple[list[str], Counter[str]]:
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
        elif allowed_codes is not None and label not in allowed_codes:
            errors.append(
                f"line {line_number}: label '{label}' is not registered in the code recipe"
            )
        if not (labels_file.parent / relative_path).is_file():
            errors.append(f"line {line_number}: missing image '{relative_path}'")
        counts[label] += 1
    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate gateKeeper PaddleOCR label file")
    parser.add_argument("labels", type=Path)
    parser.add_argument("--recipe", type=Path, help="Code recipe JSON used to restrict labels")
    args = parser.parse_args()
    recipe = CodeRecipe.load(args.recipe) if args.recipe else None
    errors, counts = validate_ocr_labels(args.labels, recipe.all_codes if recipe else None)
    print(f"label counts: {dict(counts)}")
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
