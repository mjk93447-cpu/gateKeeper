from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

VALID_LABEL = re.compile(r"^[A-Z0-9]{4}$")


def validate_ocr_labels(
    labels_file: Path, allowed_codes: frozenset[str] | None = None
) -> tuple[list[str], Counter[str]]:
    """Validate OCR labels and optionally require registered code values."""

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
            errors.append(f"line {line_number}: label must contain exactly four A-Z/0-9 characters")
        elif allowed_codes is not None and label not in allowed_codes:
            errors.append(
                f"line {line_number}: label '{label}' is not registered in the code recipe"
            )
        if not (labels_file.parent / relative_path).is_file():
            errors.append(f"line {line_number}: missing image '{relative_path}'")
        counts[label] += 1
    return errors, counts
