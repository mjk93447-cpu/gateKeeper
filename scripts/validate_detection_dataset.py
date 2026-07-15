from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def validate(path: Path) -> tuple[list[str], Counter[int]]:
    errors: list[str] = []
    counts: Counter[int] = Counter()
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in payload.get("images", [])}
    categories = {item["id"]: item["name"] for item in payload.get("categories", [])}
    required = {"fpcb_surface", "code_roi"}
    category_names = {name.lower() for name in categories.values()}
    if not required.issubset(category_names):
        errors.append("categories must include 'fpcb_surface' and 'code_roi'")
    for annotation in payload.get("annotations", []):
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox", [])
        if image_id not in images:
            errors.append(f"annotation {annotation.get('id')} references missing image")
        if category_id not in categories:
            errors.append(f"annotation {annotation.get('id')} references missing category")
        if len(bbox) != 4 or any(value <= 0 for value in bbox[2:]):
            errors.append(f"annotation {annotation.get('id')} has invalid bbox")
        counts[category_id] += 1
    for image in images.values():
        image_path = path.parent / image["file_name"]
        if not image_path.is_file():
            errors.append(f"missing image: {image_path}")
    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate gateKeeper COCO detector dataset")
    parser.add_argument("annotations", type=Path)
    args = parser.parse_args()
    errors, counts = validate(args.annotations)
    print(f"annotation counts: {dict(counts)}")
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
