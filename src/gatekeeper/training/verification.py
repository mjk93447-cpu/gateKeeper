from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_group_split(manifest_path: str | Path) -> list[str]:
    """Return leakage errors for a manifest with image -> split/group records."""

    payload: dict[str, Any] = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    errors: list[str] = []
    groups: dict[str, str] = {}
    for record in payload.get("images", payload.get("records", [])):
        group = str(record.get("panel_id") or record.get("lot_id") or record.get("group", ""))
        split = str(record.get("split", ""))
        if not group or not split:
            errors.append("every record requires a group and split")
            continue
        previous = groups.setdefault(group, split)
        if previous != split:
            errors.append(f"group {group!r} appears in both {previous!r} and {split!r}")
    required = {"train", "val", "test"}
    present = {str(record.get("split", "")) for record in payload.get("images", [])}
    missing = required - present
    if missing:
        errors.append(f"missing required splits: {sorted(missing)}")
    return errors
