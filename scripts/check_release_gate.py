from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate production model approval metrics")
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "approved":
        raise SystemExit("release blocked: model manifest status must be 'approved'")
    validation = manifest.get("validation") or {}
    gates = {
        "exact_code_accuracy": 0.99,
        "problem_recall": 0.99,
        "problem_false_normal": 0.0,
    }
    for key, minimum in gates.items():
        value = validation.get(key)
        if value is None:
            raise SystemExit(f"release blocked: validation.{key} is missing")
        if key == "problem_false_normal":
            passed = float(value) <= minimum
        else:
            passed = float(value) >= minimum
        if not passed:
            raise SystemExit(
                f"release blocked: validation.{key}={value} does not meet required gate {minimum}"
            )
    print("production release gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
