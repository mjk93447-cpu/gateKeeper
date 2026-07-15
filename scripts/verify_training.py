from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from gatekeeper.training.cpu_runner import CpuTrainingConfig, build_yolo26_command
from gatekeeper.training.verification import validate_group_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and optionally smoke-test CPU training")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--groups", type=Path, help="JSON panel/lot/recipe split manifest")
    parser.add_argument("--smoke", action="store_true", help="Run one CPU epoch")
    args = parser.parse_args()
    errors: list[str] = []
    if not args.data.is_file():
        errors.append(f"missing data YAML: {args.data}")
    if not args.pretrained.is_file():
        errors.append(f"missing pretrained checkpoint: {args.pretrained}")
    if args.groups:
        errors.extend(validate_group_split(args.groups))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    config = CpuTrainingConfig(
        data_yaml=args.data,
        pretrained=args.pretrained,
        epochs=1 if args.smoke else 100,
        patience=1 if args.smoke else 20,
        output_dir=Path("runs/verification" if args.smoke else "runs/gatekeeper"),
    )
    command = build_yolo26_command(config)
    print("Training command:", subprocess.list2cmdline(command))
    if not args.smoke:
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
