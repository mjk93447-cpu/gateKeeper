from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def detector_command(yolox: Path, experiment: Path, weights: Path, devices: int) -> list[str]:
    return [
        "python",
        str(yolox / "tools" / "train.py"),
        "-f",
        str(experiment),
        "-c",
        str(weights),
        "-d",
        str(devices),
        "--fp16",
        "-o",
    ]


def ocr_command(paddleocr: Path, config: Path, pretrained: Path) -> list[str]:
    return [
        "python",
        str(paddleocr / "tools" / "train.py"),
        "-c",
        str(config),
        "-o",
        f"Global.pretrained_model={pretrained}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or run pinned upstream training commands")
    sub = parser.add_subparsers(dest="task", required=True)
    detector = sub.add_parser("detector")
    detector.add_argument("--yolox", type=Path, required=True)
    detector.add_argument("--experiment", type=Path, required=True)
    detector.add_argument("--weights", type=Path, required=True)
    detector.add_argument("--devices", type=int, default=1)
    ocr = sub.add_parser("ocr")
    ocr.add_argument("--paddleocr", type=Path, required=True)
    ocr.add_argument("--config", type=Path, required=True)
    ocr.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.task == "detector":
        command = detector_command(args.yolox, args.experiment, args.weights, args.devices)
    else:
        command = ocr_command(args.paddleocr, args.config, args.pretrained)
    print(subprocess.list2cmdline(command))
    return subprocess.run(command, check=False).returncode if args.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())

