from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from gatekeeper.training.cpu_runner import CpuTrainingConfig, build_yolo26_command


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
    yolo26 = sub.add_parser("yolo26-cpu")
    yolo26.add_argument("--data", type=Path, required=True)
    yolo26.add_argument("--pretrained", type=Path, required=True)
    yolo26.add_argument("--output", type=Path, default=Path("runs/gatekeeper"))
    yolo26.add_argument("--imgsz", type=int, default=640)
    yolo26.add_argument("--epochs", type=int, default=100)
    yolo26.add_argument("--batch", type=int, default=2)
    yolo26.add_argument("--patience", type=int, default=20)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.task == "detector":
        command = detector_command(args.yolox, args.experiment, args.weights, args.devices)
    elif args.task == "ocr":
        command = ocr_command(args.paddleocr, args.config, args.pretrained)
    else:
        command = build_yolo26_command(
            CpuTrainingConfig(
                data_yaml=args.data,
                pretrained=args.pretrained,
                output_dir=args.output,
                image_size=args.imgsz,
                epochs=args.epochs,
                batch=args.batch,
                patience=args.patience,
            )
        )
    print(subprocess.list2cmdline(command))
    return subprocess.run(command, check=False).returncode if args.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())
