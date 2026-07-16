from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    """Run CPU-only detector fine-tuning from the offline training executable."""

    parser = argparse.ArgumentParser(description="gateKeeper CPU YOLO26 training runner")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"dataset YAML not found: {args.data}")
    if not args.pretrained.is_file():
        raise SystemExit(f"pretrained checkpoint not found: {args.pretrained}")
    from ultralytics import YOLO

    model = YOLO(str(args.pretrained))
    model.train(
        data=str(args.data),
        imgsz=args.image_size,
        epochs=args.epochs,
        batch=args.batch,
        device="cpu",
        workers=args.workers,
        patience=args.patience,
        amp=False,
        cache="disk",
        seed=args.seed,
        project=str(args.output),
        name="yolo26s_fpcb_cpu",
        exist_ok=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
