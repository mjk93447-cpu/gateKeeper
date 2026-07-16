from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def download(url: str, path: Path) -> str:
    digest = hashlib.sha256()
    with urlopen(url) as response, path.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-url", required=True)
    parser.add_argument("--detector-sha256", required=True)
    parser.add_argument("--checkpoint-url", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)
    detector = args.models / "detector.onnx"
    received = download(args.detector_url, detector)
    if received.lower() != args.detector_sha256.lower():
        raise RuntimeError(f"detector SHA-256 mismatch: {received}")
    checkpoint = args.models / "yolo26s-pcb-pretrained.pt"
    checkpoint_hash = download(args.checkpoint_url, checkpoint)
    if checkpoint_hash.lower() != args.checkpoint_sha256.lower():
        raise RuntimeError(f"training checkpoint SHA-256 mismatch: {checkpoint_hash}")
    ocr = args.models / "ocr"
    if not ocr.is_dir() or not any(ocr.rglob("inference.yml")):
        raise RuntimeError("models/ocr does not contain the OCR pretrained model")
    manifest = {
        "bundle_version": "release",
        "status": "candidate",
        "detector": {
            "architecture": "YOLO26s",
            "input_size": [640, 640],
            "classes": ["fpcb_surface", "code_roi", "defect_roi"],
            "path": "detector.onnx",
            "sha256": received,
            "pretraining": "betty0/pcb-defect-detection",
            "training_checkpoint": {
                "path": checkpoint.name,
                "sha256": checkpoint_hash,
            },
        },
        "ocr": {
            "architecture": "en_PP-OCRv4_mobile_rec",
            "characters": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "path": "ocr",
            "relative_roi": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
        },
    }
    (args.models / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
