from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

Box = tuple[float, float, float, float]


def draw_sample(
    code: str, rng: random.Random, size: int = 640
) -> tuple[np.ndarray, tuple[Box, Box]]:
    image = np.full((size, size), rng.randint(150, 220), dtype=np.uint8)
    board_x1, board_y1 = 45, 45
    board_x2, board_y2 = size - 45, size - 45
    cv2.rectangle(image, (board_x1, board_y1), (board_x2, board_y2), rng.randint(70, 120), -1)
    for x in range(board_x1 + 20, board_x2 - 10, 32):
        cv2.line(image, (x, board_y1 + 10), (x, board_y2 - 10), rng.randint(90, 150), 1)
    for y in range(board_y1 + 20, board_y2 - 10, 32):
        cv2.line(image, (board_x1 + 10, y), (board_x2 - 10, y), rng.randint(90, 150), 1)

    roi_w = rng.randint(170, 260)
    roi_h = rng.randint(48, 72)
    roi_x1 = rng.randint(board_x1 + 35, board_x2 - roi_w - 35)
    roi_y1 = rng.randint(board_y1 + 35, board_y2 - roi_h - 35)
    roi_x2, roi_y2 = roi_x1 + roi_w, roi_y1 + roi_h
    cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), rng.randint(205, 245), -1)
    font_scale = rng.uniform(1.15, 1.55)
    thickness = rng.randint(2, 4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(code, font, font_scale, thickness)
    text_x = roi_x1 + max(4, (roi_w - text_w) // 2)
    text_y = roi_y1 + max(text_h + 2, (roi_h + text_h) // 2)
    cv2.putText(
        image,
        code,
        (text_x, text_y),
        font,
        font_scale,
        rng.randint(0, 45),
        thickness,
        cv2.LINE_AA,
    )
    if rng.random() < 0.35:
        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.2, 0.7))
    noise = np.random.default_rng(rng.randrange(1_000_000)).normal(0, 3, image.shape)
    image = np.uint8(np.clip(image.astype(np.float32) + noise, 0, 255))
    fpcb_box = ((board_x1 + board_x2) / 2 / size, (board_y1 + board_y2) / 2 / size,
                (board_x2 - board_x1) / size, (board_y2 - board_y1) / size)
    code_box = ((roi_x1 + roi_x2) / 2 / size, (roi_y1 + roi_y2) / 2 / size,
                (roi_x2 - roi_x1) / size, (roi_y2 - roi_y1) / size)
    return image, (fpcb_box, code_box)


def write_split(root: Path, split: str, count: int, seed: int, codes: list[str]) -> None:
    rng = random.Random(seed)
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    ocr_file = root / "ocr" / f"{split}.txt"
    ocr_file.parent.mkdir(parents=True, exist_ok=True)
    with ocr_file.open("w", encoding="utf-8") as stream:
        for index in range(count):
            code = codes[index % len(codes)]
            image, (fpcb_box, code_box) = draw_sample(code, rng)
            name = f"{split}_{index:05d}"
            image_path = image_dir / f"{name}.png"
            label_path = label_dir / f"{name}.txt"
            cv2.imwrite(str(image_path), image)
            with label_path.open("w", encoding="utf-8") as labels:
                labels.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*fpcb_box))
                labels.write("1 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*code_box))
            stream.write(f"../images/{split}/{image_path.name}\t{code}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate development-only FPCB YOLO/OCR data")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/fpcb-yolo"))
    parser.add_argument("--train", type=int, default=180)
    parser.add_argument("--val", type=int, default=40)
    parser.add_argument("--test", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    codes = ["HJ04", "HJ05"]
    write_split(args.output, "train", args.train, args.seed, codes)
    write_split(args.output, "val", args.val, args.seed + 1000, codes)
    write_split(args.output, "test", args.test, args.seed + 9000, codes)
    yaml = args.output / "data.yaml"
    try:
        dataset_root = args.output.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        dataset_root = args.output.resolve().as_posix()
    yaml.write_text(
        f"path: {dataset_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: fpcb_surface\n"
        "  1: code_roi\n",
        encoding="utf-8",
    )
    print(f"generated development-only dataset at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
