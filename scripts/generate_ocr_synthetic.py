from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def render_code(code: str, size: tuple[int, int], rng: random.Random) -> Image.Image:
    width, height = size
    background = Image.new("L", size, color=rng.randint(150, 235))
    draw = ImageDraw.Draw(background)
    for _ in range(8):
        y = rng.randrange(height)
        draw.line((0, y, width, y), fill=rng.randint(90, 180), width=1)
    font = ImageFont.load_default()
    bounds = draw.textbbox((0, 0), code, font=font)
    x = (width - (bounds[2] - bounds[0])) // 2 + rng.randint(-2, 2)
    y = (height - (bounds[3] - bounds[1])) // 2 + rng.randint(-2, 2)
    draw.text((x, y), code, fill=rng.randint(0, 55), font=font)
    if rng.random() < 0.35:
        background = background.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
    array = np.asarray(background, dtype=np.int16)
    noise = np.random.default_rng(rng.randrange(1_000_000)).normal(0, 5, array.shape)
    return Image.fromarray(np.uint8(np.clip(array + noise, 0, 255)), mode="L")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate development-only four-character OCR data"
    )
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/ocr"))
    parser.add_argument("--codes", nargs="+", default=["HJ04", "HJ05"])
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(len(code) != 4 or not code.isalnum() or code.upper() != code for code in args.codes):
        raise SystemExit("codes must be four uppercase A-Z/0-9 characters")
    rng = random.Random(args.seed)
    images = args.output / "images"
    images.mkdir(parents=True, exist_ok=True)
    labels = args.output / "labels.txt"
    with labels.open("w", encoding="utf-8") as stream:
        for index in range(args.count):
            code = args.codes[index % len(args.codes)]
            image_path = images / f"synthetic_{index:06d}.png"
            render_code(code, (160, 48), rng).save(image_path)
            stream.write(f"images/{image_path.name}\t{code}\n")
    print(f"generated {args.count} development-only OCR samples in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
