from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RelativeRoi:
    """An OCR region expressed relative to the YOLO detection box."""

    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("relative ROI width and height must be positive")
        if min(self.x, self.y) < 0 or self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("relative ROI must be contained in the YOLO box")

    def apply(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        return (
            x1 + round(self.x * width),
            y1 + round(self.y * height),
            x1 + round((self.x + self.width) * width),
            y1 + round((self.y + self.height) * height),
        )
