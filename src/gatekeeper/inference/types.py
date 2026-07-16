from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class DetectionResult:
    detections: tuple[Detection, ...]

    @property
    def best(self) -> Detection | None:
        return max(self.detections, key=lambda item: item.confidence, default=None)


@dataclass(frozen=True, slots=True)
class OcrResult:
    text: str | None
    confidence: float
