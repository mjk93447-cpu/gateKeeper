from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


class InspectionState(StrEnum):
    NORMAL = "NORMAL"
    ABNORMAL = "ABNORMAL"
    PROBLEM = "PROBLEM"


@dataclass(frozen=True, slots=True)
class Thresholds:
    detector_confidence: float = 0.70
    normal_ocr_confidence: float = 0.90
    problem_ocr_confidence: float = 0.90

    def __post_init__(self) -> None:
        for name, value in (
            ("detector_confidence", self.detector_confidence),
            ("normal_ocr_confidence", self.normal_ocr_confidence),
            ("problem_ocr_confidence", self.problem_ocr_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class InspectionInput:
    expected_code: str
    problem_codes: frozenset[str]
    detected: bool
    detector_confidence: float
    ocr_text: str | None
    ocr_confidence: float
    thresholds: Thresholds = field(default_factory=Thresholds)
    panel_id: str = field(default_factory=lambda: str(uuid4()))
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class Decision:
    state: InspectionState
    panel_id: str
    expected_code: str
    recognized_code: str | None
    detector_confidence: float
    ocr_confidence: float
    reason: str
    decided_at: datetime = field(default_factory=lambda: datetime.now(UTC))

