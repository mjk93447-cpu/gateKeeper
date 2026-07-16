from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


class InspectionState(StrEnum):
    NORMAL = "NORMAL"
    ABNORMAL = "ABNORMAL"
    PROBLEM = "PROBLEM"


class DisplayState(StrEnum):
    """State presented to the operator, including infrastructure failures."""

    NORMAL = "NORMAL"
    ABNORMAL = "ABNORMAL"
    PROBLEM = "PROBLEM"
    SYSTEM_ERROR = "SYSTEM_ERROR"


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
    corrected_code: str | None
    reason: str
    decided_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class InspectionResult:
    """Immutable result passed from the worker to UI, alarms and PLC adapters."""

    panel_id: str
    sequence_id: int
    state: DisplayState
    expected_code: str
    recognized_code: str | None
    detector_confidence: float
    ocr_confidence: float
    image_path: str
    model_version: str
    latency_ms: float
    reason: str
    roi_box: tuple[int, int, int, int] | None = None
    corrected_code: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
