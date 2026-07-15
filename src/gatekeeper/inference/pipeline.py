from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from gatekeeper.domain import (
    DecisionEngine,
    DisplayState,
    InspectionInput,
    InspectionResult,
    Thresholds,
)
from gatekeeper.inference.roi import RelativeRoi
from gatekeeper.inference.types import DetectionResult, OcrResult


class InspectionPipeline:
    def __init__(
        self,
        detector: object,
        ocr: object,
        *,
        expected_code: str,
        problem_codes: frozenset[str],
        thresholds: Thresholds | None = None,
        model_version: str = "unknown",
        ocr_relative_roi: RelativeRoi | None = None,
    ) -> None:
        self.detector = detector
        self.ocr = ocr
        self.expected_code = expected_code
        self.problem_codes = problem_codes
        self.thresholds = thresholds or Thresholds()
        self.model_version = model_version
        self.ocr_relative_roi = ocr_relative_roi or RelativeRoi()
        self.engine = DecisionEngine()

    def inspect(self, image_path: str | Path, sequence_id: int) -> InspectionResult:
        started = time.perf_counter()
        path = Path(image_path)
        try:
            import cv2  # type: ignore[import-not-found]

            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"unable to read image: {path}")
            detected: DetectionResult = self.detector.detect(image)
            candidate = next(
                (item for item in detected.detections if item.label == "code_roi"),
                detected.best,
            )
            if candidate is None:
                return self._result(
                    path,
                    sequence_id,
                    DisplayState.ABNORMAL,
                    None,
                    0.0,
                    "FPCB/code ROI not detected",
                    started,
                )
            x1, y1, x2, y2 = self.ocr_relative_roi.apply(candidate.box)
            x1 = max(0, min(x1, image.shape[1]))
            y1 = max(0, min(y1, image.shape[0]))
            x2 = max(x1 + 1, min(x2, image.shape[1]))
            y2 = max(y1 + 1, min(y2, image.shape[0]))
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                return self._result(
                    path, sequence_id, DisplayState.ABNORMAL, None, candidate.confidence,
                    "code ROI is empty", started
                )
            recognized: OcrResult = self.ocr.recognize(roi)
            decision = self.engine.decide(
                InspectionInput(
                    expected_code=self.expected_code,
                    problem_codes=self.problem_codes,
                    detected=True,
                    detector_confidence=candidate.confidence,
                    ocr_text=recognized.text,
                    ocr_confidence=recognized.confidence,
                    thresholds=self.thresholds,
                    panel_id=path.stem or str(uuid4()),
                )
            )
            return self._result(
                path,
                sequence_id,
                DisplayState(decision.state.value),
                decision.recognized_code,
                decision.detector_confidence,
                decision.reason,
                started,
                ocr_confidence=decision.ocr_confidence,
                roi_box=(x1, y1, x2, y2),
                corrected_code=decision.corrected_code,
            )
        except Exception as exc:  # pipeline errors are visible system errors, never NORMAL
            return self._result(
                path, sequence_id, DisplayState.SYSTEM_ERROR, None, 0.0, str(exc), started
            )

    def _result(
        self,
        path: Path,
        sequence_id: int,
        state: DisplayState,
        code: str | None,
        detector_confidence: float,
        reason: str,
        started: float,
        *,
        ocr_confidence: float = 0.0,
        roi_box: tuple[int, int, int, int] | None = None,
        corrected_code: str | None = None,
    ) -> InspectionResult:
        return InspectionResult(
            panel_id=path.stem or str(uuid4()),
            sequence_id=sequence_id,
            state=state,
            expected_code=self.expected_code,
            recognized_code=code,
            detector_confidence=detector_confidence,
            ocr_confidence=ocr_confidence,
            image_path=str(path),
            model_version=self.model_version,
            latency_ms=(time.perf_counter() - started) * 1000,
            reason=reason,
            roi_box=roi_box,
            corrected_code=corrected_code,
        )
