from __future__ import annotations

import re

from gatekeeper.domain.models import Decision, InspectionInput, InspectionState

_CODE_SEPARATORS = re.compile(r"[\s_-]+")
_VALID_CODE = re.compile(r"^[A-Z0-9]+$")


def normalize_code(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _CODE_SEPARATORS.sub("", value.strip().upper())
    return normalized or None


class DecisionEngine:
    """Pure, deterministic decision policy. It never operates hardware."""

    def decide(self, item: InspectionInput) -> Decision:
        expected = normalize_code(item.expected_code)
        problems = frozenset(filter(None, (normalize_code(code) for code in item.problem_codes)))
        recognized = normalize_code(item.ocr_text)

        if expected is None or not _VALID_CODE.fullmatch(expected):
            raise ValueError("expected_code must contain only A-Z and 0-9")
        if expected in problems:
            raise ValueError("expected_code cannot also be a problem code")
        if any(not _VALID_CODE.fullmatch(code) for code in problems):
            raise ValueError("problem_codes must contain only A-Z and 0-9")

        common = dict(
            panel_id=item.panel_id,
            expected_code=expected,
            recognized_code=recognized,
            detector_confidence=item.detector_confidence,
            ocr_confidence=item.ocr_confidence,
        )

        if not item.detected:
            return Decision(
                state=InspectionState.ABNORMAL,
                reason="FPCB ROI not detected",
                **common,
            )
        if item.detector_confidence < item.thresholds.detector_confidence:
            return Decision(
                state=InspectionState.ABNORMAL,
                reason="detector confidence below threshold",
                **common,
            )
        if recognized == expected and item.ocr_confidence >= item.thresholds.normal_ocr_confidence:
            return Decision(
                state=InspectionState.NORMAL,
                reason="recognized code matches expected code",
                **common,
            )
        if recognized in problems and item.ocr_confidence >= item.thresholds.problem_ocr_confidence:
            return Decision(
                state=InspectionState.PROBLEM,
                reason="recognized code matches registered problem code",
                **common,
            )
        if recognized is None:
            reason = "OCR returned no code"
        elif recognized == expected or recognized in problems:
            reason = "OCR confidence below threshold"
        else:
            reason = "recognized code is neither expected nor registered problem code"
        return Decision(state=InspectionState.ABNORMAL, reason=reason, **common)

