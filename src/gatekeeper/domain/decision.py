from __future__ import annotations

import re

from gatekeeper.domain.models import Decision, InspectionInput, InspectionState

_CODE_SEPARATORS = re.compile(r"[\s_-]+")
_VALID_CODE = re.compile(r"^[A-Z0-9]{4}$")


def normalize_code(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _CODE_SEPARATORS.sub("", value.strip().upper())
    return normalized or None


def confusion_key(value: str | None) -> str | None:
    """Treat only the site-approved O/0 OCR confusion as equivalent."""

    return value.replace("O", "0") if value is not None else None


class DecisionEngine:
    """Pure, deterministic decision policy. It never operates hardware."""

    def decide(self, item: InspectionInput) -> Decision:
        expected = normalize_code(item.expected_code)
        problems = frozenset(filter(None, (normalize_code(code) for code in item.problem_codes)))
        recognized = normalize_code(item.ocr_text)
        expected_key = confusion_key(expected)
        problem_keys = {confusion_key(code): code for code in problems}
        recognized_key = confusion_key(recognized)

        if expected is None or not _VALID_CODE.fullmatch(expected):
            raise ValueError("expected_code must contain exactly 4 characters from A-Z and 0-9")
        if expected in problems:
            raise ValueError("expected_code cannot also be a problem code")
        if expected_key in problem_keys:
            raise ValueError("expected_code conflicts with a problem code after O/0 correction")
        if any(not _VALID_CODE.fullmatch(code) for code in problems):
            raise ValueError("problem_codes must contain exactly 4 characters from A-Z and 0-9")

        common = dict(
            panel_id=item.panel_id,
            expected_code=expected,
            recognized_code=recognized,
            detector_confidence=item.detector_confidence,
            ocr_confidence=item.ocr_confidence,
            corrected_code=(
                expected
                if recognized_key == expected_key
                else problem_keys.get(recognized_key)
            ),
        )

        if recognized is not None and not _VALID_CODE.fullmatch(recognized):
            return Decision(
                state=InspectionState.ABNORMAL,
                reason="OCR result is not a four-character code",
                **common,
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
        if (
            recognized_key == expected_key
            and item.ocr_confidence >= item.thresholds.normal_ocr_confidence
        ):
            return Decision(
                state=InspectionState.NORMAL,
                reason=(
                    "recognized code matches expected code"
                    if recognized == expected
                    else "recognized code matches expected code after O/0 correction"
                ),
                **common,
            )
        if (
            recognized_key in problem_keys
            and item.ocr_confidence >= item.thresholds.problem_ocr_confidence
        ):
            return Decision(
                state=InspectionState.PROBLEM,
                reason=(
                    "recognized code matches registered problem code"
                    if recognized in problems
                    else "recognized code matches problem code after O/0 correction"
                ),
                **common,
            )
        if recognized is None:
            reason = "OCR returned no code"
        elif recognized_key == expected_key or recognized_key in problem_keys:
            reason = "OCR confidence below threshold"
        else:
            reason = "recognized code is neither expected nor registered problem code"
        return Decision(state=InspectionState.ABNORMAL, reason=reason, **common)
