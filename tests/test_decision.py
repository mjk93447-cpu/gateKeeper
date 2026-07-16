from __future__ import annotations

import pytest

from gatekeeper.domain import DecisionEngine, InspectionInput, InspectionState, Thresholds


def make_input(**changes: object) -> InspectionInput:
    values: dict[str, object] = {
        "expected_code": "HJ04",
        "problem_codes": frozenset({"HJ05"}),
        "detected": True,
        "detector_confidence": 0.95,
        "ocr_text": "HJ04",
        "ocr_confidence": 0.98,
        "thresholds": Thresholds(),
    }
    values.update(changes)
    return InspectionInput(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["HJ04", "hj04", " HJ-04 ", "HJ_04"])
def test_expected_code_is_normal_after_safe_normalization(text: str) -> None:
    result = DecisionEngine().decide(make_input(ocr_text=text))
    assert result.state is InspectionState.NORMAL
    assert result.recognized_code == "HJ04"


def test_registered_problem_code_is_problem() -> None:
    result = DecisionEngine().decide(make_input(ocr_text="HJ05"))
    assert result.state is InspectionState.PROBLEM


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"detected": False}, "FPCB ROI not detected"),
        ({"detector_confidence": 0.4}, "detector confidence below threshold"),
        ({"ocr_text": "HJ06"}, "neither expected nor registered"),
        ({"ocr_text": None}, "OCR returned no code"),
        ({"ocr_text": "HJ05", "ocr_confidence": 0.5}, "OCR confidence below threshold"),
    ],
)
def test_unsafe_or_unknown_inputs_are_abnormal(changes: dict[str, object], reason: str) -> None:
    result = DecisionEngine().decide(make_input(**changes))
    assert result.state is InspectionState.ABNORMAL
    assert reason in result.reason


def test_normal_code_cannot_be_registered_as_problem() -> None:
    with pytest.raises(ValueError, match="cannot also be"):
        DecisionEngine().decide(make_input(problem_codes=frozenset({"HJ04"})))


def test_unapproved_problem_code_fuzzy_match_is_rejected() -> None:
    result = DecisionEngine().decide(make_input(ocr_text="HJ15"))
    assert result.state is InspectionState.ABNORMAL


@pytest.mark.parametrize(
    ("ocr_text", "expected_state", "corrected"),
    [("HJO4", InspectionState.NORMAL, "HJ04"), ("HJO5", InspectionState.PROBLEM, "HJ05")],
)
def test_site_approved_o_zero_correction(
    ocr_text: str, expected_state: InspectionState, corrected: str
) -> None:
    result = DecisionEngine().decide(make_input(ocr_text=ocr_text))
    assert result.state is expected_state
    assert result.corrected_code == corrected


def test_non_o_zero_fuzzy_correction_remains_disabled() -> None:
    result = DecisionEngine().decide(make_input(ocr_text="HJO6"))
    assert result.state is InspectionState.ABNORMAL
