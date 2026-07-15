from __future__ import annotations

import json

from gatekeeper.domain import DecisionEngine, InspectionInput, InspectionState
from gatekeeper.infrastructure.audit import JsonlDecisionSink


def test_jsonl_sink_writes_one_parseable_event(tmp_path) -> None:
    decision = DecisionEngine().decide(
        InspectionInput(
            expected_code="HJ04",
            problem_codes=frozenset({"HJ05"}),
            detected=True,
            detector_confidence=0.99,
            ocr_text="HJ04",
            ocr_confidence=0.99,
        )
    )
    target = tmp_path / "audit.jsonl"
    JsonlDecisionSink(target).append(decision)
    event = json.loads(target.read_text(encoding="utf-8"))
    assert event["state"] == InspectionState.NORMAL.value
    assert event["recognized_code"] == "HJ04"

