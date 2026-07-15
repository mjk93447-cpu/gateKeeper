from __future__ import annotations

from dataclasses import asdict
from typing import Protocol

from gatekeeper.domain.decision import DecisionEngine
from gatekeeper.domain.models import Decision, InspectionInput, InspectionState


class DecisionSink(Protocol):
    def append(self, decision: Decision) -> None: ...


class OutputPort(Protocol):
    def apply(self, state: InspectionState) -> None: ...


class InspectionService:
    def __init__(
        self,
        engine: DecisionEngine,
        sink: DecisionSink | None = None,
        output: OutputPort | None = None,
    ) -> None:
        self._engine = engine
        self._sink = sink
        self._output = output

    def inspect(self, item: InspectionInput) -> Decision:
        decision = self._engine.decide(item)
        if self._sink is not None:
            self._sink.append(decision)
        if self._output is not None:
            self._output.apply(decision.state)
        return decision

    @staticmethod
    def as_event(decision: Decision) -> dict[str, object]:
        event = asdict(decision)
        event["state"] = decision.state.value
        event["decided_at"] = decision.decided_at.isoformat()
        return event

