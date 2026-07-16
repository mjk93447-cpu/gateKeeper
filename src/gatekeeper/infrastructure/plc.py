from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from gatekeeper.domain.models import InspectionState


class SignalStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    DUPLICATE = "DUPLICATE"


@dataclass(frozen=True, slots=True)
class ProblemEvent:
    sequence_id: int
    panel_id: str
    reason: str
    requested_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class SignalResult:
    status: SignalStatus
    sequence_id: int


@dataclass(slots=True)
class SimulatedOutputPort:
    """Records requested actions but never communicates with equipment."""

    last_state: InspectionState | None = None
    last_applied_at: datetime | None = None
    history: list[InspectionState] = field(default_factory=list)
    reject_requests: list[ProblemEvent] = field(default_factory=list)
    _reject_sequences: set[int] = field(default_factory=set)

    def apply(self, state: InspectionState) -> None:
        self.last_state = state
        self.last_applied_at = datetime.now(UTC)
        self.history.append(state)

    def request_problem(self, event: ProblemEvent) -> SignalResult:
        if event.sequence_id in self._reject_sequences:
            return SignalResult(SignalStatus.DUPLICATE, event.sequence_id)
        self._reject_sequences.add(event.sequence_id)
        self.reject_requests.append(event)
        return SignalResult(SignalStatus.ACCEPTED, event.sequence_id)

    def heartbeat(self) -> dict[str, object]:
        return {"alive": True, "at": datetime.now(UTC).isoformat(), "driver": "simulation"}


ProtocolNeutralPlcPort = SimulatedOutputPort
