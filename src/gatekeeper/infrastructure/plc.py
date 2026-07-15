from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from gatekeeper.domain.models import InspectionState


@dataclass(slots=True)
class SimulatedOutputPort:
    """Records requested actions but never communicates with equipment."""

    last_state: InspectionState | None = None
    last_applied_at: datetime | None = None
    history: list[InspectionState] = field(default_factory=list)

    def apply(self, state: InspectionState) -> None:
        self.last_state = state
        self.last_applied_at = datetime.now(UTC)
        self.history.append(state)

