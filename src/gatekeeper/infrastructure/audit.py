from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from gatekeeper.application.inspection import InspectionService
from gatekeeper.domain.models import Decision


class JsonlDecisionSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def append(self, decision: Decision) -> None:
        event = InspectionService.as_event(decision)
        self.append_event("decision", event)

    def append_event(self, event_type: str, payload: dict[str, object]) -> None:
        event = {
            "event_type": event_type,
            "event_at": datetime.now(UTC).isoformat(),
            **payload,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
