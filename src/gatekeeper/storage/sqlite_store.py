from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any


class SQLiteEventStore:
    """Small WAL-backed event store for audit, training and UI metrics."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    sequence_id INTEGER,
                    panel_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_time ON events(occurred_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)"
            )

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        sequence_id: int | None = None,
        panel_id: str | None = None,
    ) -> int:
        occurred_at = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO events(event_type, occurred_at, sequence_id, panel_id, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    occurred_at,
                    sequence_id,
                    panel_id,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT event_type, occurred_at, sequence_id, panel_id, payload_json "
                "FROM events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "event_type": row[0],
                "occurred_at": row[1],
                "sequence_id": row[2],
                "panel_id": row[3],
                "payload": json.loads(row[4]),
            }
            for row in rows
        ]
