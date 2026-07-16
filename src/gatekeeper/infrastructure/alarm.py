from __future__ import annotations

import threading
from enum import StrEnum
from typing import Protocol


class AlarmMode(StrEnum):
    SILENT = "SILENT"
    ABNORMAL = "ABNORMAL"
    PROBLEM = "PROBLEM"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class AlarmController(Protocol):
    def normal(self) -> None: ...
    def abnormal(self) -> None: ...
    def problem(self) -> None: ...
    def system_error(self) -> None: ...
    def mute(self) -> None: ...
    def stop(self) -> None: ...


class WindowsAlarmController:
    """Non-blocking Windows speaker controller with a portable no-op fallback."""

    def __init__(self, repeat_seconds: float = 1.5) -> None:
        self.repeat_seconds = repeat_seconds
        self.mode = AlarmMode.SILENT
        self._muted = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def normal(self) -> None:
        self._set_mode(AlarmMode.SILENT)

    def abnormal(self) -> None:
        self._set_mode(AlarmMode.ABNORMAL)

    def problem(self) -> None:
        self._set_mode(AlarmMode.PROBLEM)

    def system_error(self) -> None:
        self._set_mode(AlarmMode.SYSTEM_ERROR)

    def mute(self) -> None:
        self._muted = True

    def stop(self) -> None:
        self._set_mode(AlarmMode.SILENT)

    def _set_mode(self, mode: AlarmMode) -> None:
        self.mode = mode
        self._muted = False
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        self._stop_event = threading.Event()
        if mode is not AlarmMode.SILENT:
            self._thread = threading.Thread(target=self._run, args=(mode,), daemon=True)
            self._thread.start()

    def _run(self, mode: AlarmMode) -> None:
        try:
            import winsound  # type: ignore[import-not-found]
        except ImportError:
            return
        while not self._stop_event.is_set():
            if not self._muted:
                if mode is AlarmMode.ABNORMAL:
                    winsound.Beep(850, 120)
                    winsound.Beep(850, 120)
                elif mode is AlarmMode.PROBLEM:
                    winsound.Beep(1400, 250)
                    winsound.Beep(900, 250)
                else:
                    winsound.Beep(500, 500)
            self._stop_event.wait(self.repeat_seconds)


class MemoryAlarmController:
    """Deterministic controller for tests and simulation mode."""

    def __init__(self) -> None:
        self.mode = AlarmMode.SILENT
        self.history: list[AlarmMode] = []

    def _set(self, mode: AlarmMode) -> None:
        self.mode = mode
        self.history.append(mode)

    def normal(self) -> None:
        self._set(AlarmMode.SILENT)

    def abnormal(self) -> None:
        self._set(AlarmMode.ABNORMAL)

    def problem(self) -> None:
        self._set(AlarmMode.PROBLEM)

    def system_error(self) -> None:
        self._set(AlarmMode.SYSTEM_ERROR)

    def mute(self) -> None:
        self.history.append(AlarmMode.SILENT)

    def stop(self) -> None:
        self._set(AlarmMode.SILENT)
