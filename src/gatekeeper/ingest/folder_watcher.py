from __future__ import annotations

import hashlib
import shutil
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class FolderWatcher:
    """Polling hot-folder watcher with write-stability and duplicate protection."""

    def __init__(
        self,
        directory: str | Path,
        on_file: Callable[[Path, int], None],
        *,
        on_duplicate: Callable[[Path], None] | None = None,
        poll_seconds: float = 0.25,
        stable_checks: int = 2,
        suffixes: Iterable[str] = IMAGE_SUFFIXES,
    ) -> None:
        self.directory = Path(directory)
        self.on_file = on_file
        self.on_duplicate = on_duplicate
        self.poll_seconds = poll_seconds
        self.stable_checks = max(1, stable_checks)
        self.suffixes = frozenset(s.lower() for s in suffixes)
        self._seen: set[str] = set()
        self._duplicate_notified: set[Path] = set()
        self._sequence = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="gatekeeper-folder")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None

    def scan_once(self) -> int:
        processed = 0
        for path in sorted(self.directory.iterdir(), key=lambda item: item.stat().st_mtime_ns):
            if not path.is_file() or path.suffix.lower() not in self.suffixes:
                continue
            try:
                digest = file_sha256(path)
                if digest in self._seen:
                    if self.on_duplicate is not None and path not in self._duplicate_notified:
                        self._duplicate_notified.add(path)
                        self.on_duplicate(path)
                    continue
                if not self._is_stable(path):
                    continue
                self._seen.add(digest)
                self._sequence += 1
                self.on_file(path, self._sequence)
                processed += 1
            except (OSError, PermissionError):
                continue
        return processed

    def _is_stable(self, path: Path) -> bool:
        previous: tuple[int, int] | None = None
        stable = 0
        for _ in range(self.stable_checks + 1):
            try:
                stat = path.stat()
            except OSError:
                return False
            current = (stat.st_size, stat.st_mtime_ns)
            if current == previous:
                stable += 1
            else:
                stable = 0
            previous = current
            if stable >= self.stable_checks:
                return True
            time.sleep(min(self.poll_seconds, 0.25))
        return False

    def _run(self) -> None:
        while not self._stop.is_set():
            self.scan_once()
            self._stop.wait(self.poll_seconds)


class RapidFolderWatcher:
    """Non-blocking final-file watcher for camera hot-folders.

    A camera may create a final JPG/PNG directly.  This watcher waits for the
    size and mtime to remain unchanged for ``settle_ms`` without sleeping per
    file, then dispatches it once.  It deliberately does not hash frames: the
    Frame Gate owns panel-session suppression before expensive work begins.
    """

    def __init__(
        self,
        directory: str | Path,
        on_file: Callable[[Path, int], None],
        *,
        on_not_ready: Callable[[Path], None] | None = None,
        poll_seconds: float = 0.01,
        settle_ms: int = 30,
        suffixes: Iterable[str] = IMAGE_SUFFIXES,
    ) -> None:
        self.directory = Path(directory)
        self.on_file = on_file
        self.on_not_ready = on_not_ready
        self.poll_seconds = max(0.005, poll_seconds)
        self.settle_seconds = max(0.001, settle_ms / 1000)
        self.suffixes = frozenset(s.lower() for s in suffixes)
        self._observed: dict[Path, tuple[int, int, float]] = {}
        self._dispatched: set[Path] = set()
        self._not_ready_notified: set[Path] = set()
        self._sequence = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="gatekeeper-rapid-folder"
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None

    def scan_once(self) -> int:
        now = time.monotonic()
        dispatched = 0
        live: set[Path] = set()
        try:
            paths = sorted(self.directory.iterdir(), key=lambda item: item.stat().st_mtime_ns)
        except OSError:
            return 0
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in self.suffixes:
                continue
            live.add(path)
            try:
                stat = path.stat()
            except OSError:
                continue
            marker = (stat.st_size, stat.st_mtime_ns)
            previous = self._observed.get(path)
            if previous is None or previous[:2] != marker:
                self._observed[path] = (*marker, now)
                if self.on_not_ready is not None and path not in self._not_ready_notified:
                    self._not_ready_notified.add(path)
                    self.on_not_ready(path)
                continue
            if path in self._dispatched or now - previous[2] < self.settle_seconds:
                continue
            self._dispatched.add(path)
            self._sequence += 1
            self.on_file(path, self._sequence)
            dispatched += 1
        disappeared = set(self._observed) - live
        for path in disappeared:
            self._observed.pop(path, None)
            self._dispatched.discard(path)
            self._not_ready_notified.discard(path)
        return dispatched

    def _run(self) -> None:
        while not self._stop.is_set():
            self.scan_once()
            self._stop.wait(self.poll_seconds)


class ArchiveManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def move(self, source: Path, state: str) -> Path:
        destination_dir = self.root / state.lower()
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        if destination.exists():
            destination = destination_dir / f"{source.stem}_{time.time_ns()}{source.suffix}"
        shutil.move(str(source), str(destination))
        return destination
