from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np


class FrameGateState(StrEnum):
    EMPTY = "EMPTY"
    PRESENT_MOVING = "PRESENT_MOVING"
    STABLE = "STABLE"
    INSPECTION_LOCK = "INSPECTION_LOCK"
    EMPTY_REARMED = "EMPTY_REARMED"


class FrameDisposition(StrEnum):
    EMPTY = "frame_empty"
    MOVING = "frame_moving"
    STABLE = "frame_stable"
    SELECTED = "frame_selected"
    SUPPRESSED = "frame_suppressed"
    EARLY_PANEL = "unexpected_early_panel"
    REARMED = "panel_rearmed"
    UNCALIBRATED = "frame_uncalibrated"
    CALIBRATED = "background_captured"


@dataclass(frozen=True, slots=True)
class RelativeBox:
    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("relative box must stay inside the image")

    @classmethod
    def from_mapping(cls, value: object) -> RelativeBox:
        if not isinstance(value, dict):
            return cls()
        return cls(
            x=float(value.get("x", 0.0)),
            y=float(value.get("y", 0.0)),
            width=float(value.get("width", 1.0)),
            height=float(value.get("height", 1.0)),
        )

    def to_mapping(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True, slots=True)
class FrameGateSettings:
    enabled: bool = True
    expected_camera_interval_ms: int = 150
    refractory_period_ms: int = 2000
    stable_frames_required: int = 2
    empty_frames_to_rearm: int = 2
    file_settle_ms: int = 30
    queue_depth: int = 3
    presence_threshold: float = 0.08
    motion_threshold: float = 0.025
    sharpness_threshold: float = 20.0
    presence_roi: RelativeBox = field(default_factory=RelativeBox)
    sharpness_roi: RelativeBox = field(default_factory=RelativeBox)
    diagnostic_sample_every: int = 0
    preview_max_width: int = 160

    def __post_init__(self) -> None:
        if not 50 <= self.expected_camera_interval_ms <= 1000:
            raise ValueError("expected_camera_interval_ms must be between 50 and 1000")
        if not 0 <= self.refractory_period_ms <= 60000:
            raise ValueError("refractory_period_ms must be between 0 and 60000")
        if not 1 <= self.stable_frames_required <= 10:
            raise ValueError("stable_frames_required must be between 1 and 10")
        if not 1 <= self.empty_frames_to_rearm <= 20:
            raise ValueError("empty_frames_to_rearm must be between 1 and 20")
        if not 1 <= self.file_settle_ms <= 1000:
            raise ValueError("file_settle_ms must be between 1 and 1000")
        if not 1 <= self.queue_depth <= 100:
            raise ValueError("queue_depth must be between 1 and 100")
        if not 0.0 <= self.presence_threshold <= 1.0:
            raise ValueError("presence_threshold must be between 0 and 1")
        if not 0.0 <= self.motion_threshold <= 1.0:
            raise ValueError("motion_threshold must be between 0 and 1")
        if self.sharpness_threshold < 0:
            raise ValueError("sharpness_threshold must be non-negative")
        if self.diagnostic_sample_every < 0:
            raise ValueError("diagnostic_sample_every must be non-negative")

    @classmethod
    def from_mapping(cls, value: object) -> FrameGateSettings:
        if not isinstance(value, dict):
            return cls()
        return cls(
            enabled=bool(value.get("enabled", True)),
            expected_camera_interval_ms=int(value.get("expected_camera_interval_ms", 150)),
            refractory_period_ms=int(value.get("refractory_period_ms", 2000)),
            stable_frames_required=int(value.get("stable_frames_required", 2)),
            empty_frames_to_rearm=int(value.get("empty_frames_to_rearm", 2)),
            file_settle_ms=int(value.get("file_settle_ms", 30)),
            queue_depth=int(value.get("queue_depth", 3)),
            presence_threshold=float(value.get("presence_threshold", 0.08)),
            motion_threshold=float(value.get("motion_threshold", 0.025)),
            sharpness_threshold=float(value.get("sharpness_threshold", 20.0)),
            presence_roi=RelativeBox.from_mapping(value.get("presence_roi")),
            sharpness_roi=RelativeBox.from_mapping(value.get("sharpness_roi")),
            diagnostic_sample_every=int(value.get("diagnostic_sample_every", 0)),
            preview_max_width=int(value.get("preview_max_width", 160)),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "expected_camera_interval_ms": self.expected_camera_interval_ms,
            "refractory_period_ms": self.refractory_period_ms,
            "stable_frames_required": self.stable_frames_required,
            "empty_frames_to_rearm": self.empty_frames_to_rearm,
            "file_settle_ms": self.file_settle_ms,
            "queue_depth": self.queue_depth,
            "presence_threshold": self.presence_threshold,
            "motion_threshold": self.motion_threshold,
            "sharpness_threshold": self.sharpness_threshold,
            "presence_roi": self.presence_roi.to_mapping(),
            "sharpness_roi": self.sharpness_roi.to_mapping(),
            "diagnostic_sample_every": self.diagnostic_sample_every,
            "preview_max_width": self.preview_max_width,
        }


@dataclass(frozen=True, slots=True)
class FrameGateOutcome:
    disposition: FrameDisposition
    state: FrameGateState
    path: Path
    sequence_id: int
    selected_path: Path | None = None
    selected_sequence_id: int | None = None
    discard_paths: tuple[Path, ...] = ()
    presence_score: float = 0.0
    motion_score: float | None = None
    sharpness_score: float = 0.0
    reason: str = ""
    latency_ms: float = 0.0


@dataclass(slots=True)
class _Candidate:
    path: Path
    sequence_id: int
    sharpness: float


class FrameGate:
    """Cheap image-only panel session gate placed before expensive inference."""

    def __init__(self, settings: FrameGateSettings, background_path: str | Path) -> None:
        self.settings = settings
        self.background_path = Path(background_path)
        self.state = FrameGateState.EMPTY
        self._background: np.ndarray | None = self._load_background()
        self._capture_background = False
        self._previous_presence: np.ndarray | None = None
        self._previous_was_present = False
        self._stable_count = 0
        self._held: list[_Candidate] = []
        self._empty_count = 0
        self._empty_seen_in_lock = False
        self._refractory_until = 0.0
        self._selected_at: float | None = None
        self._first_empty_at: float | None = None
        self._last_received_at: float | None = None
        self._intervals: deque[float] = deque(maxlen=100)
        self._selection_intervals: deque[float] = deque(maxlen=50)
        self._clear_times: deque[float] = deque(maxlen=50)
        self._last_selected_at: float | None = None
        self._latencies: deque[float] = deque(maxlen=500)
        self._counts: dict[str, int] = {}

    @property
    def calibrated(self) -> bool:
        return self._background is not None

    def request_background_capture(self) -> None:
        self._capture_background = True

    def metrics(self) -> dict[str, object]:
        intervals = sorted(self._intervals)
        latencies = sorted(self._latencies)
        median_interval = intervals[len(intervals) // 2] if intervals else 0.0
        p50_latency = latencies[len(latencies) // 2] if latencies else 0.0
        p95_latency = latencies[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0
        warning = ""
        if median_interval and median_interval > 0.151:
            warning = (
                "Frame interval is above 150 ms; two stable frames are not guaranteed "
                "in a 300 ms dwell."
            )
        expected_interval = self.settings.expected_camera_interval_ms / 1000
        allowed_interval_error = max(0.015, expected_interval * 0.1)
        if median_interval and abs(median_interval - expected_interval) > allowed_interval_error:
            warning = (
                "Observed camera interval differs materially from the configured line setting."
            )
        if p95_latency > 20.0:
            warning = "Frame Gate p95 latency exceeds the 20 ms target on this PC."
        if self.settings.refractory_period_ms < self._observed_clear_ms():
            warning = "Configured refractory period is shorter than observed panel-clear time."
        if self._selection_intervals and self.settings.refractory_period_ms >= int(
            min(self._selection_intervals) * 900
        ):
            warning = "Configured refractory period is near or above the observed panel interval."
        return {
            "state": self.state.value,
            "calibrated": self.calibrated,
            "live_frame_rate_fps": round(1.0 / median_interval, 2) if median_interval else 0.0,
            "median_frame_interval_ms": round(median_interval * 1000, 1),
            "p50_gate_latency_ms": round(p50_latency, 1),
            "p95_gate_latency_ms": round(p95_latency, 1),
            "counts": dict(self._counts),
            "warning": warning,
        }

    def evaluate(
        self, path: str | Path, sequence_id: int, *, now: float | None = None
    ) -> FrameGateOutcome:
        started = time.perf_counter()
        captured_at = time.monotonic() if now is None else now
        if self._last_received_at is not None:
            self._intervals.append(max(0.0, captured_at - self._last_received_at))
        self._last_received_at = captured_at
        image_path = Path(path)
        try:
            frame = self._load_preview(image_path)
        except ValueError as exc:
            return self._outcome(
                FrameDisposition.SUPPRESSED,
                image_path,
                sequence_id,
                reason=f"frame cannot be read: {exc}",
                started=started,
            )
        if self._capture_background:
            self._background = frame.copy()
            self.background_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.background_path, self._background)
            self._capture_background = False
            self._reset_panel_tracking()
            return self._outcome(
                FrameDisposition.CALIBRATED,
                image_path,
                sequence_id,
                reason="empty background captured",
                started=started,
            )
        if not self.settings.enabled:
            return self._outcome(
                FrameDisposition.SELECTED,
                image_path,
                sequence_id,
                selected_path=image_path,
                selected_sequence_id=sequence_id,
                reason="frame gate disabled",
                started=started,
            )
        if self._background is None:
            return self._outcome(
                FrameDisposition.UNCALIBRATED,
                image_path,
                sequence_id,
                reason="capture an empty background before monitoring",
                started=started,
            )

        presence_frame = self._crop(frame, self.settings.presence_roi)
        background_presence = self._crop(self._background, self.settings.presence_roi)
        presence = self._difference(presence_frame, background_presence)
        sharpness = self._sharpness(self._crop(frame, self.settings.sharpness_roi))
        present = presence >= self.settings.presence_threshold
        previous = self._previous_presence
        current_presence = presence_frame
        motion = (
            self._difference(current_presence, previous)
            if self._previous_was_present and previous is not None and present
            else None
        )
        self._previous_presence = current_presence
        self._previous_was_present = present

        if self.state is FrameGateState.INSPECTION_LOCK:
            return self._during_lock(
                image_path,
                sequence_id,
                present,
                presence,
                motion,
                sharpness,
                captured_at,
                started,
            )
        if not present:
            released = self._release_held()
            self.state = FrameGateState.EMPTY
            self._stable_count = 0
            return self._outcome(
                FrameDisposition.EMPTY,
                image_path,
                sequence_id,
                discard_paths=released,
                presence_score=presence,
                sharpness_score=sharpness,
                reason="panel is absent from presence ROI",
                started=started,
            )

        stable = motion is not None and motion <= self.settings.motion_threshold
        sharp_enough = sharpness >= self.settings.sharpness_threshold
        if motion is None and sharp_enough:
            # The first present image has no predecessor, so it is only a
            # provisional candidate. It cannot be selected until the next
            # present image confirms low motion. This permits two 150-ms
            # captures across a 300-ms stationary dwell without treating the
            # first image itself as an inspected panel.
            self.state = FrameGateState.STABLE
            self._stable_count = 1
            self._held.append(_Candidate(image_path, sequence_id, sharpness))
            return self._outcome(
                FrameDisposition.STABLE,
                image_path,
                sequence_id,
                presence_score=presence,
                motion_score=motion,
                sharpness_score=sharpness,
                reason="first sharp frame held pending motion confirmation",
                started=started,
            )
        if not stable or not sharp_enough:
            released = self._release_held()
            self.state = FrameGateState.PRESENT_MOVING
            self._stable_count = 0
            return self._outcome(
                FrameDisposition.MOVING,
                image_path,
                sequence_id,
                discard_paths=released,
                presence_score=presence,
                motion_score=motion,
                sharpness_score=sharpness,
                reason="panel is moving or code area is not sharp enough",
                started=started,
            )

        self.state = FrameGateState.STABLE
        self._stable_count += 1
        candidate = _Candidate(image_path, sequence_id, sharpness)
        self._held.append(candidate)
        if self._stable_count < self.settings.stable_frames_required:
            return self._outcome(
                FrameDisposition.STABLE,
                image_path,
                sequence_id,
                presence_score=presence,
                motion_score=motion,
                sharpness_score=sharpness,
                reason="stable frame held while waiting for confirmation",
                started=started,
            )

        selected = max(self._held, key=lambda item: item.sharpness)
        discarded = tuple(item.path for item in self._held if item.path != selected.path)
        self._held.clear()
        self._stable_count = 0
        self.state = FrameGateState.INSPECTION_LOCK
        self._refractory_until = captured_at + self.settings.refractory_period_ms / 1000
        if self._last_selected_at is not None:
            self._selection_intervals.append(captured_at - self._last_selected_at)
        self._last_selected_at = captured_at
        self._selected_at = captured_at
        self._first_empty_at = None
        self._empty_count = 0
        self._empty_seen_in_lock = False
        return self._outcome(
            FrameDisposition.SELECTED,
            image_path,
            sequence_id,
            selected_path=selected.path,
            selected_sequence_id=selected.sequence_id,
            discard_paths=discarded,
            presence_score=presence,
            motion_score=motion,
            sharpness_score=selected.sharpness,
            reason="sharpest confirmed stable frame selected for inspection",
            started=started,
        )

    def _during_lock(
        self,
        path: Path,
        sequence_id: int,
        present: bool,
        presence: float,
        motion: float | None,
        sharpness: float,
        now: float,
        started: float,
    ) -> FrameGateOutcome:
        if not present:
            if self._first_empty_at is None:
                self._first_empty_at = now
            self._empty_count += 1
            if self._empty_count >= self.settings.empty_frames_to_rearm:
                self._empty_seen_in_lock = True
            if self._empty_seen_in_lock and now >= self._refractory_until:
                if self._selected_at is not None and self._first_empty_at is not None:
                    self._clear_times.append((self._first_empty_at - self._selected_at) * 1000)
                self.state = FrameGateState.EMPTY_REARMED
                outcome = self._outcome(
                    FrameDisposition.REARMED,
                    path,
                    sequence_id,
                    presence_score=presence,
                    motion_score=motion,
                    sharpness_score=sharpness,
                    reason="refractory elapsed and empty scene confirmed",
                    started=started,
                )
                self._reset_panel_tracking()
                return outcome
            return self._outcome(
                FrameDisposition.EMPTY,
                path,
                sequence_id,
                presence_score=presence,
                motion_score=motion,
                sharpness_score=sharpness,
                reason="empty frame observed while inspection lock is active",
                started=started,
            )
        if self._empty_seen_in_lock and now < self._refractory_until:
            return self._outcome(
                FrameDisposition.EARLY_PANEL,
                path,
                sequence_id,
                presence_score=presence,
                motion_score=motion,
                sharpness_score=sharpness,
                reason="new panel appeared before configured refractory period elapsed",
                started=started,
            )
        return self._outcome(
            FrameDisposition.SUPPRESSED,
            path,
            sequence_id,
            presence_score=presence,
            motion_score=motion,
            sharpness_score=sharpness,
            reason="frame suppressed during inspection lock",
            started=started,
        )

    def _outcome(
        self,
        disposition: FrameDisposition,
        path: Path,
        sequence_id: int,
        *,
        started: float,
        selected_path: Path | None = None,
        selected_sequence_id: int | None = None,
        discard_paths: tuple[Path, ...] = (),
        presence_score: float = 0.0,
        motion_score: float | None = None,
        sharpness_score: float = 0.0,
        reason: str = "",
    ) -> FrameGateOutcome:
        latency = (time.perf_counter() - started) * 1000
        self._latencies.append(latency)
        self._counts[disposition.value] = self._counts.get(disposition.value, 0) + 1
        return FrameGateOutcome(
            disposition=disposition,
            state=self.state,
            path=path,
            sequence_id=sequence_id,
            selected_path=selected_path,
            selected_sequence_id=selected_sequence_id,
            discard_paths=discard_paths,
            presence_score=presence_score,
            motion_score=motion_score,
            sharpness_score=sharpness_score,
            reason=reason,
            latency_ms=latency,
        )

    def _load_background(self) -> np.ndarray | None:
        if not self.background_path.exists():
            return None
        try:
            return np.load(self.background_path)
        except (OSError, ValueError):
            return None

    def _load_preview(self, path: Path) -> np.ndarray:
        import cv2  # type: ignore[import-not-found]

        # JPEG cameras can decode an eighth-resolution grayscale preview in the
        # codec itself. This avoids allocating a full FHD frame for gating; the
        # selected original is still read at full resolution by inference.
        read_mode = (
            cv2.IMREAD_REDUCED_GRAYSCALE_8
            if path.suffix.lower() in {".jpg", ".jpeg"}
            else cv2.IMREAD_GRAYSCALE
        )
        image = cv2.imread(str(path), read_mode)
        if image is None or image.size == 0:
            raise ValueError("OpenCV could not decode image")
        if image.shape[1] > self.settings.preview_max_width:
            scale = self.settings.preview_max_width / image.shape[1]
            image = cv2.resize(
                image,
                (self.settings.preview_max_width, max(1, round(image.shape[0] * scale))),
            )
        if self._background is not None and self._background.shape != image.shape:
            raise ValueError("image shape differs from calibrated background")
        return image

    @staticmethod
    def _difference(left: np.ndarray, right: np.ndarray) -> float:
        delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
        # A whole-frame mean can hide a moving FPCB when it occupies only part
        # of the configured ROI. The upper decile keeps local panel movement
        # visible while the operator-tunable threshold absorbs camera noise.
        flattened = delta.reshape(-1)
        upper_decile = np.partition(flattened, int(flattened.size * 0.90))[
            int(flattened.size * 0.90)
        ]
        return float(max(np.mean(delta), upper_decile) / 255.0)

    @staticmethod
    def _sharpness(image: np.ndarray) -> float:
        import cv2  # type: ignore[import-not-found]

        return float(cv2.Laplacian(image, cv2.CV_64F).var())

    @staticmethod
    def _crop(image: np.ndarray, roi: RelativeBox) -> np.ndarray:
        height, width = image.shape[:2]
        x1 = min(width - 1, max(0, round(roi.x * width)))
        y1 = min(height - 1, max(0, round(roi.y * height)))
        x2 = min(width, max(x1 + 1, round((roi.x + roi.width) * width)))
        y2 = min(height, max(y1 + 1, round((roi.y + roi.height) * height)))
        return image[y1:y2, x1:x2]

    def _release_held(self) -> tuple[Path, ...]:
        released = tuple(item.path for item in self._held)
        self._held.clear()
        return released

    def _reset_panel_tracking(self) -> None:
        self.state = FrameGateState.EMPTY
        self._previous_presence = None
        self._previous_was_present = False
        self._stable_count = 0
        self._held.clear()
        self._empty_count = 0
        self._empty_seen_in_lock = False
        self._selected_at = None
        self._first_empty_at = None

    def _observed_clear_ms(self) -> int:
        return round(max(self._clear_times, default=0.0))
