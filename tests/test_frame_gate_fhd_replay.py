from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from gatekeeper.ingest.frame_gate import FrameDisposition, FrameGate, FrameGateSettings


def _empty_frame() -> np.ndarray:
    frame = np.full((1080, 1920), 26, dtype=np.uint8)
    cv2.rectangle(frame, (260, 290), (1660, 800), 32, 2)
    return frame


def _panel_frame(offset_x: int = 0) -> np.ndarray:
    frame = _empty_frame()
    x, y = 650 + offset_x, 410
    cv2.rectangle(frame, (x, y), (x + 600, y + 250), 184, -1)
    cv2.rectangle(frame, (x + 90, y + 72), (x + 510, y + 192), 222, -1)
    cv2.putText(
        frame, "HJ04", (x + 155, y + 165), cv2.FONT_HERSHEY_SIMPLEX, 3.1, 25, 8
    )
    return frame


def _write_jpeg(path: Path, frame: np.ndarray) -> Path:
    assert cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def test_fhd_150ms_camera_replay_selects_one_stationary_frame_per_panel(tmp_path: Path) -> None:
    """Replay 100 FHD camera frames: blank, moving, stable, lock, and re-arm."""

    gate = FrameGate(
        FrameGateSettings(
            expected_camera_interval_ms=150,
            refractory_period_ms=2000,
            stable_frames_required=2,
            empty_frames_to_rearm=2,
            presence_threshold=0.02,
            motion_threshold=0.01,
            sharpness_threshold=5.0,
        ),
        tmp_path / "empty_background.npy",
    )
    now = 0.0
    sequence = 0
    outcomes = []

    def submit(kind: str, frame: np.ndarray) -> None:
        nonlocal now, sequence
        sequence += 1
        path = _write_jpeg(tmp_path / f"{sequence:03d}_{kind}.jpg", frame)
        outcomes.append(gate.evaluate(path, sequence, now=now))
        now += 0.150

    gate.request_background_capture()
    submit("background", _empty_frame())
    for _ in range(4):
        submit("initial_empty", _empty_frame())

    for panel in range(5):
        # A changing offset deliberately makes the second candidate moving.
        submit(f"p{panel}_moving_a", _panel_frame(0))
        submit(f"p{panel}_moving_b", _panel_frame(55))
        submit(f"p{panel}_stable_a", _panel_frame(0))
        submit(f"p{panel}_stable_b", _panel_frame(0))
        # The return from a moving frame is itself motion. The next two
        # stationary frames therefore provide the two-frame confirmation.
        submit(f"p{panel}_stable_c", _panel_frame(0))
        # Fourteen empty frames cover the 2 s refractory time and re-arm gate.
        for empty_index in range(14):
            submit(f"p{panel}_empty_{empty_index:02d}", _empty_frame())

    assert sequence == 100
    selected = [item for item in outcomes if item.disposition is FrameDisposition.SELECTED]
    moving = [item for item in outcomes if item.disposition is FrameDisposition.MOVING]
    rearmed = [item for item in outcomes if item.disposition is FrameDisposition.REARMED]
    assert len(selected) == 5
    assert all(
        item.selected_path is not None and "stable" in item.selected_path.name
        for item in selected
    )
    assert len(moving) >= 5
    assert len(rearmed) == 5
    assert not any(
        item.disposition is FrameDisposition.SELECTED and "moving" in item.path.name
        for item in outcomes
    )
    assert float(gate.metrics()["p95_gate_latency_ms"]) < 20.0
