from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from gatekeeper.ingest.folder_watcher import RapidFolderWatcher
from gatekeeper.ingest.frame_gate import FrameDisposition, FrameGate, FrameGateSettings


def _write_image(path: Path, image: np.ndarray) -> Path:
    assert cv2.imwrite(str(path), image)
    return path


def _panel() -> np.ndarray:
    image = np.zeros((90, 160), dtype=np.uint8)
    cv2.rectangle(image, (30, 20), (130, 70), 180, -1)
    cv2.putText(image, "HJ04", (38, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    return image


def _gate(tmp_path: Path) -> FrameGate:
    settings = FrameGateSettings(
        refractory_period_ms=2000,
        stable_frames_required=2,
        empty_frames_to_rearm=2,
        presence_threshold=0.01,
        motion_threshold=0.001,
        sharpness_threshold=1.0,
    )
    gate = FrameGate(settings, tmp_path / "empty.npy")
    background = _write_image(tmp_path / "empty.png", np.zeros((90, 160), dtype=np.uint8))
    gate.request_background_capture()
    assert gate.evaluate(background, 1, now=0.0).disposition is FrameDisposition.CALIBRATED
    return gate


def test_frame_gate_selects_one_sharp_stable_frame_and_rearms(tmp_path: Path) -> None:
    gate = _gate(tmp_path)
    first = _write_image(tmp_path / "panel_1.png", _panel())
    second = _write_image(tmp_path / "panel_2.png", _panel())
    third = _write_image(tmp_path / "panel_3.png", _panel())
    assert gate.evaluate(first, 2, now=0.10).disposition is FrameDisposition.STABLE
    selected = gate.evaluate(second, 3, now=0.20)
    assert selected.disposition is FrameDisposition.SELECTED
    assert selected.selected_path == first

    empty = _write_image(tmp_path / "empty_2.png", np.zeros((90, 160), dtype=np.uint8))
    assert gate.evaluate(third, 4, now=0.30).disposition is FrameDisposition.SUPPRESSED
    assert gate.evaluate(empty, 5, now=0.40).disposition is FrameDisposition.EMPTY
    assert gate.evaluate(empty, 6, now=0.50).disposition is FrameDisposition.EMPTY
    early = gate.evaluate(first, 7, now=0.60)
    assert early.disposition is FrameDisposition.EARLY_PANEL
    assert gate.evaluate(empty, 8, now=2.40).disposition is FrameDisposition.REARMED


def test_frame_gate_requires_background_before_inspection(tmp_path: Path) -> None:
    gate = FrameGate(FrameGateSettings(), tmp_path / "missing.npy")
    path = _write_image(tmp_path / "panel.png", _panel())
    outcome = gate.evaluate(path, 1, now=0.0)
    assert outcome.disposition is FrameDisposition.UNCALIBRATED


def test_rapid_folder_watcher_dispatches_only_after_settle(tmp_path: Path) -> None:
    received: list[tuple[Path, int]] = []
    not_ready: list[Path] = []
    watcher = RapidFolderWatcher(
        tmp_path,
        lambda path, sequence: received.append((path, sequence)),
        on_not_ready=not_ready.append,
        settle_ms=30,
    )
    path = _write_image(tmp_path / "camera_001.png", _panel())
    assert watcher.scan_once() == 0
    time.sleep(0.04)
    assert watcher.scan_once() == 1
    assert received == [(path, 1)]
    assert not_ready == [path]
