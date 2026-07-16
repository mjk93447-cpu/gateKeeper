"""Generate and replay deterministic FHD camera sequences through Frame Gate.

Development verification only. Generated images are held in a temporary folder
and never enter the release bundle or a factory dataset.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from gatekeeper.ingest.frame_gate import FrameDisposition, FrameGate, FrameGateSettings, RelativeBox

WIDTH = 1920
HEIGHT = 1080


def _background(rng: np.random.Generator) -> np.ndarray:
    base = rng.normal(30, 4, (HEIGHT, WIDTH, 3)).clip(0, 255).astype(np.uint8)
    for x in range(0, WIDTH, 160):
        cv2.line(base, (x, 0), (x, HEIGHT), (38, 38, 38), 1)
    for y in range(0, HEIGHT, 135):
        cv2.line(base, (0, y), (WIDTH, y), (38, 38, 38), 1)
    return base


def _panel_frame(background: np.ndarray, code: str, offset: int, blur: bool) -> np.ndarray:
    frame = background.copy()
    left, top = 560 + offset, 330
    cv2.rectangle(frame, (left, top), (left + 800, top + 390), (65, 76, 88), -1)
    cv2.rectangle(frame, (left + 22, top + 28), (left + 778, top + 362), (90, 106, 120), 3)
    for x in range(left + 90, left + 760, 85):
        cv2.line(frame, (x, top + 85), (x, top + 300), (135, 150, 162), 2)
    cv2.rectangle(frame, (left + 258, top + 145), (left + 548, top + 248), (210, 210, 196), -1)
    cv2.putText(
        frame,
        code,
        (left + 285, top + 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (15, 15, 15),
        5,
        cv2.LINE_AA,
    )
    if blur:
        frame = cv2.GaussianBlur(frame, (21, 21), 6)
    return frame


def _run(interval: float, frames: int, dwell: float) -> dict[str, object]:
    rng = np.random.default_rng(20260716)
    counts: dict[str, int] = {}
    selected_sessions: list[dict[str, str]] = []
    frame_metadata: dict[Path, dict[str, str]] = {}
    max_latency = 0.0
    with tempfile.TemporaryDirectory(prefix="gatekeeper-fhd-") as directory:
        root = Path(directory)
        settings = FrameGateSettings(
            refractory_period_ms=2000,
            stable_frames_required=2,
            empty_frames_to_rearm=2,
            presence_threshold=0.018,
            motion_threshold=0.010,
            sharpness_threshold=12.0,
            preview_max_width=160,
            presence_roi=RelativeBox(0.25, 0.25, 0.5, 0.5),
            sharpness_roi=RelativeBox(0.25, 0.25, 0.5, 0.5),
        )
        gate = FrameGate(settings, root / "background.npy")
        empty = _background(rng)
        background_path = root / "background.jpg"
        cv2.imwrite(str(background_path), empty, [cv2.IMWRITE_JPEG_QUALITY, 92])
        gate.request_background_capture()
        gate.evaluate(background_path, 0, now=0.0)

        started = perf_counter()
        for sequence in range(1, frames + 1):
            now = sequence * interval
            cycle_time = now % 4.0
            panel_index = int(now // 4.0)
            present = cycle_time < dwell + 1e-9
            # Alternate stationary panels with deliberately blurred/translated
            # arrivals so the replay tests both accepted and rejected sessions.
            moving = present and panel_index % 2 == 1 and cycle_time < 0.12
            code = "HJ04" if panel_index % 2 == 0 else "HJ05"
            offset = round(500 * cycle_time) if moving else 0
            image = _panel_frame(empty, code, offset, moving) if present else empty
            path = root / f"camera_{sequence:04d}.jpg"
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            frame_metadata[path] = {
                "code": code,
                "session": "moving" if panel_index % 2 else "stationary",
            }
            outcome = gate.evaluate(path, sequence, now=now)
            counts[outcome.disposition.value] = counts.get(outcome.disposition.value, 0) + 1
            max_latency = max(max_latency, outcome.latency_ms)
            if outcome.disposition is FrameDisposition.SELECTED:
                assert outcome.selected_path is not None
                selected_sessions.append(
                    {
                        **frame_metadata[outcome.selected_path],
                        "confirmation_motion": f"{outcome.motion_score or 0.0:.5f}",
                    }
                )

        elapsed = perf_counter() - started
        metrics = gate.metrics()
    return {
        "interval_ms": interval * 1000,
        "frames": frames,
        "dwell_ms": dwell * 1000,
        "wall_seconds": round(elapsed, 3),
        "max_gate_latency_ms": round(max_latency, 3),
        "selected_panels": len(selected_sessions),
        "selected_sessions": selected_sessions,
        "events": counts,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--dwell-ms", type=int, default=300)
    args = parser.parse_args()
    result = _run(args.interval_ms / 1000, args.frames, args.dwell_ms / 1000)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
