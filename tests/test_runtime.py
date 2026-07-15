from __future__ import annotations

import hashlib
import json

from gatekeeper.domain import DecisionEngine, DisplayState, InspectionInput
from gatekeeper.inference.pipeline import InspectionPipeline
from gatekeeper.inference.roi import RelativeRoi
from gatekeeper.inference.types import Detection, DetectionResult, OcrResult
from gatekeeper.infrastructure.alarm import AlarmMode, MemoryAlarmController
from gatekeeper.infrastructure.model_registry import ModelRegistry
from gatekeeper.infrastructure.plc import ProblemEvent, SignalStatus, SimulatedOutputPort
from gatekeeper.ingest.folder_watcher import ArchiveManager, FolderWatcher
from gatekeeper.runtime.paths import RuntimePaths
from gatekeeper.storage.sqlite_store import SQLiteEventStore


def test_alarm_modes_are_replaced_by_latest_result() -> None:
    alarm = MemoryAlarmController()
    alarm.problem()
    alarm.normal()
    assert alarm.mode is AlarmMode.SILENT
    assert alarm.history == [AlarmMode.PROBLEM, AlarmMode.SILENT]


def test_sqlite_event_store_round_trip(tmp_path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    store.append("decision", {"state": "NORMAL"}, sequence_id=4, panel_id="p4")
    event = store.recent(1)[0]
    assert event["event_type"] == "decision"
    assert event["sequence_id"] == 4
    assert event["payload"]["state"] == "NORMAL"


def test_folder_watcher_stability_and_duplicate_guard(tmp_path) -> None:
    seen: list[tuple[str, int]] = []
    watcher = FolderWatcher(
        tmp_path,
        lambda path, seq: seen.append((path.name, seq)),
        stable_checks=1,
    )
    image = tmp_path / "panel.png"
    image.write_bytes(b"image")
    assert watcher.scan_once() == 1
    assert watcher.scan_once() == 0
    assert seen == [("panel.png", 1)]

    duplicates: list[str] = []
    watcher.on_duplicate = lambda path: duplicates.append(path.name)
    duplicate = tmp_path / "panel-copy.png"
    duplicate.write_bytes(b"image")
    assert watcher.scan_once() == 0
    assert "panel.png" in duplicates and "panel-copy.png" in duplicates


def test_archive_manager_routes_result(tmp_path) -> None:
    source = tmp_path / "watch" / "panel.png"
    source.parent.mkdir()
    source.write_bytes(b"image")
    destination = ArchiveManager(tmp_path / "archive").move(source, "PROBLEM")
    assert destination == tmp_path / "archive" / "problem" / "panel.png"
    assert destination.is_file()


def test_four_character_codes_remain_safe() -> None:
    result = DecisionEngine().decide(
        InspectionInput(
            expected_code="HJ04",
            problem_codes=frozenset({"HJ05"}),
            detected=True,
            detector_confidence=0.99,
            ocr_text="HJ04",
            ocr_confidence=0.99,
        )
    )
    assert result.state.value == "NORMAL"


def test_plc_problem_request_is_idempotent() -> None:
    plc = SimulatedOutputPort()
    event = ProblemEvent(sequence_id=8, panel_id="panel-8", reason="HJ05")
    assert plc.request_problem(event).status is SignalStatus.ACCEPTED
    assert plc.request_problem(event).status is SignalStatus.DUPLICATE
    assert len(plc.reject_requests) == 1


def test_model_registry_verifies_hash(tmp_path) -> None:
    model = tmp_path / "detector.onnx"
    model.write_bytes(b"model")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "detector": {
                    "path": model.name,
                    "sha256": hashlib.sha256(b"model").hexdigest(),
                    "architecture": "YOLO26s",
                }
            }
        ),
        encoding="utf-8",
    )
    assert ModelRegistry(manifest).verify("detector").path == model


def test_runtime_paths_do_not_depend_on_current_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GATEKEEPER_HOME", str(tmp_path))
    paths = RuntimePaths.discover()
    assert paths.models == tmp_path / "models"
    assert paths.logs == tmp_path / "logs"
    assert paths.watch == tmp_path / "watch"


def test_runtime_paths_source_root_is_repository(monkeypatch) -> None:
    monkeypatch.delenv("GATEKEEPER_HOME", raising=False)
    assert RuntimePaths.discover().root.name == "gateKeeper"


class FakeDetector:
    def detect(self, image):
        return DetectionResult((Detection("code_roi", 0.99, (0, 0, 10, 10)),))


class FakeOcr:
    def recognize(self, roi):
        return OcrResult("HJ05", 0.99)


def test_pipeline_returns_problem_without_gpu(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "panel.png"
    image_path.write_bytes(b"placeholder")

    class FakeImage:
        size = 1
        shape = (10, 10, 3)

        def __getitem__(self, key):
            return self

    class FakeCv2:
        IMREAD_COLOR = 1

        @staticmethod
        def imread(path, mode):
            return FakeImage()

    monkeypatch.setitem(__import__("sys").modules, "cv2", FakeCv2)
    pipeline = InspectionPipeline(
        FakeDetector(), FakeOcr(), expected_code="HJ04", problem_codes=frozenset({"HJ05"})
    )
    result = pipeline.inspect(image_path, 1)
    assert result.state is DisplayState.PROBLEM


def test_relative_roi_is_applied_inside_detector_box() -> None:
    roi = RelativeRoi(x=0.25, y=0.25, width=0.5, height=0.5)
    assert roi.apply((100, 200, 300, 400)) == (150, 250, 250, 350)
