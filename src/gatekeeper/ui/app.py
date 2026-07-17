from __future__ import annotations

import json
import sys
import traceback
from collections import Counter
from dataclasses import replace
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gatekeeper.domain import (
    CodeRecipe,
    DecisionEngine,
    DisplayState,
    InspectionInput,
    InspectionResult,
    InspectionState,
    Thresholds,
)
from gatekeeper.inference.paddle_ocr import PaddleOcrRecognizer
from gatekeeper.inference.pipeline import InspectionPipeline
from gatekeeper.inference.roi import RelativeRoi
from gatekeeper.inference.yolo26_detector import Yolo26OnnxDetector
from gatekeeper.infrastructure.alarm import AlarmController, WindowsAlarmController
from gatekeeper.infrastructure.audit import JsonlDecisionSink
from gatekeeper.infrastructure.model_registry import ModelRegistry
from gatekeeper.infrastructure.plc import ProblemEvent, SimulatedOutputPort
from gatekeeper.ingest.folder_watcher import ArchiveManager, RapidFolderWatcher
from gatekeeper.ingest.frame_gate import (
    FrameDisposition,
    FrameGate,
    FrameGateOutcome,
    FrameGateSettings,
    RelativeBox,
)
from gatekeeper.runtime.paths import RuntimePaths
from gatekeeper.storage.sqlite_store import SQLiteEventStore
from gatekeeper.ui.labeling_view import LabelingView
from gatekeeper.ui.training_view import TrainingView

STATE_STYLE: dict[DisplayState, tuple[str, str]] = {
    DisplayState.NORMAL: ("OK", "#16a34a"),
    DisplayState.ABNORMAL: ("Abnormal", "#eab308"),
    DisplayState.PROBLEM: ("Error", "#dc2626"),
    DisplayState.SYSTEM_ERROR: ("System Error", "#7f1d1d"),
}


class ResultOverlay(QGroupBox):
    """Single non-modal result surface; each new result replaces the previous one."""

    def __init__(self) -> None:
        super().__init__("Latest result")
        self.message = QLabel("Waiting for image")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setMinimumHeight(140)
        self.message.setFont(QFont("Segoe UI", 42, QFont.Weight.Bold))
        self.details = QLabel("No inspection has been processed")
        self.details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.details.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.message)
        layout.addWidget(self.details)
        self.setMinimumWidth(420)

    def replace(self, result: InspectionResult) -> None:
        label, color = STATE_STYLE[result.state]
        self.hide()
        self.message.setText(label)
        self.message.setStyleSheet(
            f"background:{color}; color:white; border-radius:12px; padding:20px;"
        )
        self.details.setText(
            f"panel={result.panel_id}  seq={result.sequence_id}\n"
            f"expected={result.expected_code}  read={result.recognized_code or '-'} "
            f"corrected={result.corrected_code or '-'}\n"
            f"ROI={result.detector_confidence:.1%}  OCR={result.ocr_confidence:.1%}\n"
            f"crop={result.roi_box or '-'}\n"
            f"{result.reason}  ({result.latency_ms:.0f} ms)"
        )
        self.show()


class ResultPopup(QDialog):
    """Always-on-top, non-modal operator popup replaced by every newer result."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manufacturing Junction gateKeeper AI Vision result")
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumSize(620, 360)
        self.message = QLabel()
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setFont(QFont("Segoe UI", 54, QFont.Weight.Bold))
        self.details = QLabel()
        self.details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.details.setWordWrap(True)
        acknowledge = QPushButton("Acknowledge")
        acknowledge.clicked.connect(self.hide)
        layout = QVBoxLayout(self)
        layout.addWidget(self.message, 1)
        layout.addWidget(self.details)
        layout.addWidget(acknowledge)

    def replace(self, result: InspectionResult) -> None:
        label, color = STATE_STYLE[result.state]
        self.hide()
        self.message.setText(label)
        self.message.setStyleSheet(
            f"background:{color}; color:white; border-radius:14px; padding:24px;"
        )
        self.details.setText(
            f"Panel {result.panel_id} | Sequence {result.sequence_id}\n"
            f"Expected {result.expected_code} | Read {result.recognized_code or '-'} | "
            f"Corrected {result.corrected_code or '-'}\n"
            f"crop={result.roi_box or '-'}\n{result.reason}"
        )
        self.show()
        self.raise_()
        self.activateWindow()


class FolderController(QObject):
    result_ready = Signal(object)
    gate_event = Signal(object)

    def __init__(
        self,
        pipeline: InspectionPipeline,
        directory: Path,
        archive_directory: Path,
        *,
        settings: FrameGateSettings,
        background_path: Path,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.archive = ArchiveManager(archive_directory)
        self.settings = settings
        self.gate = FrameGate(settings, background_path)
        self.watcher = RapidFolderWatcher(
            directory,
            self._enqueue_frame,
            on_not_ready=self._frame_not_ready,
            poll_seconds=0.01,
            settle_ms=settings.file_settle_ms,
        )
        self._frames: Queue[tuple[Path, int] | None] = Queue(maxsize=settings.queue_depth)
        self._inference: Queue[tuple[Path, int] | None] = Queue(maxsize=1)
        self._stop = Event()
        self._gate_thread: Thread | None = None
        self._inference_thread: Thread | None = None
        self._suppressed_count = 0

    def start(self) -> None:
        self._stop.clear()
        self._gate_thread = Thread(
            target=self._gate_loop, daemon=True, name="gatekeeper-frame-gate"
        )
        self._inference_thread = Thread(
            target=self._inference_loop, daemon=True, name="gatekeeper-inference"
        )
        self._gate_thread.start()
        self._inference_thread.start()
        self.watcher.start()

    def stop(self) -> None:
        self.watcher.stop()
        self._stop.set()
        for queue in (self._frames, self._inference):
            try:
                queue.put_nowait(None)
            except Full:
                pass
        for thread in (self._gate_thread, self._inference_thread):
            if thread is not None:
                thread.join(timeout=3)
        self._gate_thread = None
        self._inference_thread = None

    def request_background_capture(self) -> None:
        self.gate.request_background_capture()

    def metrics(self) -> dict[str, object]:
        return {
            **self.gate.metrics(),
            "frame_queue_depth": self._frames.qsize(),
            "inference_queue_depth": self._inference.qsize(),
        }

    def _enqueue_frame(self, path: Path, sequence_id: int) -> None:
        self._emit_event("frame_received", {"path": str(path), "sequence_id": sequence_id})
        try:
            self._frames.put_nowait((path, sequence_id))
        except Full:
            try:
                dropped = self._frames.get_nowait()
            except Empty:
                dropped = None
            if dropped is not None:
                self._discard(dropped[0])
            self._emit_event(
                "frame_gate_overrun",
                {
                    "path": str(path),
                    "sequence_id": sequence_id,
                    "queue_depth": self.settings.queue_depth,
                },
            )
            try:
                self._frames.put_nowait((path, sequence_id))
            except Full:
                self._discard(path)

    def _frame_not_ready(self, path: Path) -> None:
        self._emit_event("frame_not_ready", {"path": str(path)})

    def _gate_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._frames.get(timeout=0.1)
            except Empty:
                continue
            if item is None:
                return
            path, sequence_id = item
            outcome = self.gate.evaluate(path, sequence_id, now=monotonic())
            self._handle_gate_outcome(outcome)

    def _handle_gate_outcome(self, outcome: FrameGateOutcome) -> None:
        payload = {
            "path": str(outcome.path),
            "sequence_id": outcome.sequence_id,
            "state": outcome.state.value,
            "presence_score": outcome.presence_score,
            "motion_score": outcome.motion_score,
            "sharpness_score": outcome.sharpness_score,
            "latency_ms": outcome.latency_ms,
            "reason": outcome.reason,
        }
        self._emit_event(outcome.disposition.value, payload)
        for discarded in outcome.discard_paths:
            self._discard(discarded)
        if outcome.disposition is FrameDisposition.SELECTED:
            assert outcome.selected_path is not None
            assert outcome.selected_sequence_id is not None
            try:
                self._inference.put_nowait((outcome.selected_path, outcome.selected_sequence_id))
            except Full:
                self._discard(outcome.selected_path)
                self._emit_system_error(
                    outcome.selected_sequence_id,
                    outcome.selected_path,
                    "inference_backpressure: selected panel could not be queued",
                )
            return
        if outcome.disposition is FrameDisposition.STABLE:
            return
        self._discard(outcome.path)

    def _inference_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._inference.get(timeout=0.1)
            except Empty:
                continue
            if item is None:
                return
            self._process_selected(*item)

    def _process_selected(self, path: Path, sequence_id: int) -> None:
        result = self.pipeline.inspect(path, sequence_id)
        try:
            archived = self.archive.move(path, result.state.value)
            result = replace(result, image_path=str(archived))
        except OSError as exc:
            result = replace(
                result,
                state=DisplayState.SYSTEM_ERROR,
                reason=f"archive failed: {exc}",
            )
        self.result_ready.emit(result)

    def _discard(self, path: Path) -> None:
        if not path.exists():
            return
        self._suppressed_count += 1
        try:
            if (
                self.settings.diagnostic_sample_every
                and self._suppressed_count % self.settings.diagnostic_sample_every == 0
            ):
                self.archive.move(path, "ignored")
            else:
                path.unlink()
        except OSError:
            pass

    def _emit_system_error(self, sequence_id: int, path: Path, reason: str) -> None:
        self.result_ready.emit(
            InspectionResult(
                panel_id=path.stem,
                sequence_id=sequence_id,
                state=DisplayState.SYSTEM_ERROR,
                expected_code=self.pipeline.expected_code,
                recognized_code=None,
                detector_confidence=0.0,
                ocr_confidence=0.0,
                image_path=str(path),
                model_version=self.pipeline.model_version,
                latency_ms=0.0,
                reason=reason,
            )
        )

    def _emit_event(self, event_type: str, payload: dict[str, object]) -> None:
        self.gate_event.emit({"event_type": event_type, "payload": payload, **self.metrics()})


class CodeRecipeDialog(QDialog):
    """Editor for the persisted normal/problem code registry."""

    def __init__(self, recipe: CodeRecipe, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage code recipe")
        self.setMinimumWidth(560)
        self._active = recipe.active_normal_code
        self.normal = QListWidget()
        self.problem = QListWidget()
        self.normal.addItems(recipe.normal_codes)
        self.problem.addItems(recipe.problem_codes)
        self.active = QLineEdit(self._active)
        self.active.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Normal codes"))
        layout.addWidget(self.normal)
        layout.addLayout(self._buttons(self.normal, "normal"))
        layout.addWidget(QLabel("Problem codes"))
        layout.addWidget(self.problem)
        layout.addLayout(self._buttons(self.problem, "problem"))
        form = QFormLayout()
        form.addRow("Active normal code", self.active)
        layout.addLayout(form)
        note = QLabel(
            "Codes must be four uppercase English letters or digits. "
            "Normal and problem codes cannot overlap after O/0 correction."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _buttons(self, target: QListWidget, kind: str) -> QHBoxLayout:
        layout = QHBoxLayout()
        add = QPushButton("Add")
        edit = QPushButton("Edit")
        delete = QPushButton("Delete")
        active = QPushButton("Set active")
        add.clicked.connect(lambda: self._add(target, kind))
        edit.clicked.connect(lambda: self._edit(target, kind))
        delete.clicked.connect(lambda: self._delete(target, kind))
        active.clicked.connect(lambda: self._set_active(target, kind))
        layout.addWidget(add)
        layout.addWidget(edit)
        layout.addWidget(delete)
        if kind == "normal":
            layout.addWidget(active)
        return layout

    @staticmethod
    def _prompt(parent: QWidget, title: str, value: str = "") -> str | None:
        text, accepted = QInputDialog.getText(parent, title, "Code", text=value)
        return text if accepted else None

    def _add(self, target: QListWidget, kind: str) -> None:
        value = self._prompt(self, f"Add {kind} code")
        if value is not None:
            target.addItem(value)

    def _edit(self, target: QListWidget, kind: str) -> None:
        item = target.currentItem()
        if item is None:
            return
        value = self._prompt(self, f"Edit {kind} code", item.text())
        if value is not None:
            if item.text() == self._active:
                self._active = value
                self.active.setText(value)
            item.setText(value)

    def _delete(self, target: QListWidget, kind: str) -> None:
        row = target.currentRow()
        if row < 0:
            return
        item = target.item(row)
        if kind == "normal" and item.text() == self._active:
            self._active = ""
            self.active.setText("")
        target.takeItem(row)

    def _set_active(self, target: QListWidget, kind: str) -> None:
        item = target.currentItem()
        if item is not None and kind == "normal":
            self._active = item.text()
            self.active.setText(item.text())

    def recipe(self) -> CodeRecipe:
        return CodeRecipe(
            normal_codes=tuple(
                self.normal.item(index).text() for index in range(self.normal.count())
            ),
            problem_codes=tuple(
                self.problem.item(index).text() for index in range(self.problem.count())
            ),
            active_normal_code=self._active,
        )

    def accept(self) -> None:
        try:
            self.recipe()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid code recipe", str(exc))
            return
        super().accept()


class FrameGateSettingsDialog(QDialog):
    """Line-specific fast-frame and refractory settings; values remain operator configurable."""

    def __init__(self, settings: FrameGateSettings, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Frame Gate Settings")
        self.setMinimumWidth(560)
        form = QFormLayout(self)
        self.camera_interval = self._integer(settings.expected_camera_interval_ms, 50, 1000)
        self.refractory = self._integer(settings.refractory_period_ms, 0, 60000)
        self.stable = self._integer(settings.stable_frames_required, 1, 10)
        self.empty = self._integer(settings.empty_frames_to_rearm, 1, 20)
        self.settle = self._integer(settings.file_settle_ms, 1, 1000)
        self.queue = self._integer(settings.queue_depth, 1, 100)
        self.presence = self._decimal(settings.presence_threshold, 0.0, 1.0, 3)
        self.motion = self._decimal(settings.motion_threshold, 0.0, 1.0, 3)
        self.sharpness = self._decimal(settings.sharpness_threshold, 0.0, 100000.0, 1)
        self.diagnostic = self._integer(settings.diagnostic_sample_every, 0, 100000)
        self.presence_roi = self._roi(settings.presence_roi)
        self.sharpness_roi = self._roi(settings.sharpness_roi)
        form.addRow("Expected camera interval (ms)", self.camera_interval)
        form.addRow("Refractory period (ms)", self.refractory)
        form.addRow("Stable frames required", self.stable)
        form.addRow("Empty frames to re-arm", self.empty)
        form.addRow("Final-file settle time (ms)", self.settle)
        form.addRow("Latest-wins frame queue depth", self.queue)
        form.addRow("Presence threshold", self.presence)
        form.addRow("Motion threshold", self.motion)
        form.addRow("Sharpness threshold", self.sharpness)
        form.addRow("Keep one ignored frame every N (0 = delete)", self.diagnostic)
        form.addRow("Presence ROI (x, y, width, height)", self.presence_roi)
        form.addRow("Sharpness ROI (x, y, width, height)", self.sharpness_roi)
        note = QLabel(
            "These values are fully manual. The dashboard warns when observed timing suggests "
            "a duplicate or missed-panel risk, but does not block saving. New values apply when "
            "hot-folder monitoring is restarted."
        )
        note.setWordWrap(True)
        form.addRow(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    @staticmethod
    def _integer(value: int, minimum: int, maximum: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    @staticmethod
    def _decimal(value: float, minimum: float, maximum: float, decimals: int) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(0.001 if decimals >= 3 else 1.0)
        widget.setValue(value)
        return widget

    def _roi(self, value: RelativeBox) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        fields = []
        for item in (value.x, value.y, value.width, value.height):
            field = self._decimal(item, 0.0, 1.0, 3)
            fields.append(field)
            layout.addWidget(field)
        row.roi_fields = fields  # type: ignore[attr-defined]
        return row

    @staticmethod
    def _read_roi(widget: QWidget) -> RelativeBox:
        values = widget.roi_fields  # type: ignore[attr-defined]
        return RelativeBox(*(field.value() for field in values))

    def settings(self) -> FrameGateSettings:
        return FrameGateSettings(
            expected_camera_interval_ms=self.camera_interval.value(),
            refractory_period_ms=self.refractory.value(),
            stable_frames_required=self.stable.value(),
            empty_frames_to_rearm=self.empty.value(),
            file_settle_ms=self.settle.value(),
            queue_depth=self.queue.value(),
            presence_threshold=self.presence.value(),
            motion_threshold=self.motion.value(),
            sharpness_threshold=self.sharpness.value(),
            diagnostic_sample_every=self.diagnostic.value(),
            presence_roi=self._read_roi(self.presence_roi),
            sharpness_roi=self._read_roi(self.sharpness_roi),
        )

    def accept(self) -> None:
        try:
            self.settings()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Frame Gate settings", str(exc))
            return
        super().accept()


class MainWindow(QMainWindow):
    folder_pipeline_ready = Signal(object)
    folder_start_failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Manufacturing Junction gateKeeper AI Vision - AAM FPCB Inspection")
        self.resize(1600, 900)
        self.setMinimumSize(1050, 700)
        self.counts: Counter[DisplayState] = Counter()
        self.last_sequence = -1
        self.last_result: InspectionResult | None = None
        self.paths = RuntimePaths.discover()
        self.watch_directory = self._load_watch_directory()
        self.recipe = CodeRecipe.load(self.paths.code_recipe)
        self.thresholds, self.candidate_confidence = self._load_thresholds()
        self.frame_gate_settings = self._load_frame_gate_settings()
        self.output = SimulatedOutputPort()
        self.alarm: AlarmController = WindowsAlarmController()
        self.events = SQLiteEventStore(self.paths.logs / "gatekeeper.sqlite3")
        self.audit = JsonlDecisionSink(self.paths.logs / "inspection.jsonl")
        self.controller: FolderController | None = None
        self._folder_start_pending = False
        self.folder_pipeline_ready.connect(self._activate_folder_pipeline)
        self.folder_start_failed.connect(self._folder_start_error)

        root = QWidget()
        root.setMinimumWidth(1050)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        self.popup = ResultPopup(self)
        page = QVBoxLayout(root)
        header = QHBoxLayout()
        title = QLabel("Manufacturing Junction gateKeeper AI Vision | AAM FPCB Code Inspection")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        header.addWidget(title)
        header.addStretch()
        self.mode = QLabel("SIMULATION / CPU")
        self.mode.setStyleSheet("color:#2563eb; font-weight:700;")
        header.addWidget(self.mode)
        legal = QPushButton("Legal notices")
        legal.clicked.connect(self._show_legal_notices)
        header.addWidget(legal)
        page.addLayout(header)

        body = QSplitter(Qt.Orientation.Horizontal)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        controls_scroll.setMinimumWidth(540)
        controls_scroll.setWidget(self._build_controls())
        body.addWidget(controls_scroll)
        result_panel = QWidget()
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        result_panel.setLayout(right)
        self.overlay = ResultOverlay()
        right.addWidget(self.overlay)
        self.detector_bar = QProgressBar()
        self.detector_bar.setFormat("ROI confidence %p%")
        self.ocr_bar = QProgressBar()
        self.ocr_bar.setFormat("OCR confidence %p%")
        right.addWidget(self.detector_bar)
        right.addWidget(self.ocr_bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        right.addWidget(self.log, 1)
        body.addWidget(result_panel)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 2)
        body.setSizes([560, 960])
        page.addWidget(body, 1)
        page.addWidget(LabelingView(self.paths.root, self.paths.code_recipe))
        page.addWidget(
            TrainingView(self.paths.root, self.paths.code_recipe, self.paths.training_runner)
        )
        self.statusBar().showMessage("Ready - waiting for an image")
        self.setStyleSheet(
            "QMainWindow{background:#f5f7fb;} QGroupBox{font-weight:600; background:white; "
            "border:1px solid #dbe2ea; border-radius:8px; margin-top:10px; padding-top:8px;} "
            "QGroupBox::title{subcontrol-origin:margin; left:10px; padding:0 4px;} "
            "QPushButton{padding:9px 14px;} QScrollArea{border:none; background:transparent;}"
        )

    def _build_controls(self) -> QGroupBox:
        box = QGroupBox("Inspection controls")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.expected = QComboBox()
        self.problems = QLineEdit()
        self.problems.setReadOnly(True)
        self._refresh_recipe_controls()
        self.expected.currentTextChanged.connect(self._set_active_normal_code)
        self.ocr_text = QLineEdit("HJ04")
        self.watch_path = QLineEdit(str(self.watch_directory))
        self.watch_path.setReadOnly(True)
        self.watch_path.setMinimumWidth(180)
        self.candidate_conf = self._confidence_box(self.candidate_confidence)
        self.det_conf = self._confidence_box(self.thresholds.detector_confidence)
        self.ocr_conf = self._confidence_box(0.98)
        self.roi_x = self._confidence_box(0.0)
        self.roi_y = self._confidence_box(0.0)
        self.roi_w = self._confidence_box(1.0)
        self.roi_h = self._confidence_box(1.0)
        form.addRow("Expected code", self.expected)
        form.addRow("Problem codes", self.problems)
        watch_row = QHBoxLayout()
        watch_row.addWidget(self.watch_path, 1)
        browse_watch = QPushButton("Choose folder")
        browse_watch.setMinimumWidth(120)
        browse_watch.clicked.connect(self._choose_watch_directory)
        watch_row.addWidget(browse_watch)
        form.addRow("Camera hot-folder", watch_row)
        manage_codes = QPushButton("Manage code recipe")
        manage_codes.clicked.connect(self._manage_codes)
        form.addRow(manage_codes)
        form.addRow("Simulation OCR", self.ocr_text)
        form.addRow("Detection candidate confidence", self.candidate_conf)
        form.addRow("Final ROI confidence", self.det_conf)
        form.addRow("OCR confidence", self.ocr_conf)
        form.addRow("OCR ROI relative X", self.roi_x)
        form.addRow("OCR ROI relative Y", self.roi_y)
        form.addRow("OCR ROI relative width", self.roi_w)
        form.addRow("OCR ROI relative height", self.roi_h)
        run = QPushButton("Process simulation image")
        run.clicked.connect(self._simulate)
        form.addRow(run)
        start = QPushButton("Start hot-folder")
        start.clicked.connect(self._start_folder)
        form.addRow(start)
        stop = QPushButton("Stop hot-folder")
        stop.clicked.connect(self._stop_folder)
        form.addRow(stop)
        frame_gate = QPushButton("Frame Gate Settings")
        frame_gate.clicked.connect(self._manage_frame_gate)
        form.addRow(frame_gate)
        capture_background = QPushButton("Capture Empty Background")
        capture_background.clicked.connect(self._capture_empty_background)
        form.addRow(capture_background)
        self.frame_gate_status = QLabel("Frame Gate: not running")
        self.frame_gate_status.setWordWrap(True)
        form.addRow("Frame Gate", self.frame_gate_status)
        self.frame_gate_warning = QLabel("")
        self.frame_gate_warning.setWordWrap(True)
        self.frame_gate_warning.setStyleSheet("color:#9a6700; font-weight:600;")
        form.addRow("Timing warning", self.frame_gate_warning)
        mute = QPushButton("Mute speaker")
        mute.clicked.connect(self.alarm.mute)
        form.addRow(mute)
        self.operator = QLineEdit()
        self.feedback_reason = QLineEdit()
        form.addRow("Operator ID", self.operator)
        form.addRow("Feedback reason", self.feedback_reason)
        feedback_row = QHBoxLayout()
        for label, value in (
            ("Approve OK", "NORMAL"),
            ("Mark abnormal", "ABNORMAL"),
            ("Mark problem", "PROBLEM"),
            ("Add to retraining", "RETRAIN"),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, choice=value: self._feedback(choice))
            feedback_row.addWidget(button)
        form.addRow(feedback_row)
        self.count_label = QLabel("OK 0 | Abnormal 0 | Error 0")
        form.addRow("Counters", self.count_label)
        hint = QLabel(
            "A new result automatically replaces the previous popup.\n"
            "Uncertain or system-error results are never treated as OK."
        )
        hint.setWordWrap(True)
        form.addRow(hint)
        return box

    @staticmethod
    def _confidence_box(value: float) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(0.0, 1.0)
        widget.setDecimals(2)
        widget.setSingleStep(0.01)
        widget.setValue(value)
        return widget

    def _simulate(self) -> None:
        try:
            decision = DecisionEngine().decide(
                InspectionInput(
                    expected_code=self.expected.currentText(),
                    problem_codes=frozenset(self.recipe.problem_codes),
                    detected=self.det_conf.value() > 0,
                    detector_confidence=self.det_conf.value(),
                    ocr_text=self.ocr_text.text(),
                    ocr_confidence=self.ocr_conf.value(),
                    thresholds=Thresholds(
                        detector_confidence=self.det_conf.value(),
                        normal_ocr_confidence=self.thresholds.normal_ocr_confidence,
                        problem_ocr_confidence=self.thresholds.problem_ocr_confidence,
                    ),
                )
            )
            self._present(
                InspectionResult(
                    panel_id=decision.panel_id,
                    sequence_id=self.last_sequence + 1,
                    state=DisplayState(decision.state.value),
                    expected_code=decision.expected_code,
                    recognized_code=decision.recognized_code,
                    detector_confidence=decision.detector_confidence,
                    ocr_confidence=decision.ocr_confidence,
                    image_path="simulation",
                    model_version="simulation",
                    latency_ms=0.0,
                    reason=decision.reason,
                    corrected_code=decision.corrected_code,
                )
            )
        except (OSError, ValueError) as exc:
            self._present_error(str(exc))

    def _start_folder(self) -> None:
        if self.controller is not None or self._folder_start_pending:
            self.statusBar().showMessage("Hot-folder monitoring is already running")
            return
        self._folder_start_pending = True
        self.mode.setText("STARTING / CPU / HOT-FOLDER")
        self.statusBar().showMessage("Loading CPU detector and OCR model in the background")
        Thread(
            target=self._build_folder_controller, daemon=True, name="gatekeeper-model-load"
        ).start()

    def _build_folder_controller(self) -> None:
        stage = "verifying detector model"
        try:
            registry = ModelRegistry(self.paths.models / "manifest.json")
            detector_artifact = registry.verify("detector")
            model_path = detector_artifact.path
            stage = "creating YOLO detector"
            detector = Yolo26OnnxDetector(
                model_path, confidence=self.candidate_conf.value()
            )
            stage = "creating local PaddleOCR predictor"
            ocr = PaddleOcrRecognizer(self.paths.models / "ocr")
            stage = "creating inspection pipeline"
            pipeline = InspectionPipeline(
                detector,
                ocr,
                expected_code=self.expected.currentText(),
                problem_codes=frozenset(self.recipe.problem_codes),
                model_version=model_path.name,
                thresholds=Thresholds(
                    detector_confidence=self.det_conf.value(),
                    normal_ocr_confidence=self.thresholds.normal_ocr_confidence,
                    problem_ocr_confidence=self.thresholds.problem_ocr_confidence,
                ),
                ocr_relative_roi=RelativeRoi(
                    x=self.roi_x.value(),
                    y=self.roi_y.value(),
                    width=self.roi_w.value(),
                    height=self.roi_h.value(),
                ),
            )
            # The expensive ML objects are safe to create off the UI thread.
            # FolderController is a QObject, so create it only after returning
            # to the Qt UI thread to preserve Qt thread affinity.
            self.folder_pipeline_ready.emit(pipeline)
        except Exception as exc:
            # The UI stays concise, while the local support log preserves the
            # complete failure chain needed for an offline field diagnosis.
            details = (
                f"[{stage}] {type(exc).__name__}: {exc}\n"
                f"args={exc.args!r}\n"
                f"cause={exc.__cause__!r}\n"
                f"context={exc.__context__!r}\n"
                f"{traceback.format_exc()}\n"
            )
            self.paths.logs.mkdir(parents=True, exist_ok=True)
            with (self.paths.logs / "startup-errors.log").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(details)
            self.folder_start_failed.emit(f"{stage}: {exc}")

    def _activate_folder_pipeline(self, pipeline: object) -> None:
        if not isinstance(pipeline, InspectionPipeline):
            self._folder_start_error("invalid hot-folder inspection pipeline")
            return
        if not self._folder_start_pending:
            return
        try:
            self.controller = FolderController(
                pipeline,
                self.watch_directory,
                self.paths.archive,
                settings=self.frame_gate_settings,
                background_path=self.paths.frame_gate_background,
            )
        except Exception as exc:
            self._folder_start_error(str(exc))
            return
        self._folder_start_pending = False
        self.controller.result_ready.connect(self._present)
        self.controller.gate_event.connect(self._handle_gate_event)
        self.controller.start()
        self.mode.setText("LIVE / CPU / HOT-FOLDER / FRAME GATE")
        if self.controller.gate.calibrated:
            self.statusBar().showMessage(
                f"Hot-folder is running: {self.watch_directory} (Frame Gate calibrated)"
            )
        else:
            self.statusBar().showMessage(
                "Hot-folder is running: clear the view and click Capture Empty Background"
            )

    def _folder_start_error(self, reason: str) -> None:
        self._folder_start_pending = False
        self.mode.setText("SIMULATION / CPU")
        self._present_error(reason)

    def _stop_folder(self) -> None:
        if self._folder_start_pending:
            self._folder_start_pending = False
            self.mode.setText("SIMULATION / CPU")
            self.statusBar().showMessage("Hot-folder startup cancelled")
            return
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
        self.mode.setText("SIMULATION / CPU")
        self.frame_gate_status.setText("Frame Gate: not running")
        self.statusBar().showMessage("Hot-folder stopped")

    def _present_error(self, reason: str) -> None:
        result = InspectionResult(
            panel_id="system",
            sequence_id=self.last_sequence + 1,
            state=DisplayState.SYSTEM_ERROR,
            expected_code=self.expected.currentText(),
            recognized_code=None,
            detector_confidence=0.0,
            ocr_confidence=0.0,
            image_path="",
            model_version="unknown",
            latency_ms=0.0,
            reason=reason,
        )
        self._present(result)

    def _present(self, result: InspectionResult) -> None:
        if result.sequence_id <= self.last_sequence:
            return
        self.last_sequence = result.sequence_id
        self.last_result = result
        self.overlay.replace(result)
        self.popup.replace(result)
        self.detector_bar.setValue(round(result.detector_confidence * 100))
        self.ocr_bar.setValue(round(result.ocr_confidence * 100))
        self.counts[result.state] += 1
        self.count_label.setText(
            f"OK {self.counts[DisplayState.NORMAL]} | "
            f"Abnormal {self.counts[DisplayState.ABNORMAL]} | "
            f"Error {self.counts[DisplayState.PROBLEM]}"
        )
        if result.state is DisplayState.NORMAL:
            self.alarm.normal()
            self.output.apply(InspectionState.NORMAL)
        elif result.state is DisplayState.ABNORMAL:
            self.alarm.abnormal()
            self.output.apply(InspectionState.ABNORMAL)
        elif result.state is DisplayState.PROBLEM:
            self.alarm.problem()
            self.output.apply(InspectionState.PROBLEM)
            self.output.request_problem(
                ProblemEvent(
                    sequence_id=result.sequence_id,
                    panel_id=result.panel_id,
                    reason=result.reason,
                )
            )
        else:
            self.alarm.system_error()
        payload = {
            "state": result.state.value,
            "panel_id": result.panel_id,
            "sequence_id": result.sequence_id,
            "expected_code": result.expected_code,
            "recognized_code": result.recognized_code,
            "corrected_code": result.corrected_code,
            "detector_confidence": result.detector_confidence,
            "ocr_confidence": result.ocr_confidence,
            "image_path": result.image_path,
            "model_version": result.model_version,
            "latency_ms": result.latency_ms,
            "reason": result.reason,
            "roi_box": result.roi_box,
        }
        self.events.append(
            "decision" if result.state is not DisplayState.SYSTEM_ERROR else "system_error",
            payload,
            sequence_id=result.sequence_id,
            panel_id=result.panel_id,
        )
        self.audit.append_event(
            "decision" if result.state is not DisplayState.SYSTEM_ERROR else "system_error",
            payload,
        )
        self.log.appendPlainText(
            f"[{result.created_at.astimezone():%H:%M:%S}] {result.state.value} "
            f"seq={result.sequence_id} read={result.recognized_code or '-'} {result.reason}"
        )
        self.statusBar().showMessage(f"Latest result: {STATE_STYLE[result.state][0]}")

    def _refresh_recipe_controls(self) -> None:
        selected = self.recipe.active_normal_code
        if hasattr(self, "expected"):
            selected = self.expected.currentText() or selected
            self.expected.clear()
            self.expected.addItems(self.recipe.normal_codes)
            selected = (
                selected if selected in self.recipe.normal_codes else self.recipe.active_normal_code
            )
            self.expected.setCurrentText(selected)
        if hasattr(self, "problems"):
            self.problems.setText(", ".join(self.recipe.problem_codes) or "None")

    def _load_thresholds(self) -> tuple[Thresholds, float]:
        try:
            payload = json.loads(self.paths.config.read_text(encoding="utf-8"))
            values = payload.get("thresholds", {})
            thresholds = Thresholds(
                detector_confidence=float(values.get("detector_confidence", 0.70)),
                normal_ocr_confidence=float(values.get("normal_ocr_confidence", 0.90)),
                problem_ocr_confidence=float(values.get("problem_ocr_confidence", 0.90)),
            )
            candidate = float(
                values.get("detection_candidate_confidence", thresholds.detector_confidence)
            )
            if not 0.0 <= candidate <= 1.0:
                raise ValueError("detection_candidate_confidence must be between 0 and 1")
            return thresholds, candidate
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid runtime threshold configuration: {exc}") from exc

    def _load_watch_directory(self) -> Path:
        try:
            payload = json.loads(self.paths.config.read_text(encoding="utf-8"))
            configured = str(payload.get("watch", {}).get("input_dir", "watch"))
            candidate = Path(configured).expanduser()
            if not candidate.is_absolute():
                candidate = self.paths.root / candidate
            return candidate.resolve()
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid hot-folder configuration: {exc}") from exc

    def _save_watch_directory(self) -> None:
        payload = json.loads(self.paths.config.read_text(encoding="utf-8"))
        watch = payload.setdefault("watch", {})
        try:
            configured = self.watch_directory.relative_to(self.paths.root).as_posix()
        except ValueError:
            configured = str(self.watch_directory)
        watch["input_dir"] = configured
        self.paths.config.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _choose_watch_directory(self) -> None:
        if self.controller is not None:
            QMessageBox.information(
                self,
                "Hot-folder is active",
                "Stop hot-folder monitoring before changing the camera input folder.",
            )
            return
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose camera hot-folder",
            str(self.watch_directory),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected:
            return
        try:
            self.watch_directory = Path(selected).resolve()
            self._save_watch_directory()
            self.watch_path.setText(str(self.watch_directory))
            self.events.append("hot_folder_updated", {"path": str(self.watch_directory)})
            self.audit.append_event("hot_folder_updated", {"path": str(self.watch_directory)})
            self.statusBar().showMessage(f"Camera hot-folder saved: {self.watch_directory}")
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Hot-folder save failed", str(exc))

    def _load_frame_gate_settings(self) -> FrameGateSettings:
        try:
            payload = json.loads(self.paths.config.read_text(encoding="utf-8"))
            return FrameGateSettings.from_mapping(payload.get("frame_gate", {}))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid Frame Gate configuration: {exc}") from exc

    def _save_frame_gate_settings(self) -> None:
        payload = json.loads(self.paths.config.read_text(encoding="utf-8"))
        payload["frame_gate"] = self.frame_gate_settings.to_mapping()
        self.paths.config.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _manage_frame_gate(self) -> None:
        dialog = FrameGateSettingsDialog(self.frame_gate_settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.frame_gate_settings = dialog.settings()
            self._save_frame_gate_settings()
            self.frame_gate_warning.setText(
                "Settings saved. Restart hot-folder monitoring to apply them."
            )
            self.statusBar().showMessage("Frame Gate settings saved")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Frame Gate save failed", str(exc))

    def _capture_empty_background(self) -> None:
        if self.controller is None:
            self.statusBar().showMessage("Start hot-folder monitoring before capturing background")
            return
        self.controller.request_background_capture()
        self.frame_gate_warning.setText(
            "Capture armed: keep the camera view empty for the next received frame."
        )
        self.statusBar().showMessage("Waiting for one empty frame to calibrate Frame Gate")

    def _handle_gate_event(self, event: object) -> None:
        if not isinstance(event, dict):
            return
        event_type = str(event.get("event_type", "frame_gate_event"))
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        sequence_id = payload.get("sequence_id")
        sequence = int(sequence_id) if isinstance(sequence_id, int) else None
        self.events.append(event_type, payload, sequence_id=sequence)
        self.audit.append_event(event_type, payload)
        state = str(event.get("state", "unknown"))
        queue_depth = event.get("frame_queue_depth", 0)
        frame_rate = event.get("live_frame_rate_fps", 0.0)
        interval = event.get("median_frame_interval_ms", 0.0)
        p50 = event.get("p50_gate_latency_ms", 0.0)
        latency = event.get("p95_gate_latency_ms", 0.0)
        selected = dict(event.get("counts", {})).get("frame_selected", 0)
        suppressed = sum(
            value
            for key, value in dict(event.get("counts", {})).items()
            if key in {"frame_suppressed", "frame_empty", "frame_moving", "unexpected_early_panel"}
        )
        self.frame_gate_status.setText(
            f"Frame Gate: {state} | {frame_rate} FPS / median {interval} ms | "
            f"queue {queue_depth} | selected {selected} | suppressed {suppressed} | "
            f"p50/p95 {p50}/{latency} ms"
        )
        warning = str(event.get("warning", ""))
        if event_type == FrameDisposition.EARLY_PANEL.value:
            warning = "Timing Warning: panel appeared before refractory period elapsed."
        if event_type == "frame_gate_overrun":
            warning = "Timing Warning: latest-wins Frame Gate queue overrun."
        self.frame_gate_warning.setText(warning)
        if event_type in {
            FrameDisposition.SELECTED.value,
            FrameDisposition.EARLY_PANEL.value,
            "frame_gate_overrun",
            FrameDisposition.CALIBRATED.value,
        }:
            self.log.appendPlainText(f"[Frame Gate] {event_type}: {payload.get('reason', '')}")

    def _manage_codes(self) -> None:
        dialog = CodeRecipeDialog(self.recipe, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.recipe = dialog.recipe()
            self.recipe.save(self.paths.code_recipe)
            self._refresh_recipe_controls()
            self.events.append(
                "code_recipe_updated",
                {
                    "normal_codes": list(self.recipe.normal_codes),
                    "problem_codes": list(self.recipe.problem_codes),
                    "active_normal_code": self.recipe.active_normal_code,
                },
            )
            self.statusBar().showMessage("Code recipe saved and will be used by new inspections")
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Code recipe save failed", str(exc))

    def _set_active_normal_code(self, value: str) -> None:
        if not value or value == self.recipe.active_normal_code:
            return
        try:
            self.recipe = CodeRecipe(
                normal_codes=self.recipe.normal_codes,
                problem_codes=self.recipe.problem_codes,
                active_normal_code=value,
            )
            self.recipe.save(self.paths.code_recipe)
        except (OSError, ValueError) as exc:
            self._present_error(f"active code update failed: {exc}")

    def _show_legal_notices(self) -> None:
        source = self.paths.root / "source"
        message = (
            "This application is distributed under GNU AGPL-3.0-or-later.\n\n"
            "The installed LICENSE and NOTICE files contain the terms. "
            "The release-specific corresponding source archive is in: "
            f"{source}\n\n"
            "Third-party model and dependency notices are in docs/THIRD_PARTY.md."
        )
        QMessageBox.information(self, "Legal notices", message)

    def _feedback(self, feedback: str) -> None:
        if self.last_result is None:
            self.statusBar().showMessage("No result is available for feedback")
            return
        operator = self.operator.text().strip()
        reason = self.feedback_reason.text().strip()
        if not operator or not reason:
            self.statusBar().showMessage("Operator ID and feedback reason are required")
            return
        payload = {
            "feedback": feedback,
            "operator_id": operator,
            "reason": reason,
            "original_state": self.last_result.state.value,
            "image_path": self.last_result.image_path,
            "model_version": self.last_result.model_version,
        }
        self.events.append(
            "user_feedback",
            payload,
            sequence_id=self.last_result.sequence_id,
            panel_id=self.last_result.panel_id,
        )
        self.audit.append_event("user_feedback", payload)
        self.statusBar().showMessage("Feedback saved; model was not changed automatically")

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._stop_folder()
        self.alarm.stop()
        event.accept()


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Manufacturing Junction gateKeeper AI Vision")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()
