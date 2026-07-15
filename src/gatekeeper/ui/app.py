from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gatekeeper.domain import (
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
from gatekeeper.ingest.folder_watcher import ArchiveManager, FolderWatcher
from gatekeeper.runtime.paths import RuntimePaths
from gatekeeper.storage.sqlite_store import SQLiteEventStore
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
        self.setWindowTitle("gateKeeper result")
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

    def __init__(
        self, pipeline: InspectionPipeline, directory: Path, archive_directory: Path
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.archive = ArchiveManager(archive_directory)
        self.watcher = FolderWatcher(
            directory,
            self._process,
            on_duplicate=lambda path: self._archive_duplicate(path),
        )

    def start(self) -> None:
        self.watcher.start()

    def stop(self) -> None:
        self.watcher.stop()

    def _process(self, path: Path, sequence_id: int) -> None:
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

    def _archive_duplicate(self, path: Path) -> None:
        try:
            self.archive.move(path, "duplicate")
        except OSError:
            pass


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("gateKeeper - AAM FPCB Inspection")
        self.resize(1320, 860)
        self.counts: Counter[DisplayState] = Counter()
        self.last_sequence = -1
        self.last_result: InspectionResult | None = None
        self.paths = RuntimePaths.discover()
        self.output = SimulatedOutputPort()
        self.alarm: AlarmController = WindowsAlarmController()
        self.events = SQLiteEventStore(self.paths.logs / "gatekeeper.sqlite3")
        self.audit = JsonlDecisionSink(self.paths.logs / "inspection.jsonl")
        self.controller: FolderController | None = None

        root = QWidget()
        self.setCentralWidget(root)
        self.popup = ResultPopup(self)
        page = QVBoxLayout(root)
        header = QHBoxLayout()
        title = QLabel("gateKeeper | AAM FPCB Code Inspection")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        header.addWidget(title)
        header.addStretch()
        self.mode = QLabel("SIMULATION / CPU")
        self.mode.setStyleSheet("color:#2563eb; font-weight:700;")
        header.addWidget(self.mode)
        page.addLayout(header)

        body = QHBoxLayout()
        body.addWidget(self._build_controls(), 1)
        right = QVBoxLayout()
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
        body.addLayout(right, 2)
        page.addLayout(body, 1)
        page.addWidget(TrainingView(self.paths.root))
        self.statusBar().showMessage("Ready - waiting for an image")
        self.setStyleSheet(
            "QMainWindow{background:#f5f7fb;} QGroupBox{font-weight:600; background:white; "
            "border:1px solid #dbe2ea; border-radius:8px; margin-top:10px; padding-top:8px;} "
            "QGroupBox::title{subcontrol-origin:margin; left:10px; padding:0 4px;} "
            "QPushButton{padding:9px 14px;}"
        )

    def _build_controls(self) -> QGroupBox:
        box = QGroupBox("Inspection controls")
        form = QFormLayout(box)
        self.expected = QLineEdit("HJ04")
        self.problems = QLineEdit("HJ05")
        self.ocr_text = QLineEdit("HJ04")
        self.det_conf = self._confidence_box(0.96)
        self.ocr_conf = self._confidence_box(0.98)
        self.roi_x = self._confidence_box(0.0)
        self.roi_y = self._confidence_box(0.0)
        self.roi_w = self._confidence_box(1.0)
        self.roi_h = self._confidence_box(1.0)
        form.addRow("Expected code", self.expected)
        form.addRow("Problem codes", self.problems)
        form.addRow("Simulation OCR", self.ocr_text)
        form.addRow("ROI confidence", self.det_conf)
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
            problems = frozenset(
                code.strip() for code in self.problems.text().split(",") if code.strip()
            )
            decision = DecisionEngine().decide(
                InspectionInput(
                    expected_code=self.expected.text(),
                    problem_codes=problems,
                    detected=self.det_conf.value() > 0,
                    detector_confidence=self.det_conf.value(),
                    ocr_text=self.ocr_text.text(),
                    ocr_confidence=self.ocr_conf.value(),
                    thresholds=Thresholds(),
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
        try:
            registry = ModelRegistry(self.paths.models / "manifest.json")
            detector_artifact = registry.verify("detector")
            model_path = detector_artifact.path
            detector = Yolo26OnnxDetector(model_path)
            ocr = PaddleOcrRecognizer(self.paths.models / "ocr")
            pipeline = InspectionPipeline(
                detector,
                ocr,
                expected_code=self.expected.text(),
                problem_codes=frozenset(
                    code.strip() for code in self.problems.text().split(",") if code.strip()
                ),
                model_version=model_path.name,
                ocr_relative_roi=RelativeRoi(
                    x=self.roi_x.value(),
                    y=self.roi_y.value(),
                    width=self.roi_w.value(),
                    height=self.roi_h.value(),
                ),
            )
            self.controller = FolderController(pipeline, self.paths.watch, self.paths.archive)
            self.controller.result_ready.connect(self._present)
            self.controller.start()
            self.mode.setText("LIVE / CPU / HOT-FOLDER")
            self.statusBar().showMessage("Hot-folder is running: watch")
        except Exception as exc:
            self._present_error(str(exc))

    def _stop_folder(self) -> None:
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
        self.mode.setText("SIMULATION / CPU")
        self.statusBar().showMessage("Hot-folder stopped")

    def _present_error(self, reason: str) -> None:
        result = InspectionResult(
            panel_id="system",
            sequence_id=self.last_sequence + 1,
            state=DisplayState.SYSTEM_ERROR,
            expected_code=self.expected.text(),
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
    app.setApplicationName("gateKeeper")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()
