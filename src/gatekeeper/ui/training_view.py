from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QProcess, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gatekeeper.training.cpu_runner import CpuTrainingConfig, build_yolo26_command


class MetricGraph(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[float] = []
        self.setMinimumHeight(140)

    def set_values(self, values: list[float]) -> None:
        self.values = values
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if len(self.values) < 2:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No training metrics")
            return
        low, high = min(self.values), max(self.values)
        span = max(high - low, 1e-9)
        width = max(1, self.width() - 20)
        height = max(1, self.height() - 20)
        points = [
            QPointF(
                10 + width * index / (len(self.values) - 1),
                10 + height * (1 - (value - low) / span),
            )
            for index, value in enumerate(self.values)
        ]
        painter.setPen(QPen(QColor("#2563eb"), 2))
        for start, end in zip(points, points[1:], strict=False):
            painter.drawLine(start, end)


class TrainingView(QGroupBox):
    def __init__(self, root: Path | None = None) -> None:
        super().__init__("Training progress")
        metrics_path = (root or Path.cwd()) / "runs/gatekeeper/yolo26s_fpcb_cpu/results.csv"
        self.path = QLineEdit(str(metrics_path))
        self.data_yaml = QLineEdit(str((root or Path.cwd()) / "data/processed/detector/data.yaml"))
        self.pretrained = QLineEdit(str((root or Path.cwd()) / "models/yolo26s-pcb-best.pt"))
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.readyReadStandardError.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(2000)
        self.refresh_timer.timeout.connect(self.refresh)
        self.progress = QProgressBar()
        self.progress.setFormat("Epoch %v / %m")
        self.summary = QLabel("Waiting for CPU training metrics")
        self.graph = MetricGraph()
        refresh = QPushButton("Refresh metrics")
        refresh.clicked.connect(self.refresh)
        self.start_button = QPushButton("Start CPU training")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        training_row = QHBoxLayout()
        training_row.addWidget(QLabel("Dataset"))
        training_row.addWidget(self.data_yaml, 1)
        training_row.addWidget(QLabel("Checkpoint"))
        training_row.addWidget(self.pretrained, 1)
        training_row.addWidget(self.start_button)
        training_row.addWidget(self.stop_button)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path, 1)
        path_row.addWidget(refresh)
        layout = QVBoxLayout(self)
        layout.addLayout(training_row)
        layout.addLayout(path_row)
        layout.addWidget(self.progress)
        layout.addWidget(self.summary)
        layout.addWidget(self.graph)

    def start_training(self) -> None:
        config = CpuTrainingConfig(
            data_yaml=Path(self.data_yaml.text()),
            pretrained=Path(self.pretrained.text()),
            output_dir=Path(self.path.text()).parent.parent,
        )
        command = build_yolo26_command(config)
        self.process.start(command[0], command[1:])
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.summary.setText("CPU training started")
        self.refresh_timer.start()

    def stop_training(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
        self._finished(-1, QProcess.ExitStatus.CrashExit)

    def _read_output(self) -> None:
        output = bytes(self.process.readAllStandardOutput()).decode(errors="replace").strip()
        error = bytes(self.process.readAllStandardError()).decode(errors="replace").strip()
        text = output or error
        if text:
            self.summary.setText(text.splitlines()[-1][-240:])

    def _finished(self, exit_code: int, status: Any) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.refresh_timer.stop()
        self.refresh()
        self.summary.setText(f"CPU training finished with exit code {exit_code}")

    def refresh(self) -> None:
        path = Path(self.path.text())
        if not path.is_file():
            self.summary.setText(f"Metrics file not found: {path}")
            return
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            self.summary.setText("Metrics file is empty")
            return
        metric_name = next(
            (
                name
                for name in rows[0]
                if "map50" in name.lower() or "fitness" in name.lower()
            ),
            next(iter(rows[0])),
        )
        values: list[float] = []
        for row in rows:
            try:
                values.append(float(row[metric_name]))
            except (KeyError, TypeError, ValueError):
                continue
        self.progress.setMaximum(max(1, len(rows)))
        self.progress.setValue(len(rows))
        self.graph.set_values(values)
        latest = values[-1] if values else 0.0
        self.summary.setText(
            f"Epochs {len(rows)} | {metric_name.strip()}={latest:.4f} | CPU mode"
        )
