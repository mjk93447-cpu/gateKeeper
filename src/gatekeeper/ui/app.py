from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gatekeeper.application.inspection import InspectionService
from gatekeeper.domain import DecisionEngine, InspectionInput, InspectionState, Thresholds
from gatekeeper.infrastructure.audit import JsonlDecisionSink
from gatekeeper.infrastructure.plc import SimulatedOutputPort

STATE_STYLE = {
    InspectionState.NORMAL: ("정상", "#1f9d55"),
    InspectionState.ABNORMAL: ("비정상", "#d97706"),
    InspectionState.PROBLEM: ("문제성", "#dc2626"),
}


class MetricCard(QGroupBox):
    def __init__(self, title: str, color: str) -> None:
        super().__init__(title)
        self.value = QLabel("0")
        self.value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.value.setStyleSheet(f"color: {color};")
        layout = QVBoxLayout(self)
        layout.addWidget(self.value)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("gateKeeper · AAM FPCB Code Inspection")
        self.resize(1280, 820)
        self.counts: Counter[InspectionState] = Counter()
        self.output = SimulatedOutputPort()
        self.service = InspectionService(
            DecisionEngine(), JsonlDecisionSink(Path("logs/inspection.jsonl")), self.output
        )

        root = QWidget()
        self.setCentralWidget(root)
        page = QVBoxLayout(root)

        header = QHBoxLayout()
        title = QLabel("AAM 투입 코드 검사")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self.mode = QLabel("● SIMULATION · PLC 출력 차단")
        self.mode.setStyleSheet("color:#2563eb; font-weight:600;")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.mode)
        page.addLayout(header)

        cards = QGridLayout()
        self.cards: dict[InspectionState, MetricCard] = {}
        for column, (state, (label, color)) in enumerate(STATE_STYLE.items()):
            card = MetricCard(label, color)
            self.cards[state] = card
            cards.addWidget(card, 0, column)
        page.addLayout(cards)

        body = QHBoxLayout()
        body.addWidget(self._build_control_panel(), 1)
        body.addWidget(self._build_result_panel(), 2)
        page.addLayout(body, 1)

        self.statusBar().showMessage("준비 · 시뮬레이션 입력 대기")
        self.setStyleSheet(
            "QMainWindow{background:#f5f7fb;} QGroupBox{font-weight:600; "
            "background:white; border:1px solid #dbe2ea; border-radius:8px; "
            "margin-top:10px; padding-top:8px;} QGroupBox::title{subcontrol-origin:margin; "
            "left:10px; padding:0 4px;} QPushButton{padding:9px 14px;}"
        )

    def _build_control_panel(self) -> QGroupBox:
        box = QGroupBox("검사 조건 / 입력 시뮬레이터")
        form = QFormLayout(box)
        self.expected = QLineEdit("HJ04")
        self.problems = QLineEdit("HJ05")
        self.ocr_text = QLineEdit("HJ04")
        self.det_conf = self._confidence_box(0.96)
        self.ocr_conf = self._confidence_box(0.98)
        form.addRow("정상 코드", self.expected)
        form.addRow("문제성 코드(쉼표 구분)", self.problems)
        form.addRow("모사 OCR 결과", self.ocr_text)
        form.addRow("ROI 검출 신뢰도", self.det_conf)
        form.addRow("OCR 신뢰도", self.ocr_conf)
        run_button = QPushButton("1개 패널 검사 실행")
        run_button.clicked.connect(self._inspect)
        form.addRow(run_button)
        hint = QLabel(
            "현재 화면은 실제 카메라/PLC에 연결하지 않습니다.\n"
            "판정 규칙과 작업자 흐름을 안전하게 확인하는 단계입니다."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#64748b; font-weight:400;")
        form.addRow(hint)
        return box

    def _build_result_panel(self) -> QGroupBox:
        box = QGroupBox("실시간 판정 / 감사 로그")
        layout = QVBoxLayout(box)
        self.result = QLabel("대기")
        self.result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result.setMinimumHeight(90)
        self.result.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        self.result.setStyleSheet("background:#e2e8f0; border-radius:8px; color:#334155;")
        layout.addWidget(self.result)
        confidence_row = QHBoxLayout()
        self.detector_bar = QProgressBar()
        self.detector_bar.setFormat("ROI %p%")
        self.ocr_bar = QProgressBar()
        self.ocr_bar.setFormat("OCR %p%")
        confidence_row.addWidget(self.detector_bar)
        confidence_row.addWidget(self.ocr_bar)
        layout.addLayout(confidence_row)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["상태", "정상 코드", "판독 코드", "OCR", "사유"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        layout.addWidget(self.log)
        return box

    @staticmethod
    def _confidence_box(value: float) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(0.0, 1.0)
        widget.setDecimals(2)
        widget.setSingleStep(0.01)
        widget.setValue(value)
        return widget

    def _inspect(self) -> None:
        try:
            problem_codes = frozenset(
                code.strip() for code in self.problems.text().split(",") if code.strip()
            )
            item = InspectionInput(
                expected_code=self.expected.text(),
                problem_codes=problem_codes,
                detected=self.det_conf.value() > 0,
                detector_confidence=self.det_conf.value(),
                ocr_text=self.ocr_text.text(),
                ocr_confidence=self.ocr_conf.value(),
                thresholds=Thresholds(),
            )
            decision = self.service.inspect(item)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "검사 설정 오류", str(exc))
            return

        self.counts[decision.state] += 1
        label, color = STATE_STYLE[decision.state]
        self.result.setText(label)
        self.result.setStyleSheet(f"background:{color}; border-radius:8px; color:white;")
        self.detector_bar.setValue(round(decision.detector_confidence * 100))
        self.ocr_bar.setValue(round(decision.ocr_confidence * 100))
        for state, card in self.cards.items():
            card.value.setText(str(self.counts[state]))

        row = 0
        self.table.insertRow(row)
        values = [
            label,
            decision.expected_code,
            decision.recognized_code or "-",
            f"{decision.ocr_confidence:.1%}",
            decision.reason,
        ]
        for column, value in enumerate(values):
            cell = QTableWidgetItem(value)
            if column == 0:
                cell.setForeground(QColor(color))
            self.table.setItem(row, column, cell)
        self.log.appendPlainText(
            f"[{decision.decided_at.astimezone():%H:%M:%S}] {decision.state.value} · "
            f"expected={decision.expected_code} read={decision.recognized_code or '-'} · "
            f"{decision.reason}"
        )
        self.statusBar().showMessage(
            f"마지막 PLC 요청(시뮬레이션): {self.output.last_state.value}"
        )


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("gateKeeper")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()
