from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gatekeeper.domain.code_recipe import CodeRecipe
from gatekeeper.training.annotation_store import AnnotationStore, RectangleAnnotation
from gatekeeper.training.yolo_export import export_grouped_yolo_dataset


class AnnotationCanvas(QWidget):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(520, 300)
        self.setMouseTracking(True)
        self.source = QPixmap()
        self.boxes: dict[str, QRect] = {}
        self.active_kind = "fpcb_surface"
        self.drag_start: QPoint | None = None
        self.dragging: QRect | None = None

    def load_image(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            raise ValueError(f"unable to render image: {path}")
        self.source = pixmap
        self.boxes = {}
        self.drag_start = None
        self.dragging = None
        self.update()
        self.changed.emit()

    def source_size(self) -> tuple[int, int]:
        return self.source.width(), self.source.height()

    def annotation_rectangles(self) -> tuple[RectangleAnnotation, ...]:
        rectangles: list[RectangleAnnotation] = []
        for category, box in self.boxes.items():
            rectangles.append(
                RectangleAnnotation(category, box.x(), box.y(), box.width(), box.height())
            )
        return tuple(rectangles)

    def _display_rect(self) -> QRect:
        if self.source.isNull():
            return QRect()
        size = self.source.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRect(
            (self.width() - size.width()) // 2,
            (self.height() - size.height()) // 2,
            size.width(),
            size.height(),
        )

    def _to_source(self, point: QPoint) -> QPoint | None:
        display = self._display_rect()
        if not display.contains(point) or display.width() <= 0 or display.height() <= 0:
            return None
        x = round((point.x() - display.x()) * self.source.width() / display.width())
        y = round((point.y() - display.y()) * self.source.height() / display.height())
        return QPoint(
            max(0, min(x, self.source.width() - 1)),
            max(0, min(y, self.source.height() - 1)),
        )

    def _to_display(self, box: QRect) -> QRect:
        display = self._display_rect()
        if display.width() <= 0 or display.height() <= 0:
            return QRect()
        return QRect(
            display.x() + round(box.x() * display.width() / self.source.width()),
            display.y() + round(box.y() * display.height() / self.source.height()),
            round(box.width() * display.width() / self.source.width()),
            round(box.height() * display.height() / self.source.height()),
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start = self._to_source(event.position().toPoint())
            self.dragging = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.drag_start is None:
            return
        current = self._to_source(event.position().toPoint())
        if current is not None:
            self.dragging = QRect(self.drag_start, current).normalized()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self.drag_start is None:
            return
        current = self._to_source(event.position().toPoint())
        if current is not None:
            box = QRect(self.drag_start, current).normalized()
            if box.width() > 1 and box.height() > 1:
                self.boxes[self.active_kind] = box
                self.changed.emit()
        self.drag_start = None
        self.dragging = None
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#e2e8f0"))
        if self.source.isNull():
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a local panel image")
            return
        display = self._display_rect()
        painter.drawPixmap(display, self.source)
        for category, box in self.boxes.items():
            color = QColor("#2563eb") if category == "fpcb_surface" else QColor("#dc2626")
            painter.setPen(QPen(color, 3))
            painter.drawRect(self._to_display(box))
            painter.drawText(self._to_display(box).topLeft() + QPoint(4, 18), category)
        if self.dragging is not None:
            painter.setPen(QPen(QColor("#16a34a"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(self._to_display(self.dragging))


class LabelingView(QGroupBox):
    """Local image labeling workbench with rectangular FPCB and code ROI masks."""

    def __init__(self, root: Path, recipe_path: Path) -> None:
        super().__init__("Image labeling workbench")
        self.root = root
        self.recipe_path = recipe_path
        self.image_path: Path | None = None
        self.canvas = AnnotationCanvas()
        self.canvas.changed.connect(self._refresh_status)
        self.kind = QComboBox()
        self.kind.addItems(["fpcb_surface", "code_roi"])
        self.kind.currentTextChanged.connect(self._set_kind)
        self.code = QComboBox()
        self.group = QLineEdit()
        self.status = QLabel("Open a local panel image, then draw both rectangular masks.")
        self.status.setWordWrap(True)
        open_image = QPushButton("Open local panel image")
        open_image.clicked.connect(self.open_image)
        save = QPushButton("Save reviewed label")
        save.clicked.connect(self.save_label)
        export = QPushButton("Export grouped YOLO dataset")
        export.clicked.connect(self.export_dataset)
        refresh_codes = QPushButton("Refresh code recipe")
        refresh_codes.clicked.connect(self.refresh_codes)

        form = QFormLayout()
        form.addRow("Rectangle to draw", self.kind)
        form.addRow("OCR code label", self.code)
        form.addRow("Panel / lot / recipe group", self.group)
        form.addRow(open_image)
        form.addRow(save)
        form.addRow(export)
        form.addRow(refresh_codes)
        layout = QHBoxLayout(self)
        layout.addWidget(self.canvas, 3)
        controls = QVBoxLayout()
        controls.addLayout(form)
        controls.addWidget(self.status)
        controls.addStretch()
        layout.addLayout(controls, 2)
        self.refresh_codes()

    def refresh_codes(self) -> None:
        try:
            recipe = CodeRecipe.load(self.recipe_path)
            selected = self.code.currentText()
            self.code.clear()
            self.code.addItems(sorted(recipe.all_codes))
            active = selected if selected in recipe.all_codes else recipe.active_normal_code
            self.code.setCurrentText(active)
        except (OSError, ValueError, TypeError) as exc:
            self.status.setText(f"Code recipe error: {exc}")

    def _set_kind(self, value: str) -> None:
        self.canvas.active_kind = value

    def open_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open local panel image",
            str(self.root),
            "Image files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not filename:
            return
        try:
            self.image_path = Path(filename)
            self.canvas.load_image(self.image_path)
            self.status.setText(
                "Draw fpcb_surface, then code_roi. Use the selector before each drag."
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Image open failed", str(exc))

    def save_label(self) -> None:
        if self.image_path is None:
            QMessageBox.warning(self, "Label save", "Open a local panel image first.")
            return
        try:
            recipe = CodeRecipe.load(self.recipe_path)
            code = self.code.currentText()
            if code not in recipe.all_codes:
                raise ValueError("select a code registered in the active code recipe")
            path = AnnotationStore(self.root / "data" / "processed").save(
                self.image_path,
                self.canvas.source_size(),
                code,
                self.canvas.annotation_rectangles(),
                self.group.text(),
            )
            self.status.setText(
                f"Saved COCO rectangles, rectangular masks, and OCR crop to {path.parent}."
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Label save failed", str(exc))

    def _refresh_status(self) -> None:
        kinds = ", ".join(sorted(self.canvas.boxes)) or "none"
        self.status.setText(f"Current rectangles: {kinds}")

    def export_dataset(self) -> None:
        try:
            path = export_grouped_yolo_dataset(self.root / "data" / "processed")
            self.status.setText(f"Grouped YOLO dataset exported: {path}")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "YOLO dataset export failed", str(exc))
