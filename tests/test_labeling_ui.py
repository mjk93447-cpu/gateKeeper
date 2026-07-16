from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication

from gatekeeper.ui.labeling_view import LabelingView


def test_labeling_workbench_saves_reviewed_rois_and_ocr_crop(tmp_path) -> None:
    application = QApplication.instance() or QApplication([])
    image = tmp_path / "panel.png"
    assert cv2.imwrite(str(image), np.full((80, 160, 3), 200, dtype=np.uint8))
    workbench = LabelingView(tmp_path, tmp_path / "config/code_recipe.json")
    workbench.image_path = Path(image)
    workbench.canvas.load_image(image)
    workbench.canvas.boxes = {
        "fpcb_surface": QRect(10, 10, 140, 60),
        "code_roi": QRect(40, 25, 80, 25),
    }
    workbench.group.setText("lot-01")
    workbench.save_label()
    assert (tmp_path / "data/processed/detector/annotations.json").is_file()
    assert (tmp_path / "data/processed/ocr/train.txt").is_file()
    workbench.close()
    application.processEvents()
