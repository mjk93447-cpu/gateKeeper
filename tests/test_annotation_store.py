from __future__ import annotations

import json

import cv2
import numpy as np

from gatekeeper.training.annotation_store import AnnotationStore, RectangleAnnotation
from gatekeeper.training.code_labels import validate_ocr_labels
from gatekeeper.training.yolo_export import export_grouped_yolo_dataset


def test_reviewed_rectangles_create_coco_ocr_and_grouped_yolo_dataset(tmp_path) -> None:
    source = tmp_path / "panel.png"
    assert cv2.imwrite(str(source), np.full((80, 160, 3), 180, dtype=np.uint8))
    store = AnnotationStore(tmp_path / "data" / "processed")
    boxes = (
        RectangleAnnotation("fpcb_surface", 10, 10, 140, 60),
        RectangleAnnotation("code_roi", 40, 25, 80, 25),
    )
    for group in ("panel-a", "panel-b", "panel-c"):
        copy = tmp_path / f"{group}.png"
        copy.write_bytes(source.read_bytes())
        store.save(copy, (160, 80), "HJ04", boxes, group)
    payload = json.loads(store.annotations_path.read_text(encoding="utf-8"))
    assert len(payload["images"]) == 3
    assert all("segmentation" in item for item in payload["annotations"])
    errors, _ = validate_ocr_labels(store.ocr_root / "train.txt", frozenset({"HJ04"}))
    assert not errors
    data_yaml = export_grouped_yolo_dataset(tmp_path / "data" / "processed")
    assert data_yaml.is_file()
    splits = json.loads((tmp_path / "data" / "processed/manifests/panel-splits.json").read_text())
    assert {item["split"] for item in splits["images"]} == {"train", "val", "test"}


def test_annotation_store_requires_both_rectangles(tmp_path) -> None:
    source = tmp_path / "panel.png"
    assert cv2.imwrite(str(source), np.zeros((20, 20, 3), dtype=np.uint8))
    store = AnnotationStore(tmp_path / "data" / "processed")
    try:
        store.save(
            source,
            (20, 20),
            "HJ04",
            (RectangleAnnotation("code_roi", 1, 1, 10, 10),),
            "panel-a",
        )
    except ValueError as exc:
        assert "one fpcb_surface" in str(exc)
    else:
        raise AssertionError("incomplete reviewed rectangles must be rejected")
