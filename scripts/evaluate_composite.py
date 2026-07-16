from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from gatekeeper.inference.paddle_ocr import PaddleOcrRecognizer
from gatekeeper.inference.roi import RelativeRoi
from gatekeeper.inference.yolo26_detector import Yolo26OnnxDetector


def yolo_box_to_xyxy(values: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy, box_w, box_h = values
    return (
        round((cx - box_w / 2) * width),
        round((cy - box_h / 2) * height),
        round((cx + box_w / 2) * width),
        round((cy + box_h / 2) * height),
    )


def iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def load_ocr_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        relative, code = line.rsplit("\t", 1)
        labels[Path(relative).name] = code.strip().upper()
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate YOLO coordinates followed by PaddleOCR")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ocr-model", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("runs/composite-report.json"))
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--roi-x", type=float, default=0.0)
    parser.add_argument("--roi-y", type=float, default=0.0)
    parser.add_argument("--roi-width", type=float, default=1.0)
    parser.add_argument("--roi-height", type=float, default=1.0)
    args = parser.parse_args()
    relative_roi = RelativeRoi(args.roi_x, args.roi_y, args.roi_width, args.roi_height)
    onnx_detector = (
        Yolo26OnnxDetector(
            args.model,
            class_names=("fpcb_surface", "code_roi"),
            confidence=args.confidence,
        )
        if args.model.suffix.lower() == ".onnx"
        else None
    )
    model = None if onnx_detector else YOLO(args.model)
    ocr = PaddleOcrRecognizer(args.ocr_model)
    labels = load_ocr_labels(args.dataset / "ocr" / "test.txt")
    tp = fp = fn = panel_tp = panel_fp = panel_fn = 0
    raw_exact = corrected_exact = normal_total = normal_exact = problem_total = 0
    problem_tp = problem_fp = problem_fn = 0
    rows: list[dict[str, object]] = []
    for image_path in sorted((args.dataset / "images" / "test").glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        label_line = (args.dataset / "labels" / "test" / f"{image_path.stem}.txt").read_text(
            encoding="utf-8"
        ).splitlines()[1]
        ground_truth_box = yolo_box_to_xyxy(
            [float(value) for value in label_line.split()[1:]], width, height
        )
        predictions: list[tuple[float, tuple[int, int, int, int]]] = []
        if onnx_detector is not None:
            detections = onnx_detector.detect(image).detections
            predictions = [
                (item.confidence, item.box) for item in detections if item.label == "code_roi"
            ]
        else:
            result = model.predict(
                image, imgsz=640, conf=args.confidence, device="cpu", verbose=False
            )[0]
            for box in result.boxes:
                if int(box.cls.item()) != 1:
                    continue
                coords = tuple(round(value) for value in box.xyxy[0].tolist())
                predictions.append((float(box.conf.item()), coords))
        best = max(predictions, default=None, key=lambda item: item[0])
        matched = best is not None and iou(best[1], ground_truth_box) >= 0.5
        if matched:
            panel_tp += 1
        else:
            # A wrong detection is both a false positive and a missed panel.
            # Counting it as FN keeps panel recall honest instead of treating
            # an incorrect box as if the panel had no ground-truth miss.
            panel_fn += 1
            if best is not None:
                panel_fp += 1
        if matched:
            tp += 1
            fp += max(0, len(predictions) - 1)
        else:
            fn += 1
            fp += len(predictions)
        expected = labels[image_path.name]
        recognized: str | None = None
        confidence = 0.0
        if best is not None:
            x1, y1, x2, y2 = relative_roi.apply(best[1])
            crop = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            if crop.size:
                ocr_result = ocr.recognize(crop)
                recognized = (ocr_result.text or "").replace(" ", "").upper() or None
                confidence = ocr_result.confidence
        corrected = recognized.replace("O", "0") if recognized else None
        expected_key = expected.replace("O", "0")
        is_raw_exact = recognized == expected
        is_exact = corrected == expected_key
        raw_exact += int(is_raw_exact)
        corrected_exact += int(is_exact)
        if expected == "HJ04":
            normal_total += 1
            normal_exact += int(is_exact)
        else:
            problem_total += 1
            problem_tp += int(corrected == "HJ05")
            problem_fn += int(corrected != "HJ05")
        problem_fp += int(expected != "HJ05" and corrected == "HJ05")
        rows.append(
            {
                "image": image_path.name,
                "expected": expected,
                "recognized": recognized,
                "corrected": corrected,
                "ocr_confidence": confidence,
                "detections": len(predictions),
                "matched_iou50": matched,
                "best_iou": iou(best[1], ground_truth_box) if best else 0.0,
                "roi_relative": {
                    "x": relative_roi.x,
                    "y": relative_roi.y,
                    "width": relative_roi.width,
                    "height": relative_roi.height,
                },
            }
        )
    count = len(rows)
    report = {
        "images": count,
        "detector_precision_iou50": tp / (tp + fp) if tp + fp else 0.0,
        "detector_recall_iou50": tp / (tp + fn) if tp + fn else 0.0,
        "panel_best_precision_iou50": panel_tp / (panel_tp + panel_fp)
        if panel_tp + panel_fp
        else 0.0,
        "panel_best_recall_iou50": panel_tp / (panel_tp + panel_fn)
        if panel_tp + panel_fn
        else 0.0,
        "ocr_raw_exact_accuracy": raw_exact / count if count else 0.0,
        "ocr_o0_corrected_exact_accuracy": corrected_exact / count if count else 0.0,
        "normal_hj04_corrected_accuracy": normal_exact / normal_total if normal_total else 0.0,
        "problem_hj05_precision": (
            problem_tp / (problem_tp + problem_fp) if problem_tp + problem_fp else 0.0
        ),
        "problem_hj05_recall": (
            problem_tp / (problem_tp + problem_fn) if problem_tp + problem_fn else 0.0
        ),
        "counts": {"tp": tp, "fp": fp, "fn": fn},
        "samples": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
