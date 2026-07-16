from __future__ import annotations

import argparse
import json
from pathlib import Path

from gatekeeper.domain import DisplayState, Thresholds
from gatekeeper.inference.paddle_ocr import PaddleOcrRecognizer
from gatekeeper.inference.pipeline import InspectionPipeline
from gatekeeper.inference.yolo26_detector import Yolo26OnnxDetector


def labels(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        relative, code = line.rsplit("\t", 1)
        values[Path(relative).name] = code.strip().upper()
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the deployed decision pipeline")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ocr-model", type=Path, required=True)
    parser.add_argument("--candidate-confidence", type=float, default=0.01)
    parser.add_argument("--roi-confidence", type=float, default=0.01)
    parser.add_argument("--ocr-confidence", type=float, default=0.80)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    expected_labels = labels(args.dataset / "ocr" / "test.txt")
    detector = Yolo26OnnxDetector(args.model, confidence=args.candidate_confidence)
    ocr = PaddleOcrRecognizer(args.ocr_model)
    pipeline = InspectionPipeline(
        detector,
        ocr,
        expected_code="HJ04",
        problem_codes=frozenset({"HJ05"}),
        thresholds=Thresholds(
            detector_confidence=args.roi_confidence,
            normal_ocr_confidence=args.ocr_confidence,
            problem_ocr_confidence=args.ocr_confidence,
        ),
        model_version=args.model.name,
    )
    rows: list[dict[str, object]] = []
    for sequence, image in enumerate(sorted((args.dataset / "images" / "test").glob("*.png")), 1):
        result = pipeline.inspect(image, sequence)
        expected_code = expected_labels[image.name]
        expected_state = DisplayState.NORMAL if expected_code == "HJ04" else DisplayState.PROBLEM
        rows.append(
            {
                "image": image.name,
                "expected_code": expected_code,
                "expected_state": expected_state.value,
                "actual_state": result.state.value,
                "recognized_code": result.recognized_code,
                "corrected_code": result.corrected_code,
                "latency_ms": result.latency_ms,
                "reason": result.reason,
            }
        )
    normal = [row for row in rows if row["expected_state"] == DisplayState.NORMAL.value]
    problem = [row for row in rows if row["expected_state"] == DisplayState.PROBLEM.value]
    normal_correct = sum(row["actual_state"] == DisplayState.NORMAL.value for row in normal)
    problem_correct = sum(row["actual_state"] == DisplayState.PROBLEM.value for row in problem)
    false_normal = sum(
        row["expected_state"] == DisplayState.PROBLEM.value
        and row["actual_state"] == DisplayState.NORMAL.value
        for row in rows
    )
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    report = {
        "images": len(rows),
        "normal_exact_code_accuracy": normal_correct / len(normal) if normal else 0.0,
        "problem_recall": problem_correct / len(problem) if problem else 0.0,
        "problem_false_normal": false_normal,
        "p50_latency_ms": latencies[len(latencies) // 2] if latencies else 0.0,
        "p95_latency_ms": latencies[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0,
        "thresholds": {
            "candidate": args.candidate_confidence,
            "roi": args.roi_confidence,
            "ocr": args.ocr_confidence,
        },
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
