from __future__ import annotations

from pathlib import Path
from typing import Any

from gatekeeper.inference.types import Detection, DetectionResult


class Yolo26OnnxDetector:
    """CPU YOLO26 ONNX adapter for the official [x1,y1,x2,y2,score,class] output."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: tuple[int, int] = (640, 640),
        class_names: tuple[str, ...] = ("fpcb_surface", "code_roi", "defect_roi"),
        confidence: float = 0.70,
    ) -> None:
        import onnxruntime as ort  # type: ignore[import-not-found]

        self.input_size = input_size
        self.class_names = class_names
        self.confidence = confidence
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image: Any) -> DetectionResult:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]

        height, width = image.shape[:2]
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        raw = self.session.run(None, {self.input_name: tensor})[0]
        rows = np.asarray(raw)[0]
        detections: list[Detection] = []
        for row in rows:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, score, class_id = map(float, row[:6])
            if score < self.confidence:
                continue
            # Checkpoint exports may use normalized or pixel coordinates.
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1, x2 = x1 * width, x2 * width
                y1, y2 = y1 * height, y2 * height
            else:
                x1, x2 = x1 * width / self.input_size[0], x2 * width / self.input_size[0]
                y1, y2 = y1 * height / self.input_size[1], y2 * height / self.input_size[1]
            class_index = int(class_id)
            label = (
                self.class_names[class_index]
                if 0 <= class_index < len(self.class_names)
                else "unknown"
            )
            detections.append(
                Detection(
                    label=label,
                    confidence=score,
                    box=(
                        max(0, int(x1)),
                        max(0, int(y1)),
                        min(width, int(x2)),
                        min(height, int(y2)),
                    ),
                )
            )
        return DetectionResult(tuple(detections))
