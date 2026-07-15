from __future__ import annotations

from pathlib import Path
from typing import Any

from gatekeeper.inference.types import OcrResult


class PaddleOcrRecognizer:
    """PaddleOCR recognition-only adapter for the CPU English mobile model."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        from paddleocr import TextRecognition  # type: ignore[import-not-found]

        kwargs: dict[str, object] = {"model_name": "en_PP-OCRv4_mobile_rec"}
        if model_dir is not None:
            resolved = Path(model_dir)
            candidates = sorted(resolved.glob("**/inference.yml"))
            model_root = next(
                (candidate.parent for candidate in candidates if candidate.is_file()), resolved
            )
            kwargs["model_dir"] = str(model_root)
        self.engine = TextRecognition(**kwargs)

    def recognize(self, roi: Any) -> OcrResult:
        result = self.engine.predict(roi)
        text: str | None = None
        confidence = 0.0
        for item in result:
            payload = item.json if isinstance(item.json, dict) else item.json()
            payload = payload.get("res", payload)
            if isinstance(payload, dict):
                text = str(payload.get("rec_text", "")) or None
                confidence = float(payload.get("rec_score", 0.0))
                break
        return OcrResult(text=text, confidence=confidence)
