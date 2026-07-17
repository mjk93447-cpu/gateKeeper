from __future__ import annotations

import os
import unicodedata
from pathlib import Path
from threading import Lock
from typing import Any

from gatekeeper.inference.types import OcrResult

# The deployed application is intentionally offline. Prevent Paddle's optional
# model-host connectivity probe before importing any PaddleOCR component.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# `idna` imports this extension dynamically inside PaddleX's request helper.
# Keep a direct import so PyInstaller always collects the CPython extension
# into the offline one-folder application.
_UNICODEDATA_EXTENSION = unicodedata


class PaddleOcrRecognizer:
    """PaddleOCR recognition-only adapter for the CPU English mobile model."""

    _engines: dict[str, Any] = {}
    _engine_lock = Lock()

    def __init__(self, model_dir: str | Path | None = None) -> None:
        kwargs: dict[str, object] = {"model_name": "en_PP-OCRv4_mobile_rec"}
        engine_key = "builtin:en_PP-OCRv4_mobile_rec"
        if model_dir is not None:
            resolved = Path(model_dir).resolve()
            candidates = sorted(resolved.glob("**/inference.yml"))
            model_root = next(
                (candidate.parent for candidate in candidates if candidate.is_file()), resolved
            )
            kwargs["model_dir"] = str(model_root)
            engine_key = str(model_root.resolve())
        # PaddleX/PDX permits only one initialization per process. Reuse the
        # recognition engine across hot-folder stops/starts and reject repeated
        # Start clicks at the UI boundary instead of constructing another one.
        with self._engine_lock:
            engine = self._engines.get(engine_key)
            if engine is None:
                from paddleocr import TextRecognition  # type: ignore[import-not-found]

                engine = TextRecognition(**kwargs)
                self._engines[engine_key] = engine
        self.engine = engine

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
