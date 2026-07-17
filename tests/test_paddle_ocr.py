from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from gatekeeper.inference.paddle_ocr import PaddleOcrRecognizer


def test_ocr_engine_is_reused_for_the_same_local_model(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class FakeTextRecognition:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

        def predict(self, roi: object) -> list[object]:
            return []

    model = tmp_path / "ocr" / "en_PP-OCRv4_mobile_rec"
    model.mkdir(parents=True)
    (model / "inference.yml").write_text("model: test\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules, "paddleocr", SimpleNamespace(TextRecognition=FakeTextRecognition)
    )
    PaddleOcrRecognizer._engines.clear()

    first = PaddleOcrRecognizer(model.parent)
    second = PaddleOcrRecognizer(model.parent)

    assert first.engine is second.engine
    assert len(calls) == 1
