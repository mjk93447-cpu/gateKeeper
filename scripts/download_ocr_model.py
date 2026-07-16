from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    target = Path(os.environ.get("GATEKEEPER_OCR_DIR", "models/ocr"))
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(target.resolve()))
    from paddleocr import PaddleOCR  # type: ignore[import-not-found]

    PaddleOCR(
        lang="en",
        text_recognition_model_name="en_PP-OCRv4_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    if not any(target.rglob("inference.yml")):
        raise RuntimeError(f"OCR model was not downloaded into {target}")
    print(f"OCR model bundle ready at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
