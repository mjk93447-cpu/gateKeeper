# Third-party component review

| Component | Purpose | License or review point |
|---|---|---|
| Ultralytics YOLO26 | Detector training and inference | AGPL-3.0 or Enterprise terms; review before closed commercial distribution |
| PCB pretrained checkpoint | Detector initialization | Check upstream model and dataset redistribution terms before release |
| PaddleOCR / PaddlePaddle | CPU OCR training and inference | Review package, model, and dictionary notices |
| ONNX Runtime | CPU detector inference | MIT; include required native provider files |
| PySide6 / Qt | Windows user interface | LGPL/GPL or commercial terms; follow the selected Qt distribution obligations |
| PyInstaller | Windows executable packaging | GPL exception; retain required notices |

This table is an engineering checklist, not legal advice. The release workflow
must preserve the exact dependency versions and include an English license and
notice bundle for every shipped artifact, including pretrained models.
