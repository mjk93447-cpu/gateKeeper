# Model and dependency review

The default initialization checkpoint is the YOLO26s PCB checkpoint recorded in
`models/manifest.example.json`. It must be copied into the local model bundle and
its SHA-256 recorded before use.

YOLO26 and the upstream Ultralytics runtime are used under the project AGPL-3.0
or-later distribution model. The PCB dataset mirror and every pretrained or
fine-tuned checkpoint retain their own redistribution conditions. Each release
must include a hash, provenance record, applicable notices, and the
corresponding source archive; do not treat an upstream checkpoint as project
copyright.

The OCR baseline is PaddleOCR `en_PP-OCRv4_mobile_rec` with the restricted
`0-9A-Z` character dictionary. Record the exact PaddleOCR/PaddlePaddle versions
in the model manifest used for a production build.
