# Model and dependency review

The default initialization checkpoint is the YOLO26s PCB checkpoint recorded in
`models/manifest.example.json`. It must be copied into the local model bundle and
its SHA-256 recorded before use.

YOLO26 and the upstream Ultralytics runtime have licensing requirements that must
be reviewed before distributing a closed commercial Windows application. The
PCB dataset mirror also has its own terms. Keep model files out of Git and ship
only approved, hash-pinned artifacts.

The OCR baseline is PaddleOCR `en_PP-OCRv4_mobile_rec` with the restricted
`0-9A-Z` character dictionary. Record the exact PaddleOCR/PaddlePaddle versions
in the model manifest used for a production build.
