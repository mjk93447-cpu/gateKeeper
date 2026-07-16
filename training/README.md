# Training configuration

Training uses the CPU-only YOLO26 and PaddleOCR runners in `scripts/` and
`src/gatekeeper/training/`. Upstream source repositories are not copied into
this project; use pinned package versions from `pyproject.toml`.

Keep project-specific configuration in this directory and preserve the exact
configuration used for every run under `runs/<run-id>/`. Use relative data roots
or the `GATEKEEPER_HOME` environment variable; never commit machine-specific
absolute paths.

All detector and OCR splits must be grouped by panel, lot, and recipe. The final
holdout must not be used for threshold selection or model promotion.
