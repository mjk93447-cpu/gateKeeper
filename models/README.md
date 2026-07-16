# Model registry

Model binaries and production images are excluded from Git to keep the source
repository usable. Every approved GitHub Release instead includes the detector,
the CPU fine-tuning checkpoint, and the selected OCR recognition model with a
hash manifest and source/provenance notices. For a source checkout, copy the
approved locally fine-tuned YOLO26s detector and OCR model directory here, then
create `manifest.json` from `manifest.example.json` and replace every placeholder
hash.

```text
models/
  detector.onnx
  yolo26s-pcb-pretrained.pt
  ocr/
  manifest.json
```

The live worker refuses to start when the detector is missing or its SHA-256 does
not match the manifest. The detector classes used by the production pipeline are
`fpcb_surface`, `code_roi`, and optional `defect_roi`.
