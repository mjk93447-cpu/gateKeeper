# Development-only data augmentation

The repository includes a small deterministic generator for OCR pipeline tests:

```powershell
python scripts/generate_ocr_synthetic.py `
  --output data/synthetic/ocr `
  --codes HJ04 HJ05 `
  --count 200 `
  --seed 42
```

Synthetic data is intentionally limited to development and test use. It must not
be copied into production releases, the live image archive, or the final holdout.
Use it to test the four-character label parser, training command, OCR ROI crop,
and UI decision paths. Production fine-tuning must include approved site images
with real FPCB lighting, printing, reflection and contamination.

If a public PCB dataset is used for development, keep it under `data/public/`,
record its license and source URL in the dataset manifest, and convert it into a
separate development dataset version. The release builder copies only application
assets and model artifacts; it never copies `data/public/` or `data/synthetic/`.
