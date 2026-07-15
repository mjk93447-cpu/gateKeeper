# CPU fine-tuning guide

## Detector

The initial checkpoint is YOLO26s PCB detection. Fine-tune it with local labels
whose required classes are `fpcb_surface` and `code_roi`; add `defect_roi` only
when the site has enough independently labelled defect samples.

Do not split adjacent frames randomly. Generate train/validation/test manifests
grouped by panel, lot and recipe. Keep at least one complete lot as a final
holdout. The holdout is not used for threshold selection.

```powershell
python scripts/validate_detection_dataset.py data/processed/detector/annotations.json
python scripts/training_commands.py yolo26-cpu `
  --data data/processed/detector/data.yaml `
  --pretrained models/yolo26s-pcb-best.pt `
  --imgsz 640 --batch 2 --epochs 100 --patience 20
```

The command explicitly uses `device=cpu`, `workers=0`, `amp=False`, disk cache and
a fixed seed. The best checkpoint is selected using the independent holdout's
ROI recall and downstream exact-code metrics, not training loss alone.

## OCR

Use PaddleOCR `en_PP-OCRv4_mobile_rec` with a restricted
`0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` dictionary. Labels must contain exactly
four characters. Validate them before training:

```powershell
python scripts/validate_ocr_dataset.py data/processed/ocr/train.txt
```

Include HJ04/HJ05 hard negatives and real variation in exposure, reflection,
blur, tilt, contamination and print defects. Never place frames from one panel
in different splits.

## Overfitting controls

- panel/lot/recipe grouped split;
- image hash duplicate check;
- fixed seed and recorded dataset manifest;
- physically plausible augmentation only;
- early stopping and weight decay;
- per-class and per-code confusion matrix;
- holdout evaluation after all threshold tuning;
- model promotion only through the model manifest with SHA-256.
