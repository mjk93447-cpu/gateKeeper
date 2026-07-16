# Manufacturing Junction gateKeeper AI Vision CPU fine-tuning guide

Use this procedure only after a reviewed local dataset and recipe have been
approved. The installed application contains an offline CPU training runner; it
does not download packages or models during training.

## Before training

1. Back up `config/code_recipe.json` and confirm every intended normal and
   problem code is registered.
2. Prepare detector annotations for `fpcb_surface` and `code_roi`. The optional
   `defect_roi` class is allowed only when enough independent samples exist.
3. Prepare OCR labels as `relative-image-path<TAB>FOUR_CHARACTER_CODE`.
4. Keep every image from one panel, lot, or recipe in exactly one split.
5. Preserve an independent lot holdout. It is not used to tune thresholds or
   select the promoted checkpoint.

Use **Image labeling workbench** for local reviewed images. It produces COCO
rectangles and rectangular masks for `fpcb_surface` and `code_roi`, crops the
code ROI for OCR, validates the selected registered code, and exports a
group-separated YOLO dataset. It does not invent labels or accept incomplete
annotations.

In the application, enter the detector YAML path and checkpoint path in
**Training progress**. Select **Validate OCR labels** before starting a run.
The label file must match the registered code recipe. **Start CPU training**
launches the bundled training executable and displays the latest training output,
epoch count, and validation graph. **Stop** terminates only the current training
process; it does not promote a checkpoint.

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
python scripts/validate_ocr_dataset.py data/processed/ocr/train.txt --recipe config/code_recipe.json
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

## Promotion checklist

Before replacing a live detector, run the composite holdout test using the
detector's `code_roi` output and the installed PaddleOCR recognition model. The
promotion record must include the following measurements on the independent
holdout:

- exact four-character code accuracy;
- problem-code recall;
- problem false-normal count;
- per-code precision and recall, including every registered problem code;
- detector ROI recall and p95 CPU latency;
- dataset version, code recipe snapshot, checkpoint hash, and command line.

Do not promote based on training loss, a random image split, or a synthetic-only
test. The release gate requires at least 99% exact-code accuracy, at least 99%
problem recall, and zero recorded problem false-normal results, but local site
approval is still required before use on a production line.

When detector candidates are sparse at a high confidence value, test a lower
candidate threshold while keeping the final ROI and OCR decision thresholds
explicit. The live pipeline ranks all retained `code_roi` candidates and passes
only the highest-confidence candidate to OCR. Never change a threshold solely to
increase a training metric; repeat the deployed decision-pipeline evaluation.
