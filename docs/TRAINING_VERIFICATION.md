# Training verification procedure

The training path must be verified in three stages before a checkpoint can be
used by the live worker.

## 1. Static validation

```powershell
python scripts/validate_detection_dataset.py data/processed/detector/annotations.json
python scripts/validate_ocr_dataset.py data/processed/ocr/train.txt
python scripts/verify_training.py `
  --data data/processed/detector/data.yaml `
  --pretrained models/yolo26s-pcb-best.pt `
  --groups data/manifests/panel-splits.json
```

The group manifest must contain train, val and test records. A panel, lot, or
recipe group appearing in more than one split is a hard failure.

## 2. CPU smoke training

Run the same command with `--smoke`. It performs one epoch with workers=0,
`device=cpu`, and `amp=False`. Verify that `results.csv`, `weights/best.pt`, and
the run metadata are created under the configured relative `runs/` directory.

## 3. Full training and promotion

Use the generated 100-epoch command with early stopping. Inspect the training
progress graph and validation metrics after every run. Promote a checkpoint only
when the independent lot holdout meets all of the following:

- detector ROI recall target;
- four-character exact-code accuracy target;
- HJ04/HJ05 confusion and problem recall target;
- false-normal rate target;
- CPU p95 latency target.

Record the dataset version, command line, package lock, model hash, and holdout
metrics in the model manifest. The live worker refuses unverified or hash-mismatched
detectors.
