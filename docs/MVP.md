# gateKeeper CPU MVP

## Runtime

```powershell
uv venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install -e ".[desktop,runtime,dev]"
python -m gatekeeper
```

The application starts in simulation mode. Enter `HJ04` as the expected code and
`HJ05` as a problem code, then use **Process simulation image** to exercise the
green `OK`, yellow `Abnormal`, and red `Error` states.

## Hot-folder

Place completed image files in `watch/`. A file must be fully written before it is
processed. The watcher hashes each image and ignores duplicates. Results are
recorded in `logs/gatekeeper.sqlite3` and the JSONL audit file.

The live worker is enabled only after `models/detector.onnx` and `models/ocr` have
been installed and validated. The detector must be a locally fine-tuned YOLO26s
model whose classes include `fpcb_surface` and `code_roi`.

The OCR crop can be configured as a relative rectangle inside the YOLO `code_roi`
box using `models.ocr_relative_roi` (`x`, `y`, `width`, `height`, each in 0..1).
The applied absolute crop is recorded in every inspection event.

Synthetic and public datasets are development/test-only assets. They are kept
under ignored data directories and are never copied into the application or
release bundle. Production fine-tuning must use approved site data.

## Training

Use group-separated datasets (panel/lot/recipe) and validate labels before training:

```powershell
python scripts/validate_ocr_dataset.py data/processed/ocr/train.txt
python scripts/training_commands.py yolo26-cpu `
  --data data/processed/detector/data.yaml `
  --pretrained models/yolo26s-pcb-best.pt
```

The generated command is CPU-only, disables AMP, uses workers=0 and records a
reproducible seed. Do not promote a checkpoint without an independent lot holdout.
