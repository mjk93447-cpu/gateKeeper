# gateKeeper

CPU-only Windows MVP for OLED AAM FPCB model-code inspection.

gateKeeper watches a hot-folder, detects the FPCB/code ROI with a locally
fine-tuned YOLO26s model, reads the four-character `A-Z0-9` code with
PaddleOCR `en_PP-OCRv4_mobile_rec`, and produces one of three safe decisions:

- `NORMAL` (green `OK`), no alarm;
- `ABNORMAL` (yellow `Abnormal`), warning sound;
- `PROBLEM` (red `Error`), repeating alarm and a simulated/protocol-neutral PLC
  `RejectRequest`.

Application/model/storage failures are shown as dark-red `System Error` and are
never treated as a normal pass. Each new image replaces the previous result
popup. The last result remains visible until the next image is processed.

## Local setup

```powershell
uv venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install -e ".[desktop,runtime,dev]"
python -m gatekeeper
```

The application starts in simulation mode. Use the controls to exercise HJ04,
HJ05 and uncertain OCR results. Live hot-folder mode requires a hash-pinned
local model manifest at `models/manifest.json`, `models/detector.onnx`, and the
PaddleOCR model directory.

## CPU fine-tuning

```powershell
python scripts/validate_detection_dataset.py data/processed/detector/annotations.json
python scripts/validate_ocr_dataset.py data/processed/ocr/train.txt
python scripts/training_commands.py yolo26-cpu `
  --data data/processed/detector/data.yaml `
  --pretrained models/yolo26s-pcb-best.pt
```

Keep panel/lot/recipe groups together across train/validation/test and retain an
independent lot holdout. See [docs/TRAINING.md](docs/TRAINING.md).

## Documentation

- [MVP operation](docs/MVP.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Operator guide](docs/OPERATOR_GUIDE.md)
- [PLC integration contract](docs/PLC_INTEGRATION.md)
- [Model and dependency licenses](docs/MODEL_LICENSES.md)

The repository does not contain model binaries, production images, or physical
PLC addresses. Those artifacts remain local and must be approved and hash-pinned
before deployment.
