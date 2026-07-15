# Release bundle contract

Every GitHub Release produced by `.github/workflows/release.yml` is a
self-contained Windows bundle. The ZIP contains:

- the PyInstaller `gateKeeper` executable and all Python/Qt runtime libraries;
- the YOLO26s detector checkpoint and `models/manifest.json` hash;
- the YOLO26s `.pt` checkpoint used by the local CPU fine-tuning workflow;
- the complete `en_PP-OCRv4_mobile_rec` pretrained model directory;
- `config`, `docs`, `plugins`, `watch`, `archive`, `logs`, and `overlays` folders;
- `BUILD_MANIFEST.json` with every file size and SHA-256;
- English operator, training, safety, license and PLC integration manuals.

Synthetic images and downloaded public datasets are explicitly excluded from
the release bundle. They may be used to develop and test the training pipeline,
but production training and deployment use approved local site data only.

The application resolves its root from the executable directory when frozen.
It never assumes the current working directory. For source-mode diagnostics,
`GATEKEEPER_HOME` can override the discovered root.

The release job fails if either pretrained model is missing, if the detector hash
does not match the pinned value, or if the OCR model directory does not contain
an `inference.yml` artifact.

It also fails closed unless `models/manifest.json` is explicitly marked
`approved` and records at least 99.9% exact-code accuracy, at least 99.9% problem
recall, and zero problem false-normal results. A short development run or a
candidate checkpoint can therefore never be published as a production release.
