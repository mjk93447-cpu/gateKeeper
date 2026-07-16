# Release bundle contract

Every GitHub Release produced by `.github/workflows/release.yml` is a
self-contained Windows bundle. The ZIP contains:

- the `Manufacturing Junction gateKeeper AI Vision` inspection executable and all required Python/Qt runtime libraries;
- the separate `Manufacturing Junction gateKeeper Training` CPU-only executable and its training runtime, so local fine-tuning works offline without enlarging the live inspection process;
- the YOLO26s detector checkpoint and `models/manifest.json` hash;
- the YOLO26s `.pt` checkpoint used by the local CPU fine-tuning workflow;
- the complete `en_PP-OCRv4_mobile_rec` pretrained model directory;
- `config`, `docs`, `plugins`, `watch`, `archive`, `logs`, and `overlays` folders;
- `BUILD_MANIFEST.json` with every file size and SHA-256;
- `LICENSE`, `NOTICE`, and a release-specific corresponding-source archive;
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
`approved` and records at least 99% exact-code accuracy, at least 99% problem
recall, and zero problem false-normal results. A short development run or a
candidate checkpoint can therefore never be published as a production release.

Before creating a tag, execute and archive the evidence described in
[COMPOSITE_TEST_PLAN.md](COMPOSITE_TEST_PLAN.md). The installer and ZIP remain
software release artifacts, not authorization to use a model on a production
line without an approved site holdout.
