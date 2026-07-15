# Bundled plugins

The release bundle includes all runtime adapters used by the MVP:

- `yolo26_detector`: CPU ONNX detector;
- `paddle_ocr`: English/numeric four-character OCR;
- `folder_watcher`: hot-folder ingestion and duplicate routing;
- `alarm`: Windows speaker alarms;
- `plc`: simulation and protocol-neutral reject request;
- `training`: CPU fine-tuning command and metric reader.

These are packaged into the application by PyInstaller. The directory is also
included in source bundles so an installation can be audited offline.
