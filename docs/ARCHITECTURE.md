# Manufacturing Junction gateKeeper AI Vision architecture

```text
hot-folder -> stability/duplicate guard -> YOLO26s ONNX CPU detector
           -> code ROI preprocessing -> PaddleOCR English mobile recognizer
           -> four-character code policy -> DecisionEngine
           -> result popup + alarm + SQLite/JSONL audit + PLC adapter
```

The UI never runs inference directly. `FolderWatcher` invokes an inference
worker, which emits an immutable `InspectionResult` to the Qt event loop. The
single `ResultPopup` is replaced only when a higher sequence number arrives.

The domain decision engine has no camera, UI or PLC dependency. `NORMAL`,
`ABNORMAL`, and `PROBLEM` are production decisions; model, storage, or input
failures are `SYSTEM_ERROR` and cannot become a normal pass.

Model files are external artifacts referenced by a hash-pinned manifest. Training
and production model promotion are separate operations. The PLC implementation
in this repository is simulation/protocol-neutral only; the physical I/O map and
machine safety logic remain with the PLC developer.
