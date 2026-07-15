# Composite training and inference test report

This report records the CPU-only development test executed before packaging.
The generated images are development artifacts under `data/synthetic/` and are
not part of the application or any release bundle.

## Test design

- Dataset: 180 train, 40 validation and 40 untouched test images.
- Codes: balanced `HJ04` (normal) and `HJ05` (registered problem code).
- Split: deterministic and disjoint by generated panel seed.
- Detector: public PCB YOLO checkpoint fine-tuned with Ultralytics on CPU at
  640 pixels, `workers=0`, `amp=False`, `seed=42`.
- OCR: PaddleOCR `en_PP-OCRv4_mobile_rec` on CPU.
- Composite path: exported YOLO ONNX coordinates -> configurable relative ROI
  crop -> PaddleOCR -> four-character decision engine.
- O/0 policy: only `O`/`0` equivalence is applied; raw and corrected values are
  both retained in the report.

## Runs

The first batch-2 run completed two full epochs before the ten-minute cutoff.
The improved batch-8 run completed three full epochs in approximately nine
minutes and promoted its best fully written checkpoint as a local candidate.
Neither candidate is production-approved.

The final deployment-path test used `models/yolo26s-fpcb-promoted.onnx` and the
untouched 40-image test set. A confidence threshold of 0.01 was used to expose
low-confidence detections for diagnosis.

## Final ONNX composite result

| Metric | Result |
| --- | ---: |
| Detector precision (IoU 0.50, all boxes) | 0.0836 |
| Detector recall (IoU 0.50, all boxes) | 0.6750 |
| Panel-best precision (IoU 0.50) | 0.6750 |
| Panel-best recall (IoU 0.50) | 0.6750 |
| OCR raw exact accuracy | 0.4750 |
| OCR exact accuracy after O/0 correction | 0.6500 |
| Normal HJ04 corrected accuracy | 0.7000 |
| Problem HJ05 precision | 1.0000 |
| Problem HJ05 recall | 0.6000 |

The result is well below the required 99.9% normal exact accuracy and 99.9%
problem recall. The candidate is therefore correctly blocked from production
promotion. The test is still valuable because it exercised the actual ONNX
deployment path, verified that O/0 correction improves recognition, and
exposed detector false positives and missed OCR boxes for the next local-data
training cycle.

## Improvements made after the run

1. Added explicit relative ROI arguments to the composite evaluator so the
   evaluation path matches the runtime configuration.
2. Corrected panel-level recall accounting: a wrong detection is now counted as
   both a false positive and a missed ground-truth panel instead of silently
   inflating recall.
3. Kept O/0 correction deliberately narrow and auditable; unrelated fuzzy
   substitutions remain disabled.
4. Added release documentation and a release workflow that bundles the pinned
   public checkpoint and OCR model, while excluding synthetic/public datasets.

## Release decision

The source, tests, workflow and packaging checks are ready to publish. A
production model release is not approved until an independently held-out,
site-collected dataset meets both 99.9% gates. The local candidate checkpoint
must not be copied into a production bundle solely because a short CPU run
completed.
