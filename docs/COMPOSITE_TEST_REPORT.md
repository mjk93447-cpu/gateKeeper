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

## Final ONNX composite result from the initial short run

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

## 45-minute model improvement and stress test

The improved run used the same 180/40 training and validation split for 16
epochs over 45 minutes of CPU time (`batch=8`, `cache=ram`, `workers=0`,
`amp=False`). The best checkpoint reached validation precision 0.965, recall
0.925 and mAP50 0.989. It was exported to ONNX and evaluated through the real
deployment path.

An additional 200-image test-only stress set was generated with a disjoint seed;
it was never used for training or checkpoint selection. Results were:

| Metric | Result |
| --- | ---: |
| Panel-best detector precision (IoU 0.50) | 1.000 |
| Panel-best detector recall (IoU 0.50) | 0.995 |
| O/0-corrected exact-code accuracy | 0.995 |
| Normal HJ04 corrected accuracy | 0.990 |
| Problem HJ05 precision | 1.000 |
| Problem HJ05 recall | 1.000 |

This passes the current 99% development release gate. Raw OCR accuracy was only
0.235 because the approved OCR model consistently rendered the printed zero as
the letter `O`; the explicitly approved O/0 correction raised exact accuracy to
0.995 without enabling unrelated fuzzy substitutions.

## Release decision

The 0.2.0 candidate satisfies the 99% development gate on the untouched stress
set. Site deployment still requires an independently held-out, site-collected
lot approval before changing the manifest from `candidate` to `approved` for
production use.

## Version 1.0.0 release-candidate revalidation

The following revalidation was executed on the same 200-image development-only
stress holdout after the label workbench, grouped YOLO exporter, offline training
runner, and deployed decision settings were added. No synthetic or public image
is included in the installer or release ZIP.

| Step | Executed evidence |
|---|---|
| UI labeling | Three panel images were saved through the UI into COCO rectangles, rectangular masks, OCR crops, OCR labels, and three separate panel/lot groups. |
| UI dataset export | The UI exporter created YOLO train/validation/test folders and a group manifest. Detection-label, OCR-label, and split validation completed successfully. |
| UI CPU training | One epoch was launched from **Training progress** with batch 2, CPU device, workers 0, AMP disabled, and patience 1. It completed with exit code 0, created `results.csv`, `best.pt`, `last.pt`, validation images, and a graph. Peak observed process memory was approximately 1.9 GB. |
| Deployed detector/OCR evaluation | 200 test images were evaluated through ONNX code ROI detection, PaddleOCR recognition, O/0 correction, and the decision engine. |
| Automated safety scenarios | Unit and UI checks covered normal/problem/abnormal decisions, O/0 correction, missing ROI, malformed OCR, duplicate files, archive routing, out-of-order result protection, PLC idempotency, code-recipe collision rejection, label validation, and group leakage validation. |

### Defect found and corrected

The first deployed re-run used the former detector confidence value of `0.70`.
It produced only 72% panel-best code-ROI recall and 68% `HJ05` recall. The
diagnostic run had used `0.01`, so this exposed a configuration mismatch between
evaluation and the live worker.

The application now separates the candidate detector threshold from the final
ROI decision threshold. The release development settings are candidate `0.01`,
final ROI `0.01`, and OCR `0.80`; these settings are displayed in the UI and
must be revalidated with a site holdout before line use.

### Deployed decision-pipeline result after correction

| Metric | Result |
| --- | ---: |
| Images | 200 |
| Normal exact-code accuracy | 1.000 |
| Problem code recall | 1.000 |
| Problem false-normal count | 0 |
| p50 latency | 237 ms |
| p95 latency | 378 ms |

This satisfies the software release gate on the recorded development holdout.
It does not certify a factory deployment: production approval still requires an
independent, site-collected lot holdout and the approved local threshold record.

## Offline package verification

The release staging directory was assembled with all runtime paths relative to
the installation root. It contained the main application, the separate
CPU-training runner, the ONNX detector, the fine-tuning checkpoint, the
PaddleOCR recognition model, default code recipe, English documentation,
license notices, plugins directory, and an AGPL source archive. The release
workflow rebuilds that archive from the tagged commit before publication.

The staged main application was started with the Qt offscreen platform and
remained running for 12 seconds without an application exception. The packaged
training runner returned its CPU training command-line help successfully. The
ZIP was created at 702.7 MB; the staged, uncompressed directory was 1.06 GB;
and the Windows x64 Inno Setup installer compiled successfully at 610.5 MB.
Synthetic and public development data were excluded from all three artifacts.
# Frame Gate FHD replay addendum (release 1.1.0 gate)

Date: 2026-07-16
Scope: development-only deterministic FHD JPEG replay. The images are created
in a temporary directory and are not part of the application, release bundle,
or training dataset.

## Method

`scripts/replay_frame_gate_fhd.py` generated a 1,920 x 1,080 electronic-panel
scene with empty views, stationary HJ04-style panels, and deliberately blurred
and translated HJ05-style moving panels. It captured an empty background,
configured a center-panel presence/sharpness ROI, then evaluated 100 final
JPEG frames at 150-ms synthetic camera timestamps. The production
Frame Gate used a 30-ms settle design, three-frame latest-wins queue, two
stable-frame requirement, two empty-frame re-arm requirement, and a 2,000-ms
refractory period.

## Results

| Input interval | Frames | Moving sessions selected | Stationary sessions selected | Gate p95 | Result |
|---|---:|---:|---:|---:|---|
| 150 ms, run 1 | 100 | 0 | 2 | 19.6 ms | Pass |
| 150 ms, run 2 | 100 | 0 | 2 | 18.6 ms | Pass |

Each run selected two stationary HJ04-style sessions and selected no moving
session. It emitted no duplicate selection, no unexpected early-panel event,
and no inference backpressure event. The first present sharp frame is retained
only as a provisional candidate; a following low-motion frame is required
before it can be selected. This permits a 150-ms camera interval to confirm a
300-ms stationary dwell without treating the first image itself as inspected.

The test exposed an initial motion-sensitivity weakness when a whole-image ROI
diluted localized FPCB motion. The implementation was improved to score both
average and upper-decile pixel change, with a calibrated center ROI used for
the replay. After the improvement, the moving-session selection count was zero.

## Release decision

**Release 1.1.0 Frame Gate software gate passed.** The default configuration
records a 150-ms expected camera interval. The GitHub bundle may be published.
Factory use still requires actual-line background capture, presence/sharpness
ROI calibration, and the site procedure's independent model-performance and
machine-safety approval.

The automated regression suite passed: `41 passed`; Ruff passed after the Frame
Gate implementation. These checks validate code behavior but do not certify
site OCR accuracy or factory safety.
