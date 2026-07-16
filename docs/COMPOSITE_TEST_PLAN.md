# Manufacturing Junction gateKeeper AI Vision composite test plan

This plan is the release gate for the offline Windows package. Development-only
synthetic and public data may be used to verify mechanics, but the same plan
must be repeated with an independent, site-collected holdout before a line is
approved for production use.

## Required path

1. Open a local panel image in **Image labeling workbench**.
2. Draw exactly one `fpcb_surface` rectangle and one `code_roi` rectangle.
3. Select a registered code, record a panel/lot/recipe group, and save the
   COCO rectangles, rectangular masks, OCR crop, and code label.
4. Export the group-separated YOLO dataset and verify that no group appears in
   more than one split.
5. Validate OCR labels against `config/code_recipe.json`.
6. Start CPU-only fine-tuning from **Training progress**. Confirm that the
   displayed output advances, `results.csv` is created, and stopping a run does
   not promote weights.
7. Export the selected detector to ONNX and execute the deployment path:
   image -> YOLO code ROI -> relative ROI -> PaddleOCR -> decision engine.
8. Place files in the hot folder and verify archive routing, popup replacement,
   alarm replacement, audit logs, and simulation PLC signals.
9. Capture an empty Frame Gate background, then replay a panel sequence at
   0.10, 0.15, and 0.25-second image intervals. Verify two stable frames are
   required and the sharpest stable image is the only image sent to YOLO/OCR.
10. Verify `INSPECTION_LOCK` starts at frame selection. Confirm that both the
    configured refractory time and two empty frames are required before the
    next selected panel; an early reappearance must log `unexpected_early_panel`
    and show a yellow timing warning without changing decision counters.

## Scenario matrix

| Scenario | Expected state or behavior |
|---|---|
| Registered normal code `HJ04` | `NORMAL`, green `OK`, no sound, no reject request |
| Registered problem code `HJ05` | `PROBLEM`, red `Error`, repeating alarm, one reject request |
| Unregistered valid code | `ABNORMAL`, yellow warning |
| OCR output with `O` instead of `0` | Correct only the approved O/0 difference and retain raw value in audit |
| OCR confidence below threshold | `ABNORMAL`; never promote to normal |
| Missing detector ROI | `ABNORMAL` |
| Invalid crop, model, archive, or database failure | `SYSTEM_ERROR`; normal signal prohibited |
| Partially written input file | Wait for stability before processing |
| Empty camera frame | `frame_empty`; no inference, popup, alarm, PLC request, or decision counter change |
| Moving or blurred panel frame | `frame_moving`; no inference or production decision |
| Two qualifying stable frames | Exactly one sharpest frame selected and exactly one YOLO/OCR invocation |
| Frame during refractory | `frame_suppressed`; no inference or production decision |
| New panel after empty scene but before refractory expiry | `unexpected_early_panel`, yellow timing warning, no decision |
| Second selected panel while inference candidate is pending | `SYSTEM_ERROR` with `inference_backpressure`; never silently drop it |
| 10 FPS camera sequence | Frame Gate p95 <= 20 ms, bounded queue, and suppressed raw frames deleted or sampled only |
| Duplicate image hash | Move to duplicate archive and do not issue a second decision |
| Out-of-order completed result | Older sequence cannot replace the newer popup or alarm state |
| Recipe collision after O/0 correction | Save rejected |
| OCR label outside recipe | Validation rejected |
| Same panel/lot/recipe in two splits | Export/verification rejected |

## Measured release evidence

Record these values for each tested model and dataset version:

- detector code-ROI precision and recall at IoU 0.50;
- exact four-character accuracy after the approved O/0 correction;
- per-code precision and recall;
- problem-code recall and problem false-normal count;
- p50 and p95 CPU latency;
- completion of an eight-hour hot-folder soak test without queue growth;
- model hash, code recipe snapshot, dataset manifest, and application version.

The software release gate requires at least 99% exact-code accuracy, at least
99% problem-code recall, and zero problem false-normal results on the recorded
holdout. A synthetic result verifies workflow mechanics only; it cannot certify
a factory line.
