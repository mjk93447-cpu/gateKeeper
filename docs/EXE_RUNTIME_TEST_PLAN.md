# Windows EXE runtime test plan

This plan validates the packaged Windows application, not only the Python source.
It is executed in a disposable staging directory containing the EXE, its local
runtime files, the approved model package, configuration, and a local camera
hot-folder. Test images and temporary logs are not included in a release.

## Acceptance rules

- A failed required scenario blocks a release or a replacement installer.
- A warning must be visible in the UI and audit log; it must not create a
  `NORMAL` decision or a PLC simulation request.
- Every test records the EXE build hash, configuration snapshot, result,
  screenshot or log evidence, and corrective action when applicable.
- The only pass decision is an observed expected result from the packaged EXE.
  A source-level unit-test result is supporting evidence only.

## Test environment

- Windows x64 desktop at 1,920 x 1,080 display resolution.
- Local SSD staging directory and local SSD hot-folder; no network share.
- PLC output remains Simulation/Dry-run.
- Frame Gate uses a captured empty background, expected camera interval of
  150 ms, two stable frames, two empty re-arm frames, 30-ms file settle time,
  and a three-frame latest-wins queue.

## Required scenario matrix

| ID | Action | Expected packaged-EXE result | Pass criterion |
|---|---|---|---|
| EXE-01 | Launch at 1,920 x 1,080 | Dashboard is usable; no clipped controls; outer and controls scroll areas work. | All controls and Labeling/Training sections are reachable. |
| EXE-02 | Select an external local camera folder and restart the app. | The selected folder is displayed and persisted. | Start monitoring uses that exact folder. |
| EXE-03 | Click Start hot-folder once. | Models load without `No module named pandas` or PDX initialization error. | Live mode starts and audit records model load. |
| EXE-04 | Click Start hot-folder a second time. | No second OCR engine or watcher is created. | UI reports that monitoring is already running; no System Error. |
| EXE-05 | Stop, then start monitoring again. | Existing OCR engine is reused. | No `PDX has already been initialized` error. |
| EXE-06 | Capture empty background, then send empty frames. | Frames are ignored by Frame Gate. | No inference, popup replacement, decision counter, or PLC request. |
| EXE-07 | Send moving/blurred frames at 150 ms. | Frames are logged as moving and suppressed. | No NORMAL result. |
| EXE-08 | Send two sharp stationary frames at 150 ms. | One sharp frame is selected for inference. | Exactly one inspection per panel. |
| EXE-09 | Send a second panel before re-arm conditions. | Yellow timing warning and audit event. | No duplicate inference or PLC request. |
| EXE-10 | Use an unreadable or partly written image. | Frame is recorded as not ready/suppressed. | Application remains running and never emits NORMAL. |
| EXE-11 | Verify result popup and main dashboard at 1,920 x 1,080. | Popup and colored result are readable; main page remains navigable. | Screenshot evidence exists. |

## Error triage

| Symptom | First check | Required correction |
|---|---|---|
| `No module named pandas` | PyInstaller analysis and frozen dependency inventory | Include `pandas` in the inspection EXE bundle and rerun EXE-03. |
| `PDX has already been initialized` | Start/stop sequence and OCR engine construction count | Reuse one process-local OCR engine and block duplicate Start clicks; rerun EXE-04 and EXE-05. |
| Camera folder unavailable | Selected-path persistence and permissions | Select a writable local SSD directory; do not use a network share. |
| Frame Gate has no selection | Captured background, ROI, timing, sharpness, motion thresholds | Recalibrate with the actual camera scene before changing inference thresholds. |

## Release evidence

The final runtime report must list every scenario result and attach screenshots
of the main dashboard, camera-folder selection, live Frame Gate status, and a
result popup. It must also state whether real detector/OCR inference used the
approved bundled models or a controlled simulation fixture.
