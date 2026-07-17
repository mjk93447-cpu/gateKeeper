# Manufacturing Junction gateKeeper AI Vision operator guide

This guide is for the line engineer operating the local Windows application.
It describes the PC-side inspection only. It does not replace the safety PLC or
the line's approved work instructions.

## 1. Start-of-shift check

1. Start **Manufacturing Junction gateKeeper AI Vision** from the Windows Start
   menu or desktop shortcut.
2. Confirm that the header says `SIMULATION / CPU` before starting live input.
3. Check the selected **Expected code**. A new installation starts with `HJ04`.
4. Open **Manage code recipe** and confirm that `HJ05` appears under **Problem
   codes**. Do not change codes without the approved recipe change record.
5. Confirm that the model files are present and the status bar has no error.
6. Run one simulation image with OCR text `HJ04`. The result must be green `OK`
   and no alarm must remain active.
7. Run one simulation image with OCR text `HJ05`. The result must be red `Error`;
   the repeating alarm and simulated `RejectRequest` must be recorded.

If either check fails, do not start live inspection. Record the error and follow
the line escalation procedure.

## 2. Manage the code recipe

Select **Manage code recipe** to maintain the codes used by both inspection and
OCR training-label validation.

1. Use **Add**, **Edit**, or **Delete** under the required list.
2. Put passable codes in **Normal codes** and reject-required codes in **Problem
   codes**.
3. Select one normal code and choose **Set active**. It becomes the expected code
   used by the next inspection.
4. Select **Save**. The recipe is saved to `config/code_recipe.json` next to the
   installed application.

Rules enforced by the application:

- Every code is exactly four uppercase English letters or digits.
- At least one normal code is required.
- A normal code cannot also be a problem code.
- Codes that differ only by `O` and `0` cannot be placed in opposite lists.

The default normal code is `HJ04`; the default problem code is `HJ05`.

## 3. Set the OCR crop

The detector first finds a `code_roi` rectangle. The four relative OCR fields
define a second rectangle inside that detected box:

- `X` and `Y` are the relative top-left coordinates.
- `Width` and `Height` are relative dimensions.
- Each value is between `0.00` and `1.00`.

For example, `X=0.10`, `Y=0.20`, `Width=0.70`, `Height=0.40` reads the central
70% by 40% area of the detector box. Use approved review images when changing
these values. The applied crop is recorded in the audit event.

The installed configuration also separates **Detection candidate confidence**
from **Final ROI confidence**. The first value keeps low-confidence candidates
available for ranking; the second value is the decision threshold. Change either
value only after an approved holdout test, and record the before/after recipe and
metrics. The development bundle defaults are calibrated only for its included
development model and are not a substitute for site validation.

## 4. Start live inspection

1. Under **Camera hot-folder**, select **Choose folder** and select the local
   SSD directory where the camera writes completed image files. The selected
   path is shown in the control panel and is saved for the next application
   start. Stop monitoring before changing this directory.
2. Select **Start hot-folder**.
3. Confirm the header changes to `LIVE / CPU / HOT-FOLDER / FRAME GATE`.
4. With no panel in view, select **Capture Empty Background**. Keep the view
   empty until the next received camera image confirms the capture. This is a
   line-specific calibration; repeat it after changing camera position,
   lighting, presence ROI, or image resolution.
5. Set the camera application to save a unique final JPG/PNG every **150 ms**.
   Then open **Frame Gate Settings** and confirm **Expected camera interval** is
   `150 ms`; it is a monitoring check and does not control the camera itself.
   Record the line-approved refractory period. The default is 2,000 ms, but it is fully manual from 0 to 60,000 ms. The
   settings panel also defines the presence ROI, sharpness ROI, two stable-frame
   confirmation, two empty frames to re-arm, a 30-ms final-file settle check,
   and a three-frame latest-wins queue. Restart hot-folder monitoring after
   changing settings.
6. Check the Frame Gate status line during the first panels. It reports session
   state, selected/suppressed frames, queue depth, and p95 gate latency. A
   yellow **Timing Warning** means an early panel or unsafe observed timing was
   logged; it is not an inspection result. A p95 above 20 ms means the PC did
   not meet the Frame Gate performance target and needs site review.
7. Do not rename, move, or edit an image while it is being written. The camera
   must create unique final JPG/PNG names in a local SSD `watch` folder.
8. Check the latest popup and audit log during the first panels of the shift.

Selected images are moved to `archive/normal`, `archive/abnormal`,
`archive/problem`, or `archive/error`. Suppressed raw frames are deleted after
metadata logging by default to protect disk capacity; the optional diagnostic
sampling setting retains one every N under `archive/ignored`.

### Frame Gate operating rules

The camera may produce rapid images, but one physical panel is inspected only
once. The session sequence is `EMPTY → PRESENT/MOVING → STABLE →
INSPECTION_LOCK → EMPTY_REARMED`. The refractory timer begins when the sharpest
confirmed stable frame is selected, not when AI processing finishes. The next
panel is accepted only after both the refractory time has elapsed and the
configured number of empty frames has been observed.

At a 0.3-second stationary dwell, 0.10–0.15-second effective camera intervals
normally provide two or three candidates. A 0.25-second interval can provide
only one candidate and therefore cannot reliably prove that the panel stopped.
Do not use a network share for high-rate hot-folder input.

## 5. Read the result

| Screen | Meaning | Required operator response |
|---|---|---|
| Green `OK` | Expected code was read above the confidence threshold. | Continue normal operation. |
| Yellow `Abnormal` | ROI/OCR is missing, uncertain, malformed, or unregistered. | Follow the line review procedure; do not assume a pass. |
| Red `Error` | A registered problem code, such as `HJ05`, was read. | Treat as reject-required. Confirm the PLC-side response using the approved line procedure. |
| Dark red `System Error` | Application, model, folder, storage, or internal failure. | Stop automatic passing and escalate. Do not restart production until cleared. |

An early-panel timing warning, empty frame, moving frame, or Frame Gate queue
overrun is not a NORMAL/ABNORMAL/PROBLEM decision. It must not be interpreted as
a pass. Review `unexpected_early_panel`, `frame_gate_overrun`, and
`inference_backpressure` audit events before continuing production.

Every completed image replaces the previous popup. The largest sequence number
always wins, so an older delayed result cannot replace a newer result.

The site-approved OCR policy treats only uppercase `O` and digit `0` as
equivalent. For example, raw OCR values `HJO4` and `HJO5` are logged and then
corrected to `HJ04` and `HJ05`. No other fuzzy character correction is applied.

**Mute speaker** silences sound only. It does not change the decision, audit
record, or PLC event. The next result applies its own alarm policy.

## 6. Record feedback and prepare retraining data

After an operator review, enter an **Operator ID** and **Feedback reason**, then
choose one of the four feedback buttons. The application records feedback but
never changes a model automatically.

Use **Add to retraining** only for an image that has been reviewed, labelled,
and approved for local training. Keep the image, its ROI annotation, code label,
panel/lot/recipe group, and review reason together. Never add an image to a
training set solely because an alarm occurred.

### Label a reviewed panel image

1. In **Image labeling workbench**, select **Open local panel image**.
2. Select `fpcb_surface`, then drag a rectangle around the visible FPCB area.
3. Select `code_roi`, then drag a second rectangle tightly around the printed
   four-character code.
4. Select the reviewed code from **OCR code label** and enter its panel, lot, or
   recipe group. The code list is loaded from the current code recipe.
5. Select **Save reviewed label**. The application stores COCO boxes with
   rectangular segmentation masks, a cropped OCR image, its text label, and the
   group record under `data/processed`.
6. After at least three independent groups have been reviewed, select **Export
   grouped YOLO dataset**. The exporter creates group-separated train, validation,
   and test folders plus `detector/data.yaml`.

Review every rectangle before saving. The application rejects a label unless it
contains exactly one FPCB rectangle and one code rectangle inside the source
image. Do not use development-only synthetic data in a factory dataset.

## 7. End of shift and recovery

1. Select **Stop hot-folder** before planned shutdown.
2. Review `logs/gatekeeper.sqlite3` and `logs/inspection.jsonl` for system
   errors, feedback events, and problem-code events.
3. Back up the approved `config`, `logs`, and `archive` folders according to the
   site's retention policy.
4. After an unexpected PC restart, start the application and repeat the
   start-of-shift check before enabling live input.
