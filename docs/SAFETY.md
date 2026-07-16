# Safety and fail-safe behavior

Manufacturing Junction gateKeeper AI Vision never substitutes for the machine safety PLC. Its PC-side behavior is
limited to classification, operator alarms, audit events and a protocol-neutral
`RejectRequest`.

- `NORMAL` is emitted only after a valid four-character code matches the expected
  code and all confidence rules pass.
- Missing images, detector failures, OCR failures, storage failures, model hash
  mismatches and PLC timeouts are never converted to `NORMAL`.
- `PROBLEM` produces a red Error popup, repeating speaker alarm and one
  idempotent RejectRequest per sequence.
- Physical machine stop and picker isolation remain PLC responsibilities.
- Duplicate and out-of-order acknowledgements are rejected and logged.
- The operator's speaker mute control does not change the classification or PLC
  event.
