# Operator guide

- Green `OK`: expected four-character code was read with sufficient confidence.
- Yellow `Abnormal`: ROI/OCR was missing, uncertain, malformed, or not in the
  configured code lists.
- Red `Error`: a registered problem code such as `HJ05` was read. A strong alarm
  is repeated and a PLC `RejectRequest` is generated.
- Dark red `System Error`: the application, model, storage, or input path failed.

The site-approved OCR policy treats only uppercase `O` and digit `0` as
equivalent. For example, raw OCR values `HJO4` and `HJO5` are logged and then
corrected to `HJ04` and `HJ05`. No other fuzzy character correction is applied.

Every new image replaces the previous result popup. The popup remains visible
until another image is processed. **Mute speaker** silences sound only; it does
not change the decision or PLC event.
