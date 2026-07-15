# PLC integration contract

The MVP deliberately does not write physical PLC addresses. It exposes a
protocol-neutral event contract that the PLC developer can map to the line's
chosen protocol.

| Direction | Event | Required fields |
|---|---|---|
| PC -> PLC | `InspectionDone` | `sequence_id`, `result`, timestamp |
| PC -> PLC | `RejectRequest` | `sequence_id`, `panel_id`, reason, timestamp |
| PC -> PLC | `Heartbeat` | timestamp, app version, model version |
| PLC -> PC | `RejectAccepted` | `sequence_id`, timestamp |
| PLC -> PC | `RejectCompleted` | `sequence_id`, timestamp |

Only `PROBLEM` produces `RejectRequest`. `ABNORMAL` produces an alarm and requires
operator/line policy; it is never silently converted to `NORMAL`.

The PC rejects duplicate or out-of-order acknowledgements. Timeout and missing
acknowledgement are logged as system errors. Physical machine stop and vacuum
picker control remain PLC responsibilities.
