# PLC·설비 안전 기준

AI 판정 프로그램은 안전 PLC나 하드웨어 인터록을 대체하지 않습니다. `PROBLEM` 판정이 곧바로 모터/진공 밸브를 구동하지 않게 하고, PLC에 **reject request + panel sequence**를 전달한 뒤 PLC가 위치 센서, picker 준비, 진공압, 문/비상정지 상태를 확인해 동작하도록 설계합니다.

## 필수 신호(초안)

| 방향 | 신호 | 의미 |
|---|---|---|
| PLC → PC | `InspectionTrigger(seq)` | 해당 패널 촬영/판정 시작 |
| PC → PLC | `InspectionDone(seq, result)` | seq에 대한 판정 완료 |
| PC → PLC | `RejectRequest(seq)` | 문제성 패널 격리 요청 |
| PLC → PC | `RejectAccepted(seq)` | 설비가 요청을 수락 |
| PLC → PC | `RejectCompleted(seq)` | 격리 완료 확인 |
| 양방향 | `Heartbeat` | 통신 생존 감시 |

## fail-safe 초안

- PC timeout, 카메라 끊김, 모델 load 실패, ROI 미검출은 정상 통과로 처리하지 않습니다.
- sequence 불일치/중복 응답은 출력하지 않고 알람 및 수동 확인 상태로 전환합니다.
- 재시작 후 이전 패널에 대한 오래된 reject 요청을 재전송하지 않습니다.
- 실제 출력 드라이버는 simulation과 명확히 구분하고 키 스위치/권한/recipe 승인 없이 활성화하지 않습니다.
- 모든 출력 요청과 PLC acknowledgement를 monotonic/UTC 시각과 함께 감사 로그에 남깁니다.

현장 적용 전 설비 업체와 I/O map, pulse/level 방식, 타임아웃, retry, 통신 프로토콜(OPC UA/MC/Modbus/TCP 등), 안전 책임 경계를 확정해야 합니다.

