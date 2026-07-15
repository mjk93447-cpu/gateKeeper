# 아키텍처

## 처리 흐름

```text
Camera trigger/frame
       │
       ▼
YOLOX FPCB detector ── 미검출/저신뢰도 ───────────────┐
       │ ROI                                          │
       ▼                                              │
PaddleOCR recognizer ── text + confidence             │
       │                                              │
       ▼                                              ▼
DecisionEngine ─── NORMAL / ABNORMAL / PROBLEM ── Audit log
       │
       ▼
Output policy ── alarm / PLC reject request (interlocked)
```

핵심 판정 엔진은 UI, AI 프레임워크, 카메라와 PLC에서 분리합니다. 동일 입력은 항상 동일 결과가 되어야 하며 모델 업데이트 전후 회귀 시험에 그대로 사용합니다.

## 프로세스 경계

운영 앱과 학습 앱은 분리합니다. 운영 중 학습은 금지하고, 새 모델은 `candidate → offline validation → shadow run → approved → active` 상태를 거칩니다. 모델 파일에는 최소한 다음 메타데이터가 동반되어야 합니다.

- 모델 SHA-256, 학습 코드 커밋, 데이터셋 버전
- class/character dictionary
- validation precision/recall, code-level accuracy, 혼동행렬
- 승인자, 승인 시각, 적용 라인/제품
- ONNX Runtime 및 전처리 버전

## 런타임 구성요소

- `domain`: 3상태 정책, 코드 정규화, 설정 검증
- `application`: 한 패널 검사 orchestration과 감사 이벤트
- `infrastructure`: JSONL/SQLite, 카메라, ONNX, PLC 구현
- `ui`: 운영자 대시보드; 장비 I/O를 직접 제어하지 않음

초기 감사 로그는 JSONL이지만 생산 버전에서는 SQLite WAL + 일별 보관/서명 또는 MES 전송을 권장합니다. 이미지 보관은 정상 샘플링 비율과 비정상/문제성 전량 보관 정책을 별도 설정합니다.

## 카메라/트리거 권장 인터페이스

프레임에는 `panel_id`, PLC trigger sequence, monotonic timestamp, exposure/gain, recipe ID를 붙입니다. 중복 프레임과 오래된 프레임은 판정 전에 제거합니다. 카메라 SDK thread와 inference worker 사이에는 bounded queue를 두어 밀릴 때 무제한 메모리 증가를 막습니다.

