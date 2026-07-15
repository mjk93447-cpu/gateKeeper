# gateKeeper

OLED AAM 투입 직전 패널의 FPCB 모델 코드를 판독하여 **정상 / 비정상 / 문제성**으로 분류하는 Windows 비전 검사 프로그램의 초기 저장소입니다.

## 판정 규칙

| 상태 | 조건 | 기본 동작 |
|---|---|---|
| 정상 (`NORMAL`) | 판독 코드가 현재 정상 코드와 일치하고 신뢰도 기준 통과 | 동작 없음 |
| 문제성 (`PROBLEM`) | 정상 코드는 아니며 등록된 문제성 코드와 일치하고 신뢰도 기준 통과 | 강한 알람 + PLC 격리 요청 |
| 비정상 (`ABNORMAL`) | 위 두 조건 외 전부(미검출, 저신뢰도, 미등록 코드 포함) | 일반 알람 |

> PLC 출력은 안전 PLC를 대체하지 않습니다. 초기 버전은 `simulation` 드라이버만 사용하며, 실제 정지/배출 출력은 현장 I/O 정의, 타임아웃, fail-safe 및 설비 인터록 검증 후 별도 승인해야 합니다.

## 기본 모델 전략

- FPCB ROI detector: Apache-2.0의 **YOLOX-Tiny**, 공개 COCO 사전학습 가중치를 transfer-learning 초기값으로 사용
- OCR: Apache-2.0의 **PaddleOCR PP-OCRv5 mobile English recognition**, `0-9A-Z` 문자 사전과 현장 이미지/합성 이미지로 파인튜닝
- Windows inference: 모델을 ONNX로 내보내고 ONNX Runtime 사용

COCO 모델에는 `FPCB` 클래스가 없으므로 공개 가중치만 받아 운영 추론에 투입하면 안 됩니다. `models/detector.onnx`는 현장 데이터로 학습·검증·승인된 모델이어야 합니다.

## 빠른 시작

Python 3.11을 권장합니다.

```powershell
uv venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev,desktop]"
python -m gatekeeper
pytest
```

GUI는 기본적으로 장비를 움직이지 않는 시뮬레이션 모드입니다. 정상 코드와 문제성 코드를 입력하고 OCR 결과를 모사하면 판정 카드, 건수, 로그가 즉시 갱신됩니다.

## 저장소 구조

```text
src/gatekeeper/
  domain/       판정 규칙과 데이터 모델(프레임워크 독립)
  application/  검사 유스케이스와 감사 로그
  infrastructure/ 카메라/AI/PLC 어댑터 확장 지점
  ui/           PySide6 Windows 운영 화면
config/         장비/모델/코드 설정
scripts/        데이터 검증 및 학습 명령 생성기
docs/           아키텍처, 학습, 안전/PLC 설계
tests/          판정 규칙 회귀 테스트
```

상세 설계는 [아키텍처](docs/ARCHITECTURE.md), [데이터와 학습](docs/TRAINING.md), [PLC 안전 기준](docs/SAFETY.md)을 참고하십시오.

## 현재 범위

이 첫 커밋은 저장소 구조, 결정론적 3상태 판정 엔진, JSONL 감사 로그, 시뮬레이션 UI, 학습 데이터 검증기와 학습 실행 래퍼를 제공합니다. 실제 카메라 SDK, 현장 PLC 주소, 학습 데이터와 승인 모델은 장비별 정보가 확보된 뒤 연결합니다.

