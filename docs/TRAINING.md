# 데이터와 로컬 학습

## 1. FPCB ROI detector

기본 선택은 YOLOX-Tiny(416×416)입니다. COCO 사전학습 가중치는 feature 초기값으로만 쓰며, 현장 이미지에서 FPCB ROI 한 클래스를 학습합니다.

데이터는 장비/조명/제품/lot 단위로 train/validation/test를 분리해 동일 패널의 연속 프레임이 서로 다른 split에 섞이지 않게 합니다. 다음 변동을 반드시 포함합니다.

- 정상 위치와 허용 가능한 위치/회전 편차
- 반사, 흐림, 노출 변화, 오염과 부분 가림
- HJ04/HJ05 등 다양한 인쇄와 빈 FPCB
- 실제로 발생하는 다른 모델, 치구, 작업자 개입 장면

COCO 형식의 class 이름은 `fpcb`로 통일합니다.

```powershell
python scripts/validate_detection_dataset.py data/processed/detector/train/annotations.json
python scripts/training_commands.py detector `
  --yolox C:\ml\YOLOX `
  --experiment training/yolox_fpcb.py `
  --weights C:\ml\weights\yolox_tiny.pth
```

`--execute`를 붙이기 전 출력 명령과 경로를 확인합니다. upstream 저장소/가중치 버전은 모델 manifest에 고정해야 합니다.

## 2. OCR recognition

YOLO ROI 안에서 코드 한 줄만 인식하므로 범용 문서 OCR 전체가 아니라 recognition 모델을 파인튜닝합니다. 문자 사전은 우선 `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`로 제한합니다. 라벨 파일은 PaddleOCR 형식인 `<relative-image-path> TAB <label>`입니다.

```text
images/000001.png\tHJ04
images/000002.png\tHJ05
```

```powershell
python scripts/validate_ocr_dataset.py data/processed/ocr/train.txt
python scripts/training_commands.py ocr `
  --paddleocr C:\ml\PaddleOCR `
  --config training/ppocrv5_gatekeeper.yml `
  --pretrained C:\ml\weights\PP-OCRv5_mobile_rec_pretrained.pdparams
```

정상/문제성 코드 이미지를 균형 있게 수집하고, 실제 FPCB 배경·폰트·레이저/잉크 번짐을 모사한 합성 데이터를 보강합니다. HJ04와 HJ05처럼 한 글자만 다른 hard negative를 별도 평가 세트로 유지합니다. 운영 판정에는 fuzzy match를 사용하지 않습니다.

## 3. 승인 지표

전체 평균만 보지 말고 코드별로 다음을 기록합니다.

- detector miss rate 및 ROI IoU/recall
- exact-code accuracy(문자 정확도가 아님)
- `HJ04 → HJ05`, `HJ05 → HJ04` 혼동 건수
- 정상 false reject rate, 문제성 false accept rate
- p50/p95/p99 end-to-end latency
- 조명/lot/제품/카메라별 slice metric

문제성 false accept는 안전 관련 핵심 지표입니다. 목표값은 라인 takt time과 품질팀 위험평가를 반영해 승인 문서에서 수치로 확정합니다.

