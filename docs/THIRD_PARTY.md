# 제3자 구성요소 검토 기록

| 구성요소 | 용도 | 라이선스 | 주의사항 |
|---|---|---|---|
| YOLOX | detector 학습/변환 | Apache-2.0 | 공식 가중치 재배포 조건은 출시 전 별도 법무 확인 |
| PaddleOCR | OCR 학습/추론 | Apache-2.0 | 모델별 배포 조건과 notice 확인 |
| ONNX Runtime | Windows 추론 | MIT | native provider DLL 포함 여부 확인 |
| PySide6 / Qt | Windows UI | LGPLv3/GPLv3 또는 상용 | 동적 링크/재링크 및 Qt 배포 의무 검토 |

이 표는 법률 자문이 아닙니다. 제품 배포 전에 실제 고정 버전의 LICENSE/NOTICE와 가중치 사용 조건을 다시 검토합니다. 프로젝트 자체의 루트 라이선스는 소유자 정책이 정해질 때까지 의도적으로 추가하지 않았습니다.

