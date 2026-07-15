# Model registry

운영 모델 바이너리는 Git에 커밋하지 않습니다. 승인된 artifact store에서 내려받아 이 디렉터리에 배치하고 `manifest.json`의 SHA-256과 일치하는지 확인한 뒤 로드합니다.

```text
models/
  detector.onnx       현장 FPCB 1-class detector
  ocr/                현장 코드 recognition model
  manifest.json       모델 계보, 지표, 승인 정보
```

공개 COCO YOLOX 가중치는 학습 초기값이며 `detector.onnx`를 대신할 수 없습니다. `manifest.example.json`을 복사해 실제 승인 정보를 채우십시오.

