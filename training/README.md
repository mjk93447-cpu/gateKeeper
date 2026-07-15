# Training workspace

YOLOX와 PaddleOCR upstream 소스는 이 저장소에 복사하지 않고 별도 고정 버전 checkout을 사용합니다. `scripts/training_commands.py`는 실행할 명령을 먼저 출력하며 `--execute`를 명시한 경우에만 학습을 시작합니다.

프로젝트별 전체 설정 파일은 데이터 수집 후 upstream 기본 설정을 복사해 이 폴더에서 관리합니다.

- YOLOX: `yolox_fpcb.py`에서 `num_classes=1`, class `fpcb`, 416×416, 현장 COCO split 지정
- PaddleOCR: PP-OCRv5 mobile English recognition 설정을 기반으로 문자 사전 `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`와 현장 label 파일 지정

학습 설정에는 절대 경로 대신 환경별 데이터 루트를 사용하고, 실행 시 사용한 최종 설정 사본을 `runs/<run-id>/`에 보존하십시오.

