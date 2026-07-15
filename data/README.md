# Dataset layout

원본 생산 이미지는 민감한 대용량 데이터이므로 Git에 포함하지 않습니다.

```text
data/
  raw/<capture-date>/<line>/<recipe>/
  processed/detector/{train,val,test}/
  processed/ocr/{images,train.txt,val.txt,test.txt}
```

데이터셋 버전은 이미지 자체가 아닌 manifest(상대 경로, 해시, lot/recipe/조명 slice, split)를 기준으로 부여합니다. 동일 패널 또는 연속 burst 프레임은 하나의 split에만 포함합니다.

