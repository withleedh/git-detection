# Roboflow + Google Colab으로 RF-DETR 학습하기

## 1단계: Roboflow에 데이터 업로드

### 방법 A: Roboflow UI 사용 (추천)

1. **Roboflow 계정 생성**
   - https://app.roboflow.com/ 접속
   - 무료 계정 생성

2. **새 프로젝트 생성**
   - "Create New Project" 클릭
   - Project Type: **Object Detection** 선택
   - Project Name: `vehicle-damage-detection`

3. **데이터 업로드**
   - "Upload Data" 클릭
   - Format: **COCO JSON** 선택
   - 아래 폴더들을 각각 업로드:
     ```
     datasets/damage_coco_1k/train/
     datasets/damage_coco_1k/valid/
     datasets/damage_coco_1k/test/
     ```
   - 각 폴더의 `_annotations.coco.json`과 이미지들 함께 선택

4. **Version 생성**
   - "Generate" → "Create Version" 클릭
   - Preprocessing: 기본값 유지
   - Augmentation: 선택사항 (추천: Flip, Rotation)
   - "Generate" 클릭

### 방법 B: Python API 사용

```bash
# 1. upload_to_roboflow.py 파일 수정
# - ROBOFLOW_API_KEY: Roboflow Settings에서 복사
# - WORKSPACE_NAME: 워크스페이스 이름
# - PROJECT_NAME: 프로젝트 이름

# 2. 실행
python3.10 upload_to_roboflow.py
```

---

## 2단계: Google Colab에서 학습

### Colab 노트북 코드

```python
# =====================================================
# RF-DETR Vehicle Damage Detection Training on Colab
# =====================================================

# 1. 환경 설정
!pip install rfdetr roboflow

# 2. Roboflow에서 데이터셋 다운로드
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("vehicle-damage-detection")
dataset = project.version(1).download("coco")  # COCO 포맷으로 다운로드

# 3. RF-DETR Nano 학습
from rfdetr import RFDETRNano

model = RFDETRNano()

model.train(
    dataset_dir=dataset.location,  # Roboflow가 다운받은 경로
    epochs=100,
    batch_size=16,  # Colab GPU에 맞게 조정 (T4: 8-16, A100: 32)
    grad_accum_steps=1,
    lr=1e-4,
    output_dir="outputs/rfdetr_nano_damage",
    resolution=384,
    use_ema=True,
    early_stopping=True,
    early_stopping_patience=20,
    tensorboard=True,
    checkpoint_interval=10
)

# 4. 학습 완료 후 다운로드
from google.colab import files

# 최종 모델 다운로드
files.download('outputs/rfdetr_nano_damage/checkpoint_best_total.pth')

# 학습 결과 다운로드
files.download('outputs/rfdetr_nano_damage/results.json')
files.download('outputs/rfdetr_nano_damage/metrics_plot.png')
```

---

## 3단계: 학습 모니터링

### TensorBoard (Colab에서)

```python
# Colab에서 실행
%load_ext tensorboard
%tensorboard --logdir outputs/rfdetr_nano_damage/tensorboard
```

### 학습 진행 확인

```python
# 실시간 AP 확인
!tail -f outputs/rfdetr_nano_damage/log.txt | grep "Average Precision"
```

---

## 4단계: 모델 테스트

```python
from rfdetr import RFDETRNano
from PIL import Image
import supervision as sv

# 모델 로드
model = RFDETRNano(
    pretrain_weights='outputs/rfdetr_nano_damage/checkpoint_best_total.pth'
)

# 이미지 로드 및 추론
image = Image.open('test_image.jpg')
detections = model.predict(image, threshold=0.5)

# 시각화
labels = [f"{model.class_names[cls_id]} {conf:.2f}"
          for cls_id, conf in zip(detections.class_id, detections.confidence)]

annotated = image.copy()
annotated = sv.BoxAnnotator().annotate(annotated, detections)
annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)

# 결과 표시
from google.colab.patches import cv2_imshow
import cv2
import numpy as np
cv2_imshow(np.array(annotated))
```

---

## GPU 사양별 권장 설정

| GPU | Batch Size | Grad Accum | Effective Batch | 예상 시간 (100 epochs) |
|-----|------------|------------|-----------------|---------------------|
| T4 | 8 | 2 | 16 | ~4-6시간 |
| V100 | 16 | 1 | 16 | ~2-3시간 |
| A100 | 32 | 1 | 32 | ~1-2시간 |

---

## Colab 노트북 템플릿

**Colab에서 실행하기:**
1. https://colab.research.google.com/ 접속
2. "New Notebook" 생성
3. Runtime → Change runtime type → **GPU** 선택
4. 위 코드 복사 후 실행

**중요:**
- Roboflow API Key 필수
- Colab 무료: T4 GPU (~12시간 제한)
- Colab Pro: V100/A100 GPU (더 빠름)

---

## 문제 해결

### 메모리 부족 에러
```python
# batch_size 줄이기
batch_size=4
grad_accum_steps=4  # effective batch = 16 유지
```

### Colab 연결 끊김
```python
# 학습 재개
model.train(
    dataset_dir=dataset.location,
    resume='outputs/rfdetr_nano_damage/checkpoint.pth',  # 체크포인트에서 재개
    ...
)
```

---

## 다음 단계

1. ✅ Roboflow에 데이터 업로드
2. ✅ Colab에서 학습 시작
3. ⏳ 학습 완료 대기 (2-6시간)
4. 📥 모델 다운로드
5. 🧪 로컬에서 테스트

궁금한 점 있으면 알려주세요!
