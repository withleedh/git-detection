# Vehicle Damage Detection API - 사용 가이드

## 🚀 서버 실행

```bash
python3.10 app.py
```

서버는 **http://localhost:8080** 에서 실행됩니다.

---

## 📡 API 엔드포인트

### 1. Health Check (서버 상태 확인)

```bash
curl http://localhost:8080/health
```

**응답:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_path": "best.pt",
    "version": "1.0"
}
```

---

### 2. Single Image Detection (단일 이미지 검출)

```bash
curl -X POST \
  -F "image=@/path/to/your/image.jpg" \
  http://localhost:8080/predict
```

**옵션: Confidence Threshold 조정**
```bash
curl -X POST \
  -F "image=@/path/to/your/image.jpg" \
  -F "confidence=0.5" \
  http://localhost:8080/predict
```

**응답 예시:**
```json
{
    "success": true,
    "detection_count": 1,
    "detections": [
        {
            "detection_id": 0,
            "class": "scratch",
            "class_id": 0,
            "confidence": 0.9022314548492432,
            "bbox": {
                "x1": 459.37,
                "y1": 347.36,
                "x2": 489.01,
                "y2": 459.37,
                "width": 29.64,
                "height": 112.01
            }
        }
    ],
    "image_info": {
        "filename": "car.jpg",
        "width": 800,
        "height": 600
    },
    "processing_time_seconds": 0.785
}
```

---

### 3. Batch Detection (다중 이미지 검출)

```bash
curl -X POST \
  -F "images[]=@image1.jpg" \
  -F "images[]=@image2.jpg" \
  -F "images[]=@image3.jpg" \
  http://localhost:8080/predict_batch
```

**응답 예시:**
```json
{
    "success": true,
    "batch_size": 3,
    "results": [
        {
            "filename": "image1.jpg",
            "success": true,
            "detection_count": 2,
            "detections": [...]
        },
        {
            "filename": "image2.jpg",
            "success": true,
            "detection_count": 0,
            "detections": []
        }
    ],
    "total_processing_time_seconds": 2.145,
    "avg_time_per_image": 0.715
}
```

---

### 4. API Documentation (API 문서)

```bash
curl http://localhost:8080/
```

API 전체 문서를 JSON 형식으로 반환합니다.

---

## 🐍 Python에서 사용하기

```python
import requests

# 단일 이미지 검출
url = "http://localhost:8080/predict"
files = {"image": open("car_damage.jpg", "rb")}
data = {"confidence": 0.3}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"검출 개수: {result['detection_count']}")
for detection in result['detections']:
    print(f"- {detection['class']}: {detection['confidence']:.2%}")
```

---

## 📊 응답 필드 설명

### Detection Object
- `detection_id`: 검출 번호 (0부터 시작)
- `class`: 클래스 이름 ("scratch")
- `class_id`: 클래스 ID (0)
- `confidence`: 신뢰도 (0.0 ~ 1.0)
- `bbox`: Bounding Box 좌표
  - `x1, y1`: 좌측 상단 좌표
  - `x2, y2`: 우측 하단 좌표
  - `width, height`: 박스 크기

---

## ⚙️ 설정

### Confidence Threshold (기본값: 0.25)
- 값이 높을수록 확실한 검출만 반환
- 값이 낮을수록 더 많은 검출 반환 (False Positive ↑)

**추천 값:**
- `0.25` - 균형잡힌 검출 (기본값)
- `0.4` - 보수적 검출 (정확도 우선)
- `0.15` - 적극적 검출 (재현율 우선)

---

## 🔧 테스트 스크립트

간단한 CLI 테스트:
```bash
python3.10 test_model.py /path/to/image.jpg
```

---

## 📝 Notes

- **첫 번째 요청**: 모델 초기화로 인해 느릴 수 있습니다 (~6초)
- **이후 요청**: 일반적으로 0.5-1초 내에 완료
- **지원 이미지 형식**: JPG, PNG, BMP 등
- **최대 이미지 크기**: 제한 없음 (자동으로 640px로 리사이즈)

---

## 🎯 현재 모델 정보

- **모델 파일**: `best.pt` (49MB)
- **클래스**: `scratch` (단일 클래스)
- **입력 크기**: 640x640
- **프레임워크**: YOLOv11m
- **학습 데이터**: 약 40,000장의 차량 손상 이미지
