"""
Real-world performance test
Test on random images and manually verify results
"""
from ultralytics import YOLO
import cv2
from pathlib import Path
import random

print("="*60)
print("🎯 REAL-WORLD PERFORMANCE TEST")
print("="*60)

# Load model
model = YOLO('best.pt')
print(f"✓ Model loaded: {model.names}")

# Get random test images
image_dir = Path("/Users/dongho/Downloads/160._차량파손_이미지_데이터/01.데이터/1.Training/1.원천데이터/damage_part")
all_images = list(image_dir.glob("*.jpg"))
random.seed(42)
test_images = random.sample(all_images, min(20, len(all_images)))

print(f"\n📸 Testing on {len(test_images)} random images...\n")

# Statistics
stats = {
    "total_images": len(test_images),
    "images_with_detections": 0,
    "total_detections": 0,
    "high_conf_detections": 0,  # > 50%
    "by_class": {name: 0 for name in model.names.values()}
}

results_summary = []

for i, img_path in enumerate(test_images, 1):
    results = model.predict(
        str(img_path),
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=False
    )

    result = results[0]
    boxes = result.boxes

    if len(boxes) > 0:
        stats["images_with_detections"] += 1

    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]

        stats["total_detections"] += 1
        stats["by_class"][cls_name] += 1

        if conf > 0.5:
            stats["high_conf_detections"] += 1

        detections.append(f"{cls_name}({conf:.0%})")

    detection_str = ", ".join(detections) if detections else "None"
    print(f"[{i:2d}] {img_path.name[:30]:30s} → {detection_str}")

    results_summary.append({
        "filename": img_path.name,
        "detections": len(boxes),
        "details": detections
    })

# Print statistics
print("\n" + "="*60)
print("📊 REAL-WORLD PERFORMANCE STATISTICS")
print("="*60)

detection_rate = stats["images_with_detections"] / stats["total_images"] * 100
print(f"\nDetection Rate:")
print(f"  Images with detections: {stats['images_with_detections']}/{stats['total_images']} ({detection_rate:.1f}%)")

print(f"\nDetection Quality:")
print(f"  Total detections: {stats['total_detections']}")
print(f"  High confidence (>50%): {stats['high_conf_detections']} ({stats['high_conf_detections']/max(stats['total_detections'],1)*100:.1f}%)")
print(f"  Avg per image: {stats['total_detections']/stats['total_images']:.1f}")

print(f"\nDetections by Class:")
for cls_name, count in sorted(stats['by_class'].items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        pct = count / stats['total_detections'] * 100 if stats['total_detections'] > 0 else 0
        print(f"  {cls_name:12s}: {count:3d} ({pct:5.1f}%)")

print("\n" + "="*60)
print("✓ Real-world test complete")
print("="*60)

# Interpretation
print("\n💡 INTERPRETATION:")
if detection_rate > 60:
    print("✅ GOOD: Model detects damages in most images")
elif detection_rate > 30:
    print("⚠️  MODERATE: Model is selective (detects clear damages only)")
else:
    print("❌ LOW: Model may need improvement")

if stats['high_conf_detections'] / max(stats['total_detections'], 1) > 0.5:
    print("✅ GOOD: Most detections are high confidence")
else:
    print("⚠️  Detections are lower confidence")
