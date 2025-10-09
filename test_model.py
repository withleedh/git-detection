"""
Quick test script for best.pt model
"""
from ultralytics import YOLO
import cv2
import sys

print("="*60)
print("🔍 Testing Vehicle Damage Detection Model")
print("="*60)

# Load model
print("📦 Loading model: best.pt")
model = YOLO('best.pt')
print("✓ Model loaded successfully")

# Get model info
print("\n📊 Model Information:")
print(f"  - Classes: {model.names}")
print(f"  - Number of classes: {len(model.names)}")

# Test with first image from dataset if provided
if len(sys.argv) > 1:
    image_path = sys.argv[1]
    print(f"\n🖼️  Testing with image: {image_path}")

    results = model.predict(
        image_path,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=True
    )

    # Display results
    result = results[0]
    boxes = result.boxes

    print(f"\n✅ Detection Results:")
    print(f"  - Detections found: {len(boxes)}")

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]
        print(f"  - Detection {i+1}: {cls_name} ({conf:.2%} confidence)")

else:
    print("\n💡 Usage: python3.10 test_model.py <image_path>")
    print("   Example: python3.10 test_model.py sample_image.jpg")

print("\n" + "="*60)
print("✓ Test completed!")
print("="*60)
