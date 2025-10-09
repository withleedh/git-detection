"""
Vehicle Scratch Detection - Production API Server
RESTful API for real-time scratch detection using trained YOLO model
"""

from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os
from typing import Dict, List
import time
from PIL import ImageFont, ImageDraw, Image


app = Flask(__name__)

# Global model instance (loaded once at startup)
model = None
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.2


def load_model():
    """Load YOLO model at startup"""
    global model

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please train the model first using train.py"
        )

    print("="*60)
    print("🚀 LOADING SCRATCH DETECTION MODEL")
    print("="*60)
    print(f"Model path: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    print("✓ Model loaded successfully")
    print(f"✓ Model type: {type(model)}")
    print("="*60 + "\n")


def process_detection_results(results, use_korean=True) -> Dict:
    """
    Process YOLO detection results into JSON format

    Args:
        results: YOLO detection results
        use_korean: Return Korean class names (default: True)

    Returns:
        Dictionary with detection data
    """
    detections = []

    # Korean translation map (올바른 매핑)
    korean_names = {
        "Breakage": "파손",        # 0: 파손
        "Crushed": "찌그러짐",      # 1: 찌그러짐
        "Separated": "이격",        # 2: 이격
        "Scratched": "스크래치",    # 3: 스크래치
        "scratch": "스크래치"
    }

    # Extract results from first image
    result = results[0]

    # Get boxes, confidences, and class IDs
    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    # Process each detection
    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
        x1, y1, x2, y2 = box

        english_name = model.names[int(cls_id)]
        class_name = korean_names.get(english_name, english_name) if use_korean else english_name

        detection = {
            "detection_id": i,
            "class": class_name,
            "class_id": int(cls_id),
            "confidence": float(conf),
            "bbox": {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "width": float(x2 - x1),
                "height": float(y2 - y1)
            }
        }

        detections.append(detection)

    return detections


def draw_detections(image, results, transparency=0.3):
    """
    Draw detection results on image with colored boxes and semi-transparent overlay

    Args:
        image: Original image (numpy array)
        results: YOLO detection results
        transparency: Transparency level for overlay (0.0 to 1.0)

    Returns:
        Annotated image with colored detection boxes
    """
    # Clone image to avoid modifying original
    annotated = image.copy()
    overlay = image.copy()

    # Extract results
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    # Korean translation map (올바른 매핑)
    korean_names = {
        "Breakage": "파손",        # 0: 파손
        "Crushed": "찌그러짐",      # 1: 찌그러짐
        "Separated": "이격",        # 2: 이격
        "Scratched": "스크래치",    # 3: 스크래치
        "scratch": "스크래치"       # For single-class model
    }

    # Define colors (BGR format for OpenCV) - 진한 색상
    colors = {
        0: (0, 0, 255),      # Red (빨강) for 파손
        1: (0, 200, 255),    # Orange (주황) for 찌그러짐 - 더 진하게
        2: (255, 0, 255),    # Magenta (마젠타) for 이격
        3: (255, 100, 0)     # Blue (파랑) for 스크래치 - 더 진하게
    }

    # Draw each detection
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls_id)
        color = colors.get(cls_id, (0, 255, 0))  # Default to green

        # Draw filled rectangle on overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Draw border rectangle on annotated
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Add label with confidence (Korean)
        english_name = model.names[cls_id]
        korean_name = korean_names.get(english_name, english_name)
        label = f"{korean_name}: {conf:.2%}"

        # Convert BGR to RGB for PIL
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(annotated_pil)

        # Try to load Korean font
        font_loaded = False
        font = None
        for font_path in [
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
        ]:
            try:
                font = ImageFont.truetype(font_path, 28)
                font_loaded = True
                break
            except:
                continue

        if not font_loaded:
            font = ImageFont.load_default()

        # Get text size
        bbox = draw.textbbox((0, 0), label, font=font)
        label_w = bbox[2] - bbox[0]
        label_h = bbox[3] - bbox[1]

        # Draw label background (더 진한 색상)
        bg_y1 = max(0, y1 - label_h - 10)
        bg_color_bgr = tuple(max(0, int(c * 0.7)) for c in color)  # 70% 어둡게
        draw.rectangle(
            [(x1, bg_y1), (x1 + label_w + 10, y1)],
            fill=bg_color_bgr[::-1]  # Convert BGR to RGB
        )

        # Draw label text with outline for better visibility
        text_pos = (x1 + 5, bg_y1 + 2)
        # Draw outline (검은색 테두리)
        for adj in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            draw.text(
                (text_pos[0]+adj[0], text_pos[1]+adj[1]),
                label,
                font=font,
                fill=(0, 0, 0)
            )
        # Draw main text (흰색)
        draw.text(
            text_pos,
            label,
            font=font,
            fill=(255, 255, 255)
        )

        # Convert back to BGR for OpenCV
        annotated = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)

    # Blend overlay with annotated image
    result_image = cv2.addWeighted(overlay, transparency, annotated, 1 - transparency, 0)

    return result_image


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "version": "1.0"
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Scratch detection endpoint

    Expected input:
        - multipart/form-data with 'image' file

    Returns:
        JSON with detection results
    """
    start_time = time.time()

    # Validate request
    if 'image' not in request.files:
        return jsonify({
            "error": "No image file provided",
            "message": "Please upload an image using the 'image' field"
        }), 400

    file = request.files['image']

    # Validate file
    if file.filename == '':
        return jsonify({
            "error": "Empty filename",
            "message": "Please provide a valid image file"
        }), 400

    try:
        # Read image file
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                "error": "Invalid image",
                "message": "Could not decode image. Please upload a valid image file."
            }), 400

        # Get image dimensions
        img_height, img_width = image.shape[:2]

        # Run inference
        conf_threshold = request.form.get('confidence', CONFIDENCE_THRESHOLD)
        conf_threshold = float(conf_threshold)

        results = model.predict(
            image,
            conf=conf_threshold,
            iou=0.45,
            imgsz=640,
            verbose=False,
            device='mps'
        )

        # Process results
        detections = process_detection_results(results)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Prepare response
        response = {
            "success": True,
            "image_info": {
                "filename": file.filename,
                "width": img_width,
                "height": img_height
            },
            "detection_count": len(detections),
            "detections": detections,
            "model_info": {
                "model_path": MODEL_PATH,
                "confidence_threshold": conf_threshold
            },
            "processing_time_seconds": round(processing_time, 3)
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "error": "Processing failed",
            "message": str(e)
        }), 500


@app.route('/visualize', methods=['POST'])
def visualize():
    """
    Scratch detection with visualization endpoint
    Returns image with colored detection boxes

    Expected input:
        - multipart/form-data with 'image' file

    Returns:
        Annotated image file (JPG)
    """
    start_time = time.time()

    # Validate request
    if 'image' not in request.files:
        return jsonify({
            "error": "No image file provided",
            "message": "Please upload an image using the 'image' field"
        }), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({
            "error": "Empty filename",
            "message": "Please provide a valid image file"
        }), 400

    try:
        # Read image file
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                "error": "Invalid image",
                "message": "Could not decode image. Please upload a valid image file."
            }), 400

        # Run inference
        conf_threshold = request.form.get('confidence', CONFIDENCE_THRESHOLD)
        conf_threshold = float(conf_threshold)
        transparency = float(request.form.get('transparency', 0.3))

        results = model.predict(
            image,
            conf=conf_threshold,
            iou=0.45,
            imgsz=640,
            verbose=False,
            device='mps'
        )

        # Draw detections on image
        annotated_image = draw_detections(image, results, transparency)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_file.name, annotated_image)
        temp_file.close()

        # Return image file
        return send_file(
            temp_file.name,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f"annotated_{file.filename}"
        )

    except Exception as e:
        return jsonify({
            "error": "Processing failed",
            "message": str(e)
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch scratch detection endpoint

    Expected input:
        - multipart/form-data with multiple 'images[]' files

    Returns:
        JSON with batch detection results
    """
    start_time = time.time()

    # Get uploaded files
    files = request.files.getlist('images[]')

    if not files or len(files) == 0:
        return jsonify({
            "error": "No images provided",
            "message": "Please upload images using the 'images[]' field"
        }), 400

    try:
        batch_results = []
        conf_threshold = float(request.form.get('confidence', CONFIDENCE_THRESHOLD))

        for idx, file in enumerate(files):
            if file.filename == '':
                continue

            # Read image
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                batch_results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid image format"
                })
                continue

            # Get dimensions
            img_height, img_width = image.shape[:2]

            # Run inference
            results = model.predict(
                image,
                conf=conf_threshold,
                iou=0.45,
                imgsz=640,
                verbose=False,
                device='mps'
            )

            # Process results
            detections = process_detection_results(results)

            batch_results.append({
                "filename": file.filename,
                "success": True,
                "image_info": {
                    "width": img_width,
                    "height": img_height
                },
                "detection_count": len(detections),
                "detections": detections
            })

        # Calculate total processing time
        total_time = time.time() - start_time

        response = {
            "success": True,
            "batch_size": len(batch_results),
            "results": batch_results,
            "model_info": {
                "model_path": MODEL_PATH,
                "confidence_threshold": conf_threshold
            },
            "total_processing_time_seconds": round(total_time, 3),
            "avg_time_per_image": round(total_time / len(batch_results), 3) if batch_results else 0
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "error": "Batch processing failed",
            "message": str(e)
        }), 500


@app.route('/', methods=['GET'])
def index():
    """API documentation endpoint"""
    return jsonify({
        "name": "Vehicle Scratch Detection API",
        "version": "1.0",
        "description": "Production-grade AI API for detecting breakage on vehicle images",
        "endpoints": {
            "GET /": "API documentation",
            "GET /health": "Health check",
            "POST /predict": "Single image scratch detection (JSON)",
            "POST /visualize": "Single image detection with visualization (Image)",
            "POST /predict_batch": "Batch image scratch detection"
        },
        "usage": {
            "/predict": {
                "method": "POST",
                "content_type": "multipart/form-data",
                "parameters": {
                    "image": "Image file (required)",
                    "confidence": "Confidence threshold 0-1 (optional, default: 0.25)"
                },
                "example": "curl -X POST -F 'image=@car.jpg' -F 'confidence=0.3' http://localhost:5000/predict"
            },
            "/predict_batch": {
                "method": "POST",
                "content_type": "multipart/form-data",
                "parameters": {
                    "images[]": "Multiple image files (required)",
                    "confidence": "Confidence threshold 0-1 (optional, default: 0.25)"
                },
                "example": "curl -X POST -F 'images[]=@car1.jpg' -F 'images[]=@car2.jpg' http://localhost:5000/predict_batch"
            }
        },
        "model": {
            "path": MODEL_PATH,
            "target_class": "scratch",
            "input_size": 640
        }
    })


if __name__ == '__main__':
    # Load model at startup
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("\n📌 Please train the model first:")
        print("   python train.py\n")
        exit(1)

    # Start Flask server
    print("🌐 Starting API server...")
    print("="*60)
    print("API Endpoints:")
    print("  - http://localhost:8080/         (Documentation)")
    print("  - http://localhost:8080/health   (Health check)")
    print("  - http://localhost:8080/predict  (Single detection - JSON)")
    print("  - http://localhost:8080/visualize (Single detection - Image)")
    print("  - http://localhost:8080/predict_batch (Batch detection)")
    print("="*60)
    print("\n🚀 Server running at http://localhost:8080")
    print("Press CTRL+C to stop\n")

    app.run(
        host='0.0.0.0',
        port=8080,
        debug=False,
        threaded=True
    )
