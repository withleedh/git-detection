"""
Vehicle Scratch Detection - Model Training Script
High-performance YOLO training for production-grade scratch detection
"""

from ultralytics import YOLO
import torch
from pathlib import Path


def check_environment():
    """Check training environment"""
    print("="*60)
    print("🔧 ENVIRONMENT CHECK")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("⚠️  Training will run on CPU (slower)")
    print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
    print(f"MPS (Apple Silicon) torch is built: {torch.backends.mps.is_built()}")

    print("="*60 + "\n")


def train_scratch_detector(
    model_size: str = "yolo11s.pt",
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 64,
    project_name: str = "experiments",
    run_name: str = "scratch_detection_v1",
    use_tune: bool = False,
    iterations: int = 30
):
    """
    Train YOLOv11 model for scratch detection

    Args:
        model_size: Pre-trained model (yolo11n/s/m/l/x.pt)
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Batch size (adjust based on GPU memory)
        project_name: Project directory name
        run_name: Experiment run name
    """

    print("🚀 VEHICLE SCRATCH DETECTION - TRAINING")
    print("="*60)
    print(f"Model: {model_size}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Output: {project_name}/{run_name}")
    print("="*60 + "\n")

    # Check environment
    check_environment()

    # Load pre-trained model
    print(f"📥 Loading pre-trained model: {model_size}")
    model = YOLO(model_size)
    print("✓ Model loaded successfully\n")

    # --- 훈련 또는 튜닝 시작 (이 부분이 변경됩니다) ---
    if use_tune:
        print(f"🧬 Starting hyperparameter tuning for {iterations} iterations...\n")
        # model.tune() 호출
        results = model.tune(
            # --- 필수 파라미터 ---
            data="damage_multiclass.yaml",
            epochs=epochs,
            iterations=iterations,
            optimizer='AdamW',
            
            # --- 출력 및 검증 설정 ---
            project=project_name,
            name=run_name,
            plots=False,  # 튜닝 중에는 속도를 위해 보통 비활성화
            save=False,   # 공간 절약을 위해 보통 비활성화
            val=True
        )
    else:
        print("🏋️  Starting training...\n")
        # 기존 model.train() 호출
        results = model.train(
            data="damage_multiclass.yaml",
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            optimizer='AdamW',
            lr0=0.0005,
            lrf=0.01,                   # Final learning rate (lr0 * lrf)
            momentum=0.937,             # SGD momentum/Adam beta1
            weight_decay=0.0005,        # Optimizer weight decay
            warmup_epochs=3.0,          # Warmup epochs
            warmup_momentum=0.8,        # Warmup initial momentum
    
            # Data augmentation
            hsv_h=0.015,               # HSV-Hue augmentation
            hsv_s=0.7,                 # HSV-Saturation augmentation
            hsv_v=0.4,                 # HSV-Value augmentation
            degrees=0.0,               # Rotation (+/- deg)
            translate=0.1,             # Translation (+/- fraction)
            scale=0.5,                 # Image scale (+/- gain)
            shear=0.0,                 # Shear (+/- deg)
            perspective=0.0,           # Perspective (+/- fraction)
            flipud=0.0,                # Flip up-down probability
            fliplr=0.5,                # Flip left-right probability
            mosaic=1.0,                # Mosaic augmentation probability
            mixup=0.0,                 # MixUp augmentation probability
            copy_paste=0.0,            # Copy-paste augmentation probability
    
            # Validation
            val=True,                  # Validate during training
    
            # Save options
            save=True,                 # Save checkpoints
            save_period=10,            # Save checkpoint every n epochs
    
            # Output
            project=project_name,
            name=run_name,
            exist_ok=False,            # Overwrite existing project/name
    
            # # Performance
            # device='mps',
            # workers=8,                 # Number of worker threads
    
            # Logging
            verbose=True,              # Verbose output
            plots=True,                # Save plots
    
            # Advanced
            amp=True,                  # Automatic Mixed Precision training
            fraction=1.0,              # Train on fraction of data (1.0 = all data)
            patience=50,               # Early stopping patience (epochs)
    
            # Class weights (balanced for single class)
            cls=0.5,                   # Classification loss weight
            box=7.5,                   # Box loss weight
            dfl=1.5,                   # DFL loss weight
    )

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED")
    print("="*60)

    # Print results
    print(f"\n📊 Final Results:")
    print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")

    # Save path
    save_dir = Path(project_name) / run_name
    weights_dir = save_dir / "weights"

    print(f"\n💾 Model saved to:")
    print(f"   Best weights: {weights_dir / 'best.pt'}")
    print(f"   Last weights: {weights_dir / 'last.pt'}")
    print(f"\n📁 All results saved to: {save_dir.absolute()}")
    print("="*60 + "\n")

    return results


def validate_model(weights_path: str = "experiments/scratch_detection_v1/weights/best.pt"):
    """
    Validate trained model

    Args:
        weights_path: Path to trained weights
    """
    print("🔍 VALIDATING MODEL")
    print("="*60)

    model = YOLO(weights_path)

    # Run validation
    metrics = model.val(
        data="damage_multiclass.yaml",  # Multi-class dataset
        split='val',
        imgsz=640,
        batch=24,
        verbose=True,
        plots=True
    )

    print("\n📊 Validation Metrics:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("="*60 + "\n")

    return metrics


if __name__ == "__main__":

    TUNE_MODE = True

    if TUNE_MODE:
        # 하이퍼파라미터 튜닝 설정
        CONFIG = {
            "model_size": "yolo11s.pt",
            "epochs": 30,
            "img_size": 640,
            "batch_size": 64,
            "project_name": "experiments",
            "run_name": "damage_multiclass_TUNE_v1",
            # --- 튜닝 관련 설정 ---
            "use_tune": True,
            "iterations": 30  # 시도할 조합의 수 (시간이 오래 걸리면 줄여서 테스트)
        }
    else:
        # 기존 일반 훈련 설정
        CONFIG = {
            "model_size": "yolo11l.pt",
            "epochs": 30,
            "img_size": 640,
            "batch_size": 64,
            "project_name": "experiments",
            "run_name": "damage_multiclass_v1",
            # --- 일반 훈련 관련 설정 ---
            "use_tune": False,
            "iterations": 0  # 사용 안 함
        }
    # For quick testing (comment out for production run):
    # CONFIG["epochs"] = 10
    # CONFIG["run_name"] = "scratch_detection_test"

    # Train or Tune model
    results = train_scratch_detector(**CONFIG)

    # Validate best model (튜닝 후에는 자동으로 최적의 모델을 검증)
    print("\n🎯 Running final validation on best model...")
    # 튜닝은 하위 폴더에 결과를 저장하므로 경로 수정이 필요할 수 있습니다.
    # 우선은 그대로 두거나, 튜닝 후 출력되는 경로를 확인하여 수정하세요.
    validate_model(f"{CONFIG['project_name']}/{CONFIG['run_name']}/weights/best.pt")


    # Validate best model
    print("\n🎯 Running final validation on best model...")
    validate_model(f"{CONFIG['project_name']}/{CONFIG['run_name']}/weights/best.pt")

    print("\n✨ Training pipeline completed successfully!")
    print("📌 Next steps:")
    print("   1. Check experiments/ folder for training results")
    print("   2. Review plots (confusion matrix, F1 curve, PR curve)")
    print("   3. If mAP50 < 0.80, consider:")
    print("      - Increase epochs (200-300)")
    print("      - Use larger model (yolo11m.pt or yolo11l.pt)")
    print("      - Adjust data augmentation parameters")
    print("   4. Deploy model using app.py API server\n")
