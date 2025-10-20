"""
Prepare POC dataset for RF-DETR training
Convert from current structure to RF-DETR expected format
"""

import shutil
from pathlib import Path

def prepare_dataset():
    """
    Current structure:
        datasets/damage_coco_poc/
        ├── train/
        ├── val/
        ├── annotations_train.json
        └── annotations_val.json

    Required structure:
        datasets/damage_coco_poc/
        ├── train/
        │   ├── _annotations.coco.json
        │   └── images...
        └── valid/
            ├── _annotations.coco.json
            └── images...
    """

    src_dir = Path("datasets/damage_coco_poc")

    # Copy train annotations to train folder
    train_ann_src = src_dir / "annotations_train.json"
    train_ann_dst = src_dir / "train" / "_annotations.coco.json"

    if train_ann_src.exists():
        shutil.copy2(train_ann_src, train_ann_dst)
        print(f"✓ Copied {train_ann_src} -> {train_ann_dst}")

    # Rename val to valid and copy annotations
    val_dir = src_dir / "val"
    valid_dir = src_dir / "valid"

    if val_dir.exists() and not valid_dir.exists():
        shutil.copytree(val_dir, valid_dir)
        print(f"✓ Copied {val_dir} -> {valid_dir}")

    val_ann_src = src_dir / "annotations_val.json"
    val_ann_dst = valid_dir / "_annotations.coco.json"

    if val_ann_src.exists():
        shutil.copy2(val_ann_src, val_ann_dst)
        print(f"✓ Copied {val_ann_src} -> {val_ann_dst}")

    print("\n✅ Dataset prepared for RF-DETR training!")
    print(f"Dataset location: {src_dir.absolute()}")

if __name__ == "__main__":
    prepare_dataset()
