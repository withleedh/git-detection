"""
Create 1000-image dataset for RF-DETR training
- 800 train samples
- 200 val samples
- Balanced across 4 damage classes
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
import random

def create_1k_dataset():
    """Create balanced 1k dataset"""

    # Paths
    full_dataset = Path("datasets/damage_coco")
    output_dataset = Path("datasets/damage_coco_1k")

    # Load full annotations
    with open(full_dataset / "annotations_train.json", 'r') as f:
        train_data = json.load(f)

    with open(full_dataset / "annotations_val.json", 'r') as f:
        val_data = json.load(f)

    # Combine for sampling
    all_images = train_data['images'] + val_data['images']
    all_annotations = train_data['annotations'] + val_data['annotations']

    # Group annotations by image_id
    img_to_anns = defaultdict(list)
    for ann in all_annotations:
        img_to_anns[ann['image_id']].append(ann)

    # Group images by damage class (take first annotation's class)
    class_images = defaultdict(list)
    for img in all_images:
        img_id = img['id']
        if img_id in img_to_anns:
            anns = img_to_anns[img_id]
            # Use first annotation's category
            cat_id = anns[0]['category_id']
            class_images[cat_id].append(img)

    # Sample 250 images per class (total 1000)
    random.seed(42)
    selected_images = []
    selected_image_ids = set()

    categories = train_data['categories']
    cat_names = {cat['id']: cat['name'] for cat in categories}

    print("🔍 Sampling images per class:")
    for cat_id in sorted(class_images.keys()):
        images = class_images[cat_id]
        # Sample 250 images per class
        samples = random.sample(images, min(250, len(images)))
        for img in samples:
            if img['id'] not in selected_image_ids:
                selected_images.append(img)
                selected_image_ids.add(img['id'])
        print(f"  {cat_names[cat_id]}: {len([img for img in selected_images if img['id'] in selected_image_ids and img_to_anns[img['id']][0]['category_id'] == cat_id])} images")

    print(f"\nTotal selected: {len(selected_images)} images")

    # Shuffle and split 80/20
    random.shuffle(selected_images)
    split_idx = int(len(selected_images) * 0.8)

    train_images = selected_images[:split_idx]
    val_images = selected_images[split_idx:]

    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}

    # Filter annotations
    train_annotations = [ann for ann in all_annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in all_annotations if ann['image_id'] in val_image_ids]

    # Create output annotations
    output_train = {
        "info": train_data['info'],
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories
    }

    output_val = {
        "info": val_data['info'],
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories
    }

    # Create output directories
    output_train_dir = output_dataset / "train"
    output_val_dir = output_dataset / "valid"
    output_test_dir = output_dataset / "test"

    output_train_dir.mkdir(parents=True, exist_ok=True)
    output_val_dir.mkdir(parents=True, exist_ok=True)
    output_test_dir.mkdir(parents=True, exist_ok=True)

    # Re-index and copy train images
    img_id_map = {}
    ann_id = 1

    print(f"\n📦 Processing train images...")
    for new_img_id, img in enumerate(train_images, 1):
        old_img_id = img['id']
        img_id_map[old_img_id] = new_img_id

        # Copy image
        src = full_dataset / "train" / img['file_name']
        if not src.exists():
            src = full_dataset / "val" / img['file_name']
        dst = output_train_dir / img['file_name']
        shutil.copy2(src, dst)

        # Add image info
        output_train['images'].append({
            "id": new_img_id,
            "file_name": img['file_name'],
            "width": img['width'],
            "height": img['height']
        })

    # Add train annotations
    for ann in train_annotations:
        if ann['image_id'] in img_id_map:
            output_train['annotations'].append({
                "id": ann_id,
                "image_id": img_id_map[ann['image_id']],
                "category_id": ann['category_id'],
                "segmentation": ann['segmentation'],
                "bbox": ann['bbox'],
                "area": ann['area'],
                "iscrowd": ann['iscrowd']
            })
            ann_id += 1

    # Re-index and copy val images
    img_id_map = {}
    val_img_start_id = len(train_images) + 1

    print(f"📦 Processing validation images...")
    for idx, img in enumerate(val_images):
        new_img_id = val_img_start_id + idx
        old_img_id = img['id']
        img_id_map[old_img_id] = new_img_id

        # Copy image
        src = full_dataset / "train" / img['file_name']
        if not src.exists():
            src = full_dataset / "val" / img['file_name']
        dst = output_val_dir / img['file_name']
        shutil.copy2(src, dst)

        # Add image info
        output_val['images'].append({
            "id": new_img_id,
            "file_name": img['file_name'],
            "width": img['width'],
            "height": img['height']
        })

    # Add val annotations
    for ann in val_annotations:
        if ann['image_id'] in img_id_map:
            output_val['annotations'].append({
                "id": ann_id,
                "image_id": img_id_map[ann['image_id']],
                "category_id": ann['category_id'],
                "segmentation": ann['segmentation'],
                "bbox": ann['bbox'],
                "area": ann['area'],
                "iscrowd": ann['iscrowd']
            })
            ann_id += 1

    # Copy val to test (for RF-DETR requirement)
    print(f"📦 Creating test set (copy of validation)...")
    shutil.copytree(output_val_dir, output_test_dir, dirs_exist_ok=True)

    # Save annotations
    with open(output_train_dir / "_annotations.coco.json", 'w') as f:
        json.dump(output_train, f, indent=2)

    with open(output_val_dir / "_annotations.coco.json", 'w') as f:
        json.dump(output_val, f, indent=2)

    with open(output_test_dir / "_annotations.coco.json", 'w') as f:
        json.dump(output_val, f, indent=2)

    # Print statistics
    print(f"\n{'='*60}")
    print("📊 1K Dataset Created")
    print(f"{'='*60}")
    print(f"Train: {len(output_train['images'])} images, {len(output_train['annotations'])} annotations")
    print(f"Val: {len(output_val['images'])} images, {len(output_val['annotations'])} annotations")

    # Class distribution
    train_class_count = defaultdict(int)
    val_class_count = defaultdict(int)

    for ann in output_train['annotations']:
        train_class_count[ann['category_id']] += 1

    for ann in output_val['annotations']:
        val_class_count[ann['category_id']] += 1

    print(f"\nTrain class distribution:")
    for cat in categories:
        print(f"  {cat['name']}: {train_class_count[cat['id']]} annotations")

    print(f"\nVal class distribution:")
    for cat in categories:
        print(f"  {cat['name']}: {val_class_count[cat['id']]} annotations")

    print(f"\n✅ 1K dataset saved to: {output_dataset.absolute()}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    create_1k_dataset()
