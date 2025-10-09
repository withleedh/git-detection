"""
Verify preprocessing quality by checking label files
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

# Paths
raw_labels = Path("/Users/dongho/Downloads/160._차량파손_이미지_데이터/01.데이터/1.Training/2.라벨링데이터/damage_part")
raw_images = Path("/Users/dongho/Downloads/160._차량파손_이미지_데이터/01.데이터/1.Training/1.원천데이터/damage_part")

print("="*60)
print("🔍 PREPROCESSING VERIFICATION")
print("="*60)

# Sample 5 files and verify conversion
sample_files = list(raw_labels.glob("*.json"))[:5]

class_map = {
    "Breakage": 0,
    "Crushed": 1,
    "Separated": 2,
    "Scratched": 3
}

for i, json_file in enumerate(sample_files, 1):
    print(f"\n📄 Sample {i}: {json_file.name}")

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get image info
    image_info = data.get("images", {})
    img_width = image_info.get("width", 0)
    img_height = image_info.get("height", 0)
    img_filename = image_info.get("file_name", "")

    print(f"   Image: {img_filename} ({img_width}x{img_height})")

    # Check if image exists
    img_path = raw_images / img_filename
    img_exists = img_path.exists()
    print(f"   Image exists: {img_exists}")

    # Process annotations
    annotations = data.get("annotations", [])
    print(f"   Annotations: {len(annotations)}")

    for j, ann in enumerate(annotations[:3]):  # Show first 3
        damage_type = ann.get("damage")
        bbox = ann.get("bbox", [])

        if damage_type in class_map and len(bbox) == 4:
            class_id = class_map[damage_type]
            x_min, y_min, w, h = bbox

            # Convert to YOLO format
            x_center = (x_min + w / 2) / img_width
            y_center = (y_min + h / 2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height

            print(f"      [{j+1}] {damage_type} (class {class_id})")
            print(f"          Raw bbox: [{x_min:.0f}, {y_min:.0f}, {w:.0f}, {h:.0f}]")
            print(f"          YOLO: {class_id} {x_center:.4f} {y_center:.4f} {norm_w:.4f} {norm_h:.4f}")

            # Validation checks
            issues = []
            if not (0 <= x_center <= 1):
                issues.append(f"x_center out of range: {x_center}")
            if not (0 <= y_center <= 1):
                issues.append(f"y_center out of range: {y_center}")
            if not (0 < norm_w <= 1):
                issues.append(f"width out of range: {norm_w}")
            if not (0 < norm_h <= 1):
                issues.append(f"height out of range: {norm_h}")

            if issues:
                print(f"          ⚠️  ISSUES: {', '.join(issues)}")

# Global statistics
print("\n" + "="*60)
print("📊 DATASET-WIDE ANALYSIS")
print("="*60)

all_json_files = list(raw_labels.glob("*.json"))
print(f"Total JSON files: {len(all_json_files)}")

bbox_issues = Counter()
damage_distribution = Counter()
bbox_sizes = defaultdict(list)
missing_images = 0
empty_annotations = 0
total_annotations = 0

print("\nScanning first 1000 files...")
for json_file in all_json_files[:1000]:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_info = data.get("images", {})
        img_width = image_info.get("width", 0)
        img_height = image_info.get("height", 0)
        img_filename = image_info.get("file_name", "")

        # Check image exists
        if img_filename:
            img_path = raw_images / img_filename
            if not img_path.exists():
                missing_images += 1

        annotations = data.get("annotations", [])

        if not annotations:
            empty_annotations += 1
            continue

        for ann in annotations:
            damage_type = ann.get("damage")
            bbox = ann.get("bbox", [])

            if damage_type in class_map:
                damage_distribution[damage_type] += 1
                total_annotations += 1

                if len(bbox) == 4:
                    x_min, y_min, w, h = bbox

                    # Check for issues
                    if w <= 0 or h <= 0:
                        bbox_issues["zero_size"] += 1
                    if x_min < 0 or y_min < 0:
                        bbox_issues["negative_coords"] += 1
                    if x_min + w > img_width or y_min + h > img_height:
                        bbox_issues["out_of_bounds"] += 1

                    # Record size
                    area = w * h
                    bbox_sizes[damage_type].append(area)
                else:
                    bbox_issues["invalid_format"] += 1
    except Exception as e:
        bbox_issues["parse_error"] += 1

print(f"\n📈 Results (1000 files):")
print(f"   Empty annotations: {empty_annotations}")
print(f"   Missing images: {missing_images}")
print(f"   Total annotations: {total_annotations}")

print(f"\n🏷️  Damage distribution:")
for damage, count in damage_distribution.most_common():
    pct = count / total_annotations * 100 if total_annotations > 0 else 0
    print(f"   {damage:12s}: {count:5d} ({pct:5.1f}%)")

if bbox_issues:
    print(f"\n⚠️  Bbox issues found:")
    for issue, count in bbox_issues.items():
        print(f"   {issue}: {count}")

print(f"\n📏 Average bbox sizes:")
for damage, sizes in bbox_sizes.items():
    if sizes:
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        print(f"   {damage:12s}: avg={avg_size:8.0f}px², min={min_size:6.0f}, max={max_size:8.0f}")

print("\n" + "="*60)
print("✓ Verification complete")
print("="*60)
