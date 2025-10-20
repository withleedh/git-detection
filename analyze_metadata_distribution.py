import json
import os
from collections import defaultdict, Counter
from pathlib import Path
import random

def analyze_dataset(base_path):
    """Analyze metadata distribution in the dataset"""

    # Counters for various metadata
    vehicle_size_counter = Counter()
    color_counter = Counter()
    damage_counter = Counter()
    part_counter = Counter()
    year_counter = Counter()

    # Combination counters
    size_damage_counter = Counter()
    size_color_counter = Counter()
    size_damage_part_counter = Counter()

    # Track annotations per image
    annotations_per_image = []
    damage_annotations_per_image = []
    part_annotations_per_image = []

    # Sample files for analysis
    json_files = list(Path(base_path).rglob("*.json"))
    total_files = len(json_files)

    print(f"Total JSON files found: {total_files:,}")
    print(f"Analyzing metadata distribution...\n")

    # Analyze all files
    for i, json_file in enumerate(json_files):
        if (i + 1) % 50000 == 0:
            print(f"Processed {i+1:,} / {total_files:,} files...")

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Get image-level metadata
            vehicle_size = data['categories']['supercategory_name']
            vehicle_size_counter[vehicle_size] += 1

            # Count annotations
            annotations = data['annotations']
            annotations_per_image.append(len(annotations))

            # Track damage and part annotations separately
            damage_anns = [a for a in annotations if a.get('damage')]
            part_anns = [a for a in annotations if a.get('part')]

            damage_annotations_per_image.append(len(damage_anns))
            part_annotations_per_image.append(len(part_anns))

            # Analyze each annotation
            for ann in annotations:
                color = ann.get('color')
                damage = ann.get('damage')
                part = ann.get('part')
                year = ann.get('year')

                if color:
                    color_counter[color] += 1
                    size_color_counter[(vehicle_size, color)] += 1

                if damage:
                    damage_counter[damage] += 1
                    size_damage_counter[(vehicle_size, damage)] += 1

                if part:
                    part_counter[part] += 1

                if damage and part:
                    size_damage_part_counter[(vehicle_size, damage, part)] += 1

                if year:
                    year_counter[year] += 1

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    # Print results
    print("\n" + "="*80)
    print("METADATA DISTRIBUTION ANALYSIS")
    print("="*80)

    print(f"\n1. Vehicle Size Distribution (Image level - {total_files:,} images):")
    for size, count in vehicle_size_counter.most_common():
        pct = count / total_files * 100
        print(f"  {size:20s}: {count:8,} ({pct:5.2f}%)")

    print(f"\n2. Color Distribution (Annotation level - {sum(color_counter.values()):,} annotations):")
    for color, count in color_counter.most_common(15):
        pct = count / sum(color_counter.values()) * 100
        print(f"  {color:20s}: {count:8,} ({pct:5.2f}%)")
    if len(color_counter) > 15:
        print(f"  ... and {len(color_counter) - 15} more colors")

    print(f"\n3. Damage Type Distribution (Annotation level - {sum(damage_counter.values()):,} annotations):")
    for damage, count in damage_counter.most_common():
        pct = count / sum(damage_counter.values()) * 100
        print(f"  {damage:20s}: {count:8,} ({pct:5.2f}%)")

    print(f"\n4. Part Distribution (Annotation level - {sum(part_counter.values()):,} annotations):")
    for part, count in part_counter.most_common(20):
        pct = count / sum(part_counter.values()) * 100
        print(f"  {part:30s}: {count:8,} ({pct:5.2f}%)")
    if len(part_counter) > 20:
        print(f"  ... and {len(part_counter) - 20} more parts")

    print(f"\n5. Year Distribution (Annotation level - {sum(year_counter.values()):,} annotations):")
    for year, count in sorted(year_counter.items()):
        pct = count / sum(year_counter.values()) * 100
        print(f"  {year}: {count:8,} ({pct:5.2f}%)")

    print(f"\n6. Annotations per Image Statistics:")
    print(f"  Total annotations: {sum(annotations_per_image):8,}")
    print(f"  Damage annotations: {sum(damage_annotations_per_image):8,}")
    print(f"  Part annotations: {sum(part_annotations_per_image):8,}")
    print(f"  Avg total per image: {sum(annotations_per_image)/len(annotations_per_image):6.2f}")
    print(f"  Avg damage per image: {sum(damage_annotations_per_image)/len(damage_annotations_per_image):6.2f}")
    print(f"  Avg part per image: {sum(part_annotations_per_image)/len(part_annotations_per_image):6.2f}")
    print(f"  Max annotations: {max(annotations_per_image):8,}")

    print(f"\n7. Size + Damage Combinations (Top 20):")
    for combo, count in size_damage_counter.most_common(20):
        pct = count / sum(size_damage_counter.values()) * 100
        print(f"  {combo[0]:20s} + {combo[1]:15s}: {count:8,} ({pct:5.2f}%)")

    print(f"\n8. Size + Color Combinations (Top 20):")
    for combo, count in size_color_counter.most_common(20):
        pct = count / sum(size_color_counter.values()) * 100
        print(f"  {combo[0]:20s} + {combo[1]:15s}: {count:8,} ({pct:5.2f}%)")

    print(f"\n9. Combination Statistics:")
    print(f"  Unique Size+Damage combinations: {len(size_damage_counter):,}")
    print(f"  Unique Size+Color combinations: {len(size_color_counter):,}")
    print(f"  Unique Size+Damage+Part combinations: {len(size_damage_part_counter):,}")

    # Sparse combination analysis
    sparse_threshold = 10
    sparse_combos = sum(1 for count in size_damage_part_counter.values() if count < sparse_threshold)
    print(f"\n10. Sparse Combinations (< {sparse_threshold} samples):")
    print(f"  Size+Damage+Part combinations with < {sparse_threshold} samples: {sparse_combos:,} / {len(size_damage_part_counter):,}")

    print("\n" + "="*80)

if __name__ == "__main__":
    # Analyze training data
    train_path = "/Users/dongho/Downloads/new/data/Dataset/1.Training/2.라벨링데이터/damage_part"

    print("Analyzing Training Dataset...")
    analyze_dataset(train_path)
