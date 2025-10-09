import json
from pathlib import Path
from collections import Counter

# Find JSON files
labels_dir = Path("/Users/dongho/Downloads/160._차량파손_이미지_데이터/01.데이터/1.Training/2.라벨링데이터/damage_part")
json_files = list(labels_dir.glob("*.json"))[:5]

print("="*60)
print("Checking damage types in 5 sample JSON files")
print("="*60)

all_damages = []

for i, json_path in enumerate(json_files, 1):
    print(f"\n{i}. {json_path.name}")
    print("-" * 60)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])
    damages_in_file = [ann.get('damage') for ann in annotations]

    print(f"   Total annotations: {len(annotations)}")
    print(f"   Damage types in this file: {set(damages_in_file)}")

    all_damages.extend(damages_in_file)

print("\n" + "="*60)
print("SUMMARY - All damage types found:")
print("="*60)
damage_counts = Counter(all_damages)
for damage_type, count in damage_counts.most_common():
    print(f"  {damage_type}: {count}")
