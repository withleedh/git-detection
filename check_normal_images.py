"""
Check how many images have no damage annotations
"""
import json
from pathlib import Path

labels_dir = Path("/Users/dongho/Downloads/160._차량파손_이미지_데이터/01.데이터/1.Training/2.라벨링데이터/damage_part")
json_files = list(labels_dir.glob("*.json"))[:5000]  # Sample 5000

damage_types = ["Breakage", "Crushed", "Separated", "Scratched"]

has_damage = 0
no_damage = 0

for json_path in json_files:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])

    # Check if any annotation has damage
    found_damage = False
    for ann in annotations:
        if ann.get('damage') in damage_types:
            found_damage = True
            break

    if found_damage:
        has_damage += 1
    else:
        no_damage += 1

print("="*60)
print("📊 Dataset Composition Analysis (5000 samples)")
print("="*60)
print(f"Images with damage:    {has_damage:5d} ({has_damage/len(json_files)*100:.1f}%)")
print(f"Images without damage: {no_damage:5d} ({no_damage/len(json_files)*100:.1f}%)")
print("="*60)

print("\n💡 Recommendation:")
if no_damage < has_damage * 0.1:
    print("✅ Very few normal images - current approach (exclude) is fine")
    print("   Adding normal images would have minimal benefit")
elif no_damage < has_damage * 0.3:
    print("⚙️  Some normal images exist - optional to include")
    print("   Including them may reduce false positives slightly")
else:
    print("⚠️  Many normal images - consider including some")
    print("   Including 10-20% normal images can improve robustness")

print("\n📌 Current preprocessing: Excludes all images without damage")
print("   This is standard for object detection tasks")
