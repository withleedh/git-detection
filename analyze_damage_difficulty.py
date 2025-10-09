import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

labels_dir = Path("/Users/dongho/Downloads/160._차량파손_이미지_데이터/01.데이터/1.Training/2.라벨링데이터/damage_part")
json_files = list(labels_dir.glob("*.json"))[:1000]  # Sample 1000 files

print("="*60)
print("Analyzing damage characteristics (1000 samples)")
print("="*60)

damage_stats = defaultdict(lambda: {
    'count': 0,
    'areas': [],
    'aspect_ratios': []
})

for json_path in json_files:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])

    for ann in annotations:
        damage = ann.get('damage')
        if damage is None:
            continue

        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            area = w * h
            aspect_ratio = w / h if h > 0 else 0

            damage_stats[damage]['count'] += 1
            damage_stats[damage]['areas'].append(area)
            damage_stats[damage]['aspect_ratios'].append(aspect_ratio)

print("\n📊 Damage Type Analysis (Detection Difficulty)")
print("="*60)

for damage_type in sorted(damage_stats.keys()):
    stats = damage_stats[damage_type]

    if stats['count'] == 0:
        continue

    areas = np.array(stats['areas'])
    ratios = np.array(stats['aspect_ratios'])

    print(f"\n🔍 {damage_type.upper()}")
    print(f"   Count: {stats['count']}")
    print(f"   Avg bbox area: {areas.mean():.1f} px²")
    print(f"   Median area: {np.median(areas):.1f} px²")
    print(f"   Min/Max area: {areas.min():.1f} / {areas.max():.1f}")
    print(f"   Avg aspect ratio: {ratios.mean():.2f}")
    print(f"   Area std dev: {areas.std():.1f} (variation)")

print("\n" + "="*60)
print("💡 Detection Difficulty Assessment:")
print("="*60)
print("""
Factors that make detection HARDER:
  ✗ Small bbox area (harder to detect)
  ✗ High area variation (inconsistent size)
  ✗ Extreme aspect ratios (very elongated)
  ✗ Low contrast with background
  ✗ Fine details required

Factors that make detection EASIER:
  ✓ Large bbox area (prominent features)
  ✓ Consistent size (low variation)
  ✓ Balanced aspect ratios
  ✓ Clear boundaries
  ✓ Distinct visual features
""")

# Rank by difficulty
print("\n🏆 Predicted Detection Difficulty (Easiest → Hardest):")
print("="*60)

difficulty_scores = {}
for damage_type, stats in damage_stats.items():
    if stats['count'] == 0:
        continue
    areas = np.array(stats['areas'])

    # Larger area + lower variation = easier
    avg_area = areas.mean()
    area_cv = areas.std() / avg_area if avg_area > 0 else 999  # Coefficient of variation

    # Lower score = easier to detect
    difficulty_score = area_cv * 10000 / avg_area
    difficulty_scores[damage_type] = difficulty_score

sorted_damages = sorted(difficulty_scores.items(), key=lambda x: x[1])

for rank, (damage_type, score) in enumerate(sorted_damages, 1):
    difficulty = "EASY" if score < 5 else "MEDIUM" if score < 10 else "HARD"
    print(f"   {rank}. {damage_type:15s} - {difficulty:8s} (score: {score:.2f})")
