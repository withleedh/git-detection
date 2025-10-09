"""
Analyze Breakage bbox sizes to determine if larger image size would help
"""
import json
from pathlib import Path
import numpy as np

labels_dir = Path("/Users/dongho/Downloads/160._차량파손_이미지_데이터/01.데이터/1.Training/2.라벨링데이터/damage_part")
json_files = list(labels_dir.glob("*.json"))[:2000]  # Sample 2000 files

print("="*60)
print("Analyzing Breakage characteristics")
print("="*60)

breakage_stats = {
    'areas': [],
    'widths': [],
    'heights': [],
    'relative_areas': []  # Relative to image size
}

for json_path in json_files:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_info = data.get('images', {})
    img_width = image_info.get('width', 0)
    img_height = image_info.get('height', 0)

    if img_width == 0 or img_height == 0:
        continue

    img_area = img_width * img_height

    annotations = data.get('annotations', [])

    for ann in annotations:
        if ann.get('damage') == 'Breakage':
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                area = w * h

                breakage_stats['areas'].append(area)
                breakage_stats['widths'].append(w)
                breakage_stats['heights'].append(h)
                breakage_stats['relative_areas'].append(area / img_area * 100)

if not breakage_stats['areas']:
    print("No Breakage annotations found!")
    exit()

areas = np.array(breakage_stats['areas'])
widths = np.array(breakage_stats['widths'])
heights = np.array(breakage_stats['heights'])
relative_areas = np.array(breakage_stats['relative_areas'])

print(f"\n📊 Breakage Statistics (n={len(areas)})")
print("="*60)
print(f"\nBounding Box Areas (pixels²):")
print(f"  Mean:   {areas.mean():,.0f} px²")
print(f"  Median: {np.median(areas):,.0f} px²")
print(f"  Min:    {areas.min():,.0f} px²")
print(f"  Max:    {areas.max():,.0f} px²")
print(f"  Std:    {areas.std():,.0f} px²")

print(f"\nBounding Box Dimensions:")
print(f"  Mean width:  {widths.mean():.1f} px")
print(f"  Mean height: {heights.mean():.1f} px")
print(f"  Min width:   {widths.min():.1f} px")
print(f"  Min height:  {heights.min():.1f} px")

print(f"\nRelative Size (% of image area):")
print(f"  Mean:   {relative_areas.mean():.2f}%")
print(f"  Median: {np.median(relative_areas):.2f}%")

# Categorize by size
small = (areas < 2000).sum()
medium = ((areas >= 2000) & (areas < 10000)).sum()
large = (areas >= 10000).sum()

print(f"\nSize Distribution:")
print(f"  Small  (< 2,000 px²):    {small:4d} ({small/len(areas)*100:.1f}%)")
print(f"  Medium (2K-10K px²):     {medium:4d} ({medium/len(areas)*100:.1f}%)")
print(f"  Large  (> 10,000 px²):   {large:4d} ({large/len(areas)*100:.1f}%)")

print("\n" + "="*60)
print("💡 Image Size Recommendation")
print("="*60)

avg_area = areas.mean()
avg_relative = relative_areas.mean()

if avg_relative < 1.0:
    print("⚠️  Breakage objects are VERY SMALL (< 1% of image)")
    print("✅  Recommendation: img_size = 1280 or 1536")
    print("    → Larger images will preserve small details")
elif avg_relative < 5.0:
    print("⚠️  Breakage objects are SMALL (1-5% of image)")
    print("✅  Recommendation: img_size = 1280")
    print("    → Some benefit from larger images")
elif avg_relative < 15.0:
    print("✅  Breakage objects are MEDIUM (5-15% of image)")
    print("⚙️  Recommendation: img_size = 640 is sufficient")
    print("    → Minimal benefit from larger images")
else:
    print("✅  Breakage objects are LARGE (> 15% of image)")
    print("⚙️  Recommendation: img_size = 640 is sufficient")
    print("    → No benefit from larger images")

print("\n" + "="*60)
print("📌 Key Insight:")
print("="*60)
print("""
If Breakage objects are LARGE (> 10% of image area):
  → img_size 증가는 효과 적음
  → 대신 더 큰 모델 (yolo11m/l) 사용 권장

If Breakage objects are SMALL (< 3% of image area):
  → img_size 증가가 도움됨
  → 1280 또는 1536 사용 권장
""")
