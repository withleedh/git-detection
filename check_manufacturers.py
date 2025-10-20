import json
from pathlib import Path
from collections import Counter

estimate_dir = Path("/Users/dongho/Downloads/new/data/Dataset/1.Training/1.원천데이터_230126_add/TS_99. 붙임_견적서")

manufacturers = []
models = []

# Sample 1000 files
for i, json_file in enumerate(estimate_dir.glob("*.json")):
    if i >= 1000:
        break

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            manufacturer = data.get('차량정보', {}).get('제작사/차종', 'Unknown')
            model = data.get('차량정보', {}).get('차량명칭', 'Unknown')
            manufacturers.append(manufacturer)
            models.append(f"{manufacturer} - {model}")
    except:
        continue

print("=== 제조사 분포 (1000개 샘플) ===")
for mfr, count in Counter(manufacturers).most_common():
    print(f"{mfr:15s}: {count:4d} ({count/len(manufacturers)*100:5.1f}%)")

print(f"\n=== 차량 모델 분포 (Top 20) ===")
for model, count in Counter(models).most_common(20):
    print(f"{model:40s}: {count:4d}")
