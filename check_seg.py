import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Paths
LABEL_DIR = Path("/Users/dongho/Downloads/new/data/Dataset/1.Training/2.라벨링데이터/damage_part")
ESTIMATE_DIR = Path("/Users/dongho/Downloads/new/data/Dataset/1.Training/1.원천데이터_230126_add/TS_99. 붙임_견적서")

HYUNDAI_KIA = ["현대", "현대 / 승용", "현대 / RV", "기아", "기아 / 승용", "기아 / RV"]

print("1. Loading estimates...")
estimates = {}
for json_file in tqdm(list(ESTIMATE_DIR.glob("*.json"))):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            estimates[json_file.stem] = data.get('차량정보', {}).get('제작사/차종', '')
    except:
        continue

print(f"   Total estimates: {len(estimates):,}")

hyundai_kia_ids = {k for k, v in estimates.items() if v in HYUNDAI_KIA}
print(f"   Hyundai+Kia estimates: {len(hyundai_kia_ids):,}")

print("\n2. Checking labeling data...")
total_labels = 0
matched_labels = 0
unmatched_labels = 0

for json_file in tqdm(list(LABEL_DIR.glob("*.json"))):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            category_id = data['categories']['id']
            total_labels += 1

            if category_id in hyundai_kia_ids:
                matched_labels += 1
            elif category_id in estimates:
                unmatched_labels += 1
    except:
        continue

print(f"\n=== Results ===")
print(f"Total labeling JSONs: {total_labels:,}")
print(f"Matched (Hyundai+Kia): {matched_labels:,} ({matched_labels/total_labels*100:.1f}%)")
print(f"Unmatched (other brands): {unmatched_labels:,} ({unmatched_labels/total_labels*100:.1f}%)")
print(f"No estimate found: {total_labels - matched_labels - unmatched_labels:,}")
