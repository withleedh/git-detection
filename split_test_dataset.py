import os
import json
import shutil
from pathlib import Path

# 경로 설정
test_dir = "coco_hyundai_kia/test"
output_dir = "coco_hyundai_kia/test_splitted"
json_file = os.path.join(test_dir, "_annotations.coco.json")

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# COCO JSON 로드
with open(json_file, 'r') as f:
    coco_data = json.load(f)

# 이미지 파일 리스트 가져오기
image_files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
total_images = len(image_files)
print(f"총 이미지 수: {total_images}")

# 300개씩 분할
batch_size = 300
num_batches = (total_images + batch_size - 1) // batch_size

print(f"생성될 배치 수: {num_batches}")

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, total_images)
    batch_images = image_files[start_idx:end_idx]

    # 배치 디렉토리 생성
    batch_dir = os.path.join(output_dir, f"batch_{batch_idx + 1}")
    os.makedirs(batch_dir, exist_ok=True)

    print(f"\n배치 {batch_idx + 1}: {len(batch_images)}개 이미지 처리 중...")

    # 이미지 파일명 set
    batch_image_names = set(batch_images)

    # 해당 배치의 이미지 ID 찾기
    batch_image_ids = set()
    batch_coco_images = []

    for img in coco_data['images']:
        if img['file_name'] in batch_image_names:
            batch_image_ids.add(img['id'])
            batch_coco_images.append(img)

    # 해당 이미지들의 어노테이션 찾기
    batch_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in batch_image_ids]

    # 새로운 COCO JSON 생성
    batch_coco = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'images': batch_coco_images,
        'annotations': batch_annotations,
        'categories': coco_data['categories']
    }

    # JSON 저장
    batch_json_path = os.path.join(batch_dir, "_annotations.coco.json")
    with open(batch_json_path, 'w') as f:
        json.dump(batch_coco, f, indent=2)

    # 이미지 복사
    for img_file in batch_images:
        src = os.path.join(test_dir, img_file)
        dst = os.path.join(batch_dir, img_file)
        shutil.copy2(src, dst)

    print(f"  - 이미지: {len(batch_images)}개")
    print(f"  - 어노테이션: {len(batch_annotations)}개")
    print(f"  - 저장 위치: {batch_dir}")

print(f"\n완료! 총 {num_batches}개 배치 생성됨")
