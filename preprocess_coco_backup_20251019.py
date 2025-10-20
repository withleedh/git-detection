"""
Vehicle Scratch Detection - Data Preprocessing Script (COCO Format)
Converts raw dataset to COCO format for instance segmentation training
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List
import random
from datetime import datetime


class COCODataPreprocessor:
    """Preprocesses vehicle damage dataset for COCO format instance segmentation"""

    def __init__(
        self,
        raw_data_path: str,
        output_path: str = "./datasets/damage_coco",
        train_split: float = 0.8,
        val_split: float = 0.2
    ):
        """
        Initialize preprocessor

        Args:
            raw_data_path: Path to raw dataset root
            output_path: Path to save processed COCO dataset
            train_split: Ratio for training data
            val_split: Ratio for validation data
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.train_split = train_split
        self.val_split = val_split

        # Dataset paths
        self.images_dir = self.raw_data_path / "data/Dataset/1.Training/1.원천데이터/damage_part"
        self.labels_dir = self.raw_data_path / "data/Dataset/1.Training/2.라벨링데이터/damage_part"
        self.estimate_dir = self.raw_data_path / "1.Training/1.원천데이터_230126_add/TS_99. 붙임_견적서"

        # Output paths
        self.train_images = self.output_path / "train"
        self.val_images = self.output_path / "val"

        # COCO format structures
        self.train_coco = {
            "info": {
                "description": "Vehicle Damage Instance Segmentation Dataset",
                "version": "1.0",
                "year": 2025,
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        self.val_coco = {
            "info": {
                "description": "Vehicle Damage Instance Segmentation Dataset",
                "version": "1.0",
                "year": 2025,
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        # Class mapping for damage types
        self.class_map = {
            "Breakage": 1,
            "Crushed": 2,
            "Separated": 3,
            "Scratched": 4
        }

        # Initialize categories
        self.categories = [
            {"id": 1, "name": "Breakage", "supercategory": "damage"},
            {"id": 2, "name": "Crushed", "supercategory": "damage"},
            {"id": 3, "name": "Separated", "supercategory": "damage"},
            {"id": 4, "name": "Scratched", "supercategory": "damage"}
        ]

        # Statistics
        self.stats = {
            "total_files": 0,
            "files_with_damages": 0,
            "total_annotations": 0,
            "class_counts": {
                "Breakage": 0,
                "Crushed": 0,
                "Separated": 0,
                "Scratched": 0
            },
            "train_count": 0,
            "val_count": 0,
            "skipped_files": 0
        }

        # Running counters for COCO IDs
        self.image_id_counter = 1
        self.annotation_id_counter = 1

        # Hyundai/Kia filtering
        self.hyundai_kia_ids = None

    def setup_directories(self):
        """Create output directory structure"""
        for dir_path in [self.train_images, self.val_images]:
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created output directories at {self.output_path}")

    def load_hyundai_kia_vehicles(self):
        """Load Hyundai/Kia vehicle IDs from estimate files"""
        print("\n🚗 Loading Hyundai/Kia vehicle list from estimates...")

        if not self.estimate_dir.exists():
            print(f"⚠️  Estimate directory not found: {self.estimate_dir}")
            print("   Processing all vehicles without filtering")
            return None

        hyundai_kia_ids = set()
        manufacturer_counts = {}

        estimate_files = list(self.estimate_dir.glob("*.json"))
        print(f"   Found {len(estimate_files)} estimate files")

        for estimate_file in estimate_files:
            try:
                with open(estimate_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                vehicle_info = data.get('차량정보', {})
                manufacturer = vehicle_info.get('제작사/차종', '')

                # Check if Hyundai or Kia
                if manufacturer and ('현대' in manufacturer or '기아' in manufacturer):
                    vehicle_id = estimate_file.stem
                    hyundai_kia_ids.add(vehicle_id)
                    manufacturer_counts[manufacturer] = manufacturer_counts.get(manufacturer, 0) + 1

            except Exception:
                continue

        print(f"✓ Found {len(hyundai_kia_ids)} Hyundai/Kia vehicles")
        print(f"   Top manufacturers:")
        for mfr, count in sorted(manufacturer_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     - {mfr}: {count}")

        return hyundai_kia_ids

    def is_hyundai_kia_vehicle(self, json_filename: str) -> bool:
        """Check if vehicle is Hyundai/Kia based on filename"""
        if self.hyundai_kia_ids is None:
            return True  # No filtering if IDs not loaded

        # Extract vehicle ID from filename (e.g., "0000010_sc-186320.json" -> "sc-186320")
        if '_' in json_filename:
            vehicle_id = json_filename.split('_')[1].replace('.json', '')
            return vehicle_id in self.hyundai_kia_ids

        return False

    def convert_segmentation_format(self, segmentation: List) -> List:
        """
        Convert segmentation from nested list to COCO polygon format

        Args:
            segmentation: [[[[x1,y1], [x2,y2], ...]]] format

        Returns:
            [[x1, y1, x2, y2, ...]] format (COCO standard)
        """
        if not segmentation or len(segmentation) == 0:
            return []

        # Extract the polygon points from nested structure
        polygon = segmentation[0][0] if isinstance(segmentation[0][0], list) else segmentation[0]

        # Flatten to [x1, y1, x2, y2, ...] format
        flattened = []
        for point in polygon:
            if isinstance(point, list) and len(point) == 2:
                flattened.extend(point)

        return [flattened] if flattened else []

    def process_json_file(self, json_path: Path) -> Dict:
        """
        Process single JSON label file

        Args:
            json_path: Path to JSON file

        Returns:
            Processed data with image info and annotations
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract image info
        image_info = data.get("images", {})
        img_width = image_info.get("width", 0)
        img_height = image_info.get("height", 0)
        img_filename = image_info.get("file_name", "")

        if not all([img_width, img_height, img_filename]):
            return None

        # Filter damage annotations (ignore part annotations)
        annotations = data.get("annotations", [])
        damage_annotations = []

        for ann in annotations:
            damage_type = ann.get("damage")

            # Only process damage annotations (skip part annotations where damage is None)
            if damage_type and damage_type in self.class_map:
                segmentation = ann.get("segmentation", [])
                bbox = ann.get("bbox", [])
                area = ann.get("area", 0)

                if segmentation and bbox:
                    # Convert segmentation format
                    coco_segmentation = self.convert_segmentation_format(segmentation)

                    if coco_segmentation:
                        damage_annotations.append({
                            "category_id": self.class_map[damage_type],
                            "segmentation": coco_segmentation,
                            "bbox": bbox,
                            "area": area,
                            "damage_type": damage_type
                        })

        if not damage_annotations:
            return None

        return {
            "image_info": {
                "file_name": img_filename,
                "width": img_width,
                "height": img_height
            },
            "annotations": damage_annotations
        }

    def process_dataset(self):
        """Main processing pipeline"""
        print("🚀 Starting COCO dataset preprocessing...")
        print(f"📂 Source: {self.labels_dir}")
        print(f"📂 Target: {self.output_path}\n")

        # Load Hyundai/Kia vehicle IDs
        self.hyundai_kia_ids = self.load_hyundai_kia_vehicles()

        # Get all JSON files
        json_files = list(self.labels_dir.glob("*.json"))
        self.stats["total_files"] = len(json_files)

        # Filter for Hyundai/Kia vehicles if IDs loaded
        if self.hyundai_kia_ids is not None:
            original_count = len(json_files)
            json_files = [f for f in json_files if self.is_hyundai_kia_vehicle(f.name)]
            print(f"\n🔍 Filtered: {len(json_files)} / {original_count} files are Hyundai/Kia vehicles")
        else:
            print(f"\nFound {len(json_files)} label files (no filtering)")

        # Shuffle for random split
        random.seed(42)
        random.shuffle(json_files)

        # Calculate split index
        split_idx = int(len(json_files) * self.train_split)

        train_files = json_files[:split_idx]
        val_files = json_files[split_idx:]

        # Set categories for both splits
        self.train_coco["categories"] = self.categories
        self.val_coco["categories"] = self.categories

        # Process training set
        print(f"\n📊 Processing training set ({len(train_files)} files)...")
        self._process_split(train_files, self.train_images, self.train_coco, "train")

        # Process validation set
        print(f"\n📊 Processing validation set ({len(val_files)} files)...")
        self._process_split(val_files, self.val_images, self.val_coco, "val")

        # Save COCO JSON files
        self._save_coco_files()

        # Print statistics
        self._print_statistics()

    def _process_split(
        self,
        json_files: List[Path],
        img_output: Path,
        coco_dict: Dict,
        split_name: str
    ):
        """Process a data split (train/val)"""
        processed = 0

        for i, json_path in enumerate(json_files, 1):
            # Process label file
            result = self.process_json_file(json_path)

            # Skip if no damage annotations
            if not result:
                self.stats["skipped_files"] += 1
                continue

            img_filename = result["image_info"]["file_name"]
            img_src_path = self.images_dir / img_filename

            # Skip if image doesn't exist
            if not img_src_path.exists():
                self.stats["skipped_files"] += 1
                continue

            # Copy image
            img_dst_path = img_output / img_filename
            shutil.copy2(img_src_path, img_dst_path)

            # Add image info to COCO
            image_entry = {
                "id": self.image_id_counter,
                "file_name": img_filename,
                "width": result["image_info"]["width"],
                "height": result["image_info"]["height"]
            }
            coco_dict["images"].append(image_entry)

            # Add annotations to COCO
            for ann in result["annotations"]:
                annotation_entry = {
                    "id": self.annotation_id_counter,
                    "image_id": self.image_id_counter,
                    "category_id": ann["category_id"],
                    "segmentation": ann["segmentation"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": 0
                }
                coco_dict["annotations"].append(annotation_entry)

                # Update statistics
                self.stats["class_counts"][ann["damage_type"]] += 1
                self.stats["total_annotations"] += 1
                self.annotation_id_counter += 1

            # Update counters
            self.image_id_counter += 1
            processed += 1
            self.stats["files_with_damages"] += 1
            self.stats[f"{split_name}_count"] += 1

            # Progress indicator
            if i % 1000 == 0:
                print(f"  Processed {i}/{len(json_files)} files... ({processed} with damages)")

        print(f"✓ {split_name.capitalize()} set: {processed} images with {len(coco_dict['annotations'])} annotations")

    def _save_coco_files(self):
        """Save COCO JSON annotation files"""
        train_json_path = self.train_images / "_annotations.coco.json"
        val_json_path = self.val_images / "_annotations.coco.json"

        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_coco, f, indent=2, ensure_ascii=False)

        with open(val_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.val_coco, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Saved COCO annotations:")
        print(f"  - {train_json_path}")
        print(f"  - {val_json_path}")

    def _print_statistics(self):
        """Print processing statistics"""
        print("\n" + "="*60)
        print("📈 COCO INSTANCE SEGMENTATION PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Total files scanned:          {self.stats['total_files']:,}")
        print(f"Files with damages:           {self.stats['files_with_damages']:,}")
        print(f"Total annotations:            {self.stats['total_annotations']:,}")
        print(f"Skipped files:                {self.stats['skipped_files']:,}")

        print(f"\n📊 Damage Type Distribution:")
        for damage_type, count in self.stats['class_counts'].items():
            percentage = (count / self.stats['total_annotations'] * 100) if self.stats['total_annotations'] > 0 else 0
            print(f"  {damage_type:12s}: {count:6,} instances ({percentage:5.1f}%)")

        print(f"\nTraining images:              {self.stats['train_count']:,}")
        print(f"Validation images:            {self.stats['val_count']:,}")
        print(f"Train/Val split:              {self.train_split:.0%} / {self.val_split:.0%}")
        print("="*60)
        print(f"✅ COCO dataset ready at: {self.output_path.absolute()}\n")


def main():
    """Main execution"""
    # Configuration
    RAW_DATA_PATH = "./new"
    OUTPUT_PATH = "./datasets/damage_coco_new"

    # Initialize preprocessor
    preprocessor = COCODataPreprocessor(
        raw_data_path=RAW_DATA_PATH,
        output_path=OUTPUT_PATH,
        train_split=0.8,
        val_split=0.2
    )

    # Setup and process
    preprocessor.setup_directories()
    preprocessor.process_dataset()


if __name__ == "__main__":
    main()
