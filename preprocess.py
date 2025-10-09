"""
Vehicle Scratch Detection - Data Preprocessing Script
Converts raw dataset to YOLO format for training
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random


class ScratchDataPreprocessor:
    """Preprocesses vehicle damage dataset for YOLO training"""

    def __init__(
        self,
        raw_data_path: str,
        output_path: str = "./datasets/scratch",
        train_split: float = 0.8,
        val_split: float = 0.2
    ):
        """
        Initialize preprocessor

        Args:
            raw_data_path: Path to raw dataset root
            output_path: Path to save processed YOLO dataset
            train_split: Ratio for training data
            val_split: Ratio for validation data
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.train_split = train_split
        self.val_split = val_split

        # Dataset paths
        self.images_dir = self.raw_data_path / "01.데이터/1.Training/1.원천데이터/damage_part"
        self.labels_dir = self.raw_data_path / "01.데이터/1.Training/2.라벨링데이터/damage_part"

        # Output paths
        self.train_images = self.output_path / "images/train"
        self.train_labels = self.output_path / "labels/train"
        self.val_images = self.output_path / "images/val"
        self.val_labels = self.output_path / "labels/val"

        # Statistics
        self.stats = {
            "total_files": 0,
            "files_with_damages": 0,
            "total_damage_boxes": 0,
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

    def setup_directories(self):
        """Create output directory structure"""
        for dir_path in [self.train_images, self.train_labels,
                         self.val_images, self.val_labels]:
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created output directories at {self.output_path}")

    def bbox_to_yolo_format(
        self,
        bbox: List[int],
        img_width: int,
        img_height: int
    ) -> Tuple[float, float, float, float]:
        """
        Convert bbox to YOLO format

        Args:
            bbox: [x_min, y_min, width, height]
            img_width: Image width
            img_height: Image height

        Returns:
            (x_center, y_center, width, height) normalized to 0-1
        """
        x_min, y_min, w, h = bbox

        # Calculate center coordinates
        x_center = (x_min + w / 2) / img_width
        y_center = (y_min + h / 2) / img_height

        # Normalize width and height
        norm_width = w / img_width
        norm_height = h / img_height

        # Ensure values are within 0-1
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_width = max(0.0, min(1.0, norm_width))
        norm_height = max(0.0, min(1.0, norm_height))

        return x_center, y_center, norm_width, norm_height

    def process_json_file(self, json_path: Path) -> Tuple[List[str], Dict]:
        """
        Process single JSON label file

        Args:
            json_path: Path to JSON file

        Returns:
            (yolo_annotations, image_info)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract image info
        image_info = data.get("images", {})
        img_width = image_info.get("width", 0)
        img_height = image_info.get("height", 0)
        img_filename = image_info.get("file_name", "")

        if not all([img_width, img_height, img_filename]):
            return [], {}

        # Filter multi-class damage annotations
        annotations = data.get("annotations", [])
        yolo_annotations = []
        damage_types_found = []  # Track which damage types are in this file

        # Class mapping for multi-class detection
        class_map = {
            "Breakage": 0,
            "Crushed": 1,
            "Separated": 2,
            "Scratched": 3
        }

        for ann in annotations:
            damage_type = ann.get("damage")

            # Process all damage types (multi-class)
            if damage_type in class_map:
                class_id = class_map[damage_type]
                bbox = ann.get("bbox")
                if bbox and len(bbox) == 4:
                    # Convert to YOLO format
                    x_c, y_c, w, h = self.bbox_to_yolo_format(
                        bbox, img_width, img_height
                    )

                    # Multi-class YOLO format
                    yolo_line = f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                    yolo_annotations.append(yolo_line)
                    damage_types_found.append(damage_type)

        return yolo_annotations, image_info, damage_types_found

    def process_dataset(self):
        """Main processing pipeline"""
        print("🚀 Starting dataset preprocessing...")
        print(f"📂 Source: {self.labels_dir}")
        print(f"📂 Target: {self.output_path}\n")

        # Get all JSON files
        json_files = list(self.labels_dir.glob("*.json"))
        self.stats["total_files"] = len(json_files)

        print(f"Found {len(json_files)} label files")

        # Shuffle for random split
        random.seed(42)
        random.shuffle(json_files)

        # Calculate split index
        split_idx = int(len(json_files) * self.train_split)

        train_files = json_files[:split_idx]
        val_files = json_files[split_idx:]

        # Process training set
        print(f"\n📊 Processing training set ({len(train_files)} files)...")
        self._process_split(train_files, self.train_images, self.train_labels, "train")

        # Process validation set
        print(f"\n📊 Processing validation set ({len(val_files)} files)...")
        self._process_split(val_files, self.val_images, self.val_labels, "val")

        # Print statistics
        self._print_statistics()

    def _process_split(
        self,
        json_files: List[Path],
        img_output: Path,
        label_output: Path,
        split_name: str
    ):
        """Process a data split (train/val)"""
        processed = 0

        for i, json_path in enumerate(json_files, 1):
            # Process label file
            yolo_annotations, image_info, damage_types_found = self.process_json_file(json_path)

            # Skip if no damage annotations
            if not yolo_annotations:
                self.stats["skipped_files"] += 1
                continue

            img_filename = image_info.get("file_name")
            img_src_path = self.images_dir / img_filename

            # Skip if image doesn't exist
            if not img_src_path.exists():
                self.stats["skipped_files"] += 1
                continue

            # Copy image
            img_dst_path = img_output / img_filename
            shutil.copy2(img_src_path, img_dst_path)

            # Save YOLO label
            label_filename = img_filename.rsplit('.', 1)[0] + '.txt'
            label_dst_path = label_output / label_filename

            with open(label_dst_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

            # Update statistics
            processed += 1
            self.stats["files_with_damages"] += 1
            self.stats["total_damage_boxes"] += len(yolo_annotations)
            self.stats[f"{split_name}_count"] += 1

            # Count each damage type
            for damage_type in damage_types_found:
                if damage_type in self.stats["class_counts"]:
                    self.stats["class_counts"][damage_type] += 1

            # Progress indicator
            if i % 1000 == 0:
                print(f"  Processed {i}/{len(json_files)} files... ({processed} with damages)")

        print(f"✓ {split_name.capitalize()} set: {processed} files with damage annotations")

    def _print_statistics(self):
        """Print processing statistics"""
        print("\n" + "="*60)
        print("📈 MULTI-CLASS PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Total files scanned:          {self.stats['total_files']:,}")
        print(f"Files with damages:           {self.stats['files_with_damages']:,}")
        print(f"Total damage bounding boxes:  {self.stats['total_damage_boxes']:,}")
        print(f"Skipped files:                {self.stats['skipped_files']:,}")

        print(f"\n📊 Damage Type Distribution:")
        for damage_type, count in self.stats['class_counts'].items():
            percentage = (count / self.stats['total_damage_boxes'] * 100) if self.stats['total_damage_boxes'] > 0 else 0
            print(f"  {damage_type:12s}: {count:6,} boxes ({percentage:5.1f}%)")

        print(f"\nTraining samples:             {self.stats['train_count']:,}")
        print(f"Validation samples:           {self.stats['val_count']:,}")
        print(f"Train/Val split:              {self.train_split:.0%} / {self.val_split:.0%}")
        print("="*60)
        print(f"✅ Multi-class dataset ready at: {self.output_path.absolute()}\n")


def main():
    """Main execution"""
    # Configuration
    RAW_DATA_PATH = "/Users/dongho/Downloads/160._차량파손_이미지_데이터"
    OUTPUT_PATH = "./datasets/damage_multiclass"  # Changed for multi-class

    # Initialize preprocessor
    preprocessor = ScratchDataPreprocessor(
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
