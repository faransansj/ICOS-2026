"""
Dataset Builder Module
Complete pipeline for synthetic dataset generation
"""

import os
import json
import random
from typing import List, Dict, Tuple
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .text_image_generator import TextImageGenerator
from .highlight_overlay import HighlightOverlay
from .data_augmentation import DataAugmentation, LightAugmentation


class DatasetBuilder:
    """Build complete synthetic dataset with highlights"""

    def __init__(
        self,
        output_base_dir: str = "data",
        val_ratio: float = 0.3,
        test_ratio: float = 0.7
    ):
        """
        Initialize DatasetBuilder

        Args:
            output_base_dir: Base directory for dataset
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
        """
        self.output_base_dir = output_base_dir
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Create output directories
        self.synthetic_dir = os.path.join(output_base_dir, "synthetic")
        self.validation_dir = os.path.join(output_base_dir, "validation")
        self.test_dir = os.path.join(output_base_dir, "test")

        for dir_path in [self.synthetic_dir, self.validation_dir, self.test_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Initialize generators
        self.text_generator = TextImageGenerator()
        self.highlight_overlay = HighlightOverlay(alpha=0.3, irregularity=0.15)
        self.augmentor = DataAugmentation()
        self.light_augmentor = LightAugmentation()

    def generate_base_dataset(
        self,
        num_images: int = 200,
        colors: List[str] = None
    ) -> List[Dict]:
        """
        Generate base synthetic dataset with highlights

        Args:
            num_images: Number of base images to generate
            colors: Highlight colors to use

        Returns:
            List of image annotations
        """
        print(f"\n{'='*60}")
        print(f"Generating {num_images} base synthetic images...")
        print(f"{'='*60}\n")

        if colors is None:
            colors = ['yellow', 'green', 'pink']

        all_data = []

        for i in tqdm(range(num_images), desc="Generating base dataset"):
            # Generate text image
            text_img, text_annotations = self.text_generator.generate_text_image()

            # Convert PIL to numpy
            text_img_np = np.array(text_img)
            text_img_np = cv2.cvtColor(text_img_np, cv2.COLOR_RGB2BGR)

            # Apply highlights with consecutive tendency
            highlighted_img, highlight_annotations = \
                self.highlight_overlay.apply_consecutive_highlights(
                    text_img_np,
                    text_annotations,
                    consecutive_prob=0.3,
                    colors=colors
                )

            # Save image
            image_name = f"synthetic_{i:04d}.png"
            image_path = os.path.join(self.synthetic_dir, image_name)
            cv2.imwrite(image_path, highlighted_img)

            # Store annotations
            all_data.append({
                'image_id': i,
                'image_name': image_name,
                'image_path': image_path,
                'annotations': text_annotations,
                'highlight_annotations': highlight_annotations
            })

        # Save base annotations
        annotations_path = os.path.join(self.synthetic_dir, 'base_annotations.json')
        with open(annotations_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Generated {num_images} base images")
        print(f"✓ Saved to: {self.synthetic_dir}")

        return all_data

    def split_dataset(
        self,
        annotations: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split dataset into validation and test sets

        Args:
            annotations: List of all annotations

        Returns:
            Tuple of (validation_data, test_data)
        """
        print(f"\n{'='*60}")
        print("Splitting dataset...")
        print(f"{'='*60}\n")

        # Stratified split by highlight color distribution
        # Calculate color distribution for each image
        def get_color_signature(data):
            colors = [h['color'] for h in data['highlight_annotations']]
            return '_'.join(sorted(set(colors)))

        color_signatures = [get_color_signature(d) for d in annotations]

        # Split
        indices = list(range(len(annotations)))
        val_indices, test_indices = train_test_split(
            indices,
            test_size=self.test_ratio,
            stratify=color_signatures,
            random_state=42
        )

        val_data = [annotations[i] for i in val_indices]
        test_data = [annotations[i] for i in test_indices]

        print(f"✓ Validation set: {len(val_data)} images ({len(val_data)/len(annotations)*100:.1f}%)")
        print(f"✓ Test set: {len(test_data)} images ({len(test_data)/len(annotations)*100:.1f}%)")

        return val_data, test_data

    def augment_and_save(
        self,
        data: List[Dict],
        output_dir: str,
        split_name: str,
        num_augmentations: int = 2,
        use_light_augmentation: bool = False
    ) -> List[Dict]:
        """
        Apply augmentation and save to directory

        Args:
            data: List of image data
            output_dir: Output directory
            split_name: Name of split (validation/test)
            num_augmentations: Number of augmentations per image
            use_light_augmentation: Use lighter augmentation

        Returns:
            Combined original + augmented annotations
        """
        print(f"\n{'='*60}")
        print(f"Processing {split_name} set...")
        print(f"{'='*60}\n")

        augmentor = self.light_augmentor if use_light_augmentation else self.augmentor
        all_annotations = []
        image_id = 0

        for original_data in tqdm(data, desc=f"Processing {split_name}"):
            # Read original image
            orig_image = cv2.imread(original_data['image_path'])

            # Save original to new directory
            orig_image_name = f"{split_name}_{image_id:04d}_orig.png"
            orig_image_path = os.path.join(output_dir, orig_image_name)
            cv2.imwrite(orig_image_path, orig_image)

            # Original annotation
            all_annotations.append({
                'image_id': image_id,
                'image_name': orig_image_name,
                'image_path': orig_image_path,
                'annotations': original_data['annotations'],
                'highlight_annotations': original_data['highlight_annotations'],
                'is_augmented': False
            })
            image_id += 1

            # Generate augmentations
            highlight_annots = original_data['highlight_annotations']
            bboxes = [annot['bbox'] for annot in highlight_annots]
            labels = [annot['color'] for annot in highlight_annots]

            for aug_idx in range(num_augmentations):
                # Apply augmentation
                aug_image, aug_bboxes, aug_labels = augmentor.augment_image(
                    orig_image, bboxes, labels
                )

                # Save augmented image
                aug_image_name = f"{split_name}_{image_id:04d}_aug{aug_idx}.png"
                aug_image_path = os.path.join(output_dir, aug_image_name)
                cv2.imwrite(aug_image_path, aug_image)

                # Update annotations
                aug_highlight_annots = []
                for bbox, label, orig_annot in zip(aug_bboxes, aug_labels, highlight_annots):
                    aug_highlight_annots.append({
                        'text': orig_annot['text'],
                        'bbox': list(bbox),
                        'color': label
                    })

                all_annotations.append({
                    'image_id': image_id,
                    'image_name': aug_image_name,
                    'image_path': aug_image_path,
                    'annotations': original_data['annotations'],
                    'highlight_annotations': aug_highlight_annots,
                    'is_augmented': True,
                    'augmentation_index': aug_idx
                })
                image_id += 1

        # Save annotations
        annotations_path = os.path.join(output_dir, f'{split_name}_annotations.json')
        with open(annotations_path, 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved {len(all_annotations)} images to {output_dir}")

        return all_annotations

    def generate_statistics(
        self,
        val_annotations: List[Dict],
        test_annotations: List[Dict]
    ) -> Dict:
        """
        Generate dataset statistics

        Args:
            val_annotations: Validation annotations
            test_annotations: Test annotations

        Returns:
            Statistics dictionary
        """
        print(f"\n{'='*60}")
        print("Dataset Statistics")
        print(f"{'='*60}\n")

        def analyze_split(annotations, split_name):
            total_images = len(annotations)
            total_highlights = sum(len(a['highlight_annotations']) for a in annotations)

            # Color distribution
            color_counts = {'yellow': 0, 'green': 0, 'pink': 0}
            for annot in annotations:
                for h in annot['highlight_annotations']:
                    color_counts[h['color']] += 1

            # Highlight area distribution
            areas = []
            for annot in annotations:
                for h in annot['highlight_annotations']:
                    bbox = h['bbox']
                    area = bbox[2] * bbox[3]
                    areas.append(area)

            stats = {
                'split_name': split_name,
                'total_images': total_images,
                'total_highlights': total_highlights,
                'avg_highlights_per_image': total_highlights / total_images if total_images > 0 else 0,
                'color_distribution': color_counts,
                'avg_highlight_area': np.mean(areas) if areas else 0,
                'std_highlight_area': np.std(areas) if areas else 0
            }

            print(f"{split_name.upper()} SET:")
            print(f"  Total images: {stats['total_images']}")
            print(f"  Total highlights: {stats['total_highlights']}")
            print(f"  Avg highlights/image: {stats['avg_highlights_per_image']:.2f}")
            print(f"  Color distribution:")
            for color, count in color_counts.items():
                pct = count / total_highlights * 100 if total_highlights > 0 else 0
                print(f"    {color}: {count} ({pct:.1f}%)")
            print(f"  Avg highlight area: {stats['avg_highlight_area']:.1f} px²")
            print()

            return stats

        val_stats = analyze_split(val_annotations, "validation")
        test_stats = analyze_split(test_annotations, "test")

        overall_stats = {
            'validation': val_stats,
            'test': test_stats,
            'total_images': val_stats['total_images'] + test_stats['total_images'],
            'total_highlights': val_stats['total_highlights'] + test_stats['total_highlights']
        }

        # Save statistics
        stats_path = os.path.join(self.output_base_dir, 'dataset_statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, ensure_ascii=False, indent=2)

        print(f"✓ Statistics saved to: {stats_path}")

        return overall_stats

    def build_complete_dataset(
        self,
        num_base_images: int = 200,
        num_augmentations: int = 2,
        colors: List[str] = None
    ) -> Dict:
        """
        Build complete dataset pipeline

        Args:
            num_base_images: Number of base images
            num_augmentations: Augmentations per image
            colors: Highlight colors

        Returns:
            Dictionary with all dataset information
        """
        print(f"\n{'#'*60}")
        print(f"# BUILDING SYNTHETIC HIGHLIGHT DATASET")
        print(f"{'#'*60}\n")
        print(f"Configuration:")
        print(f"  Base images: {num_base_images}")
        print(f"  Augmentations per image: {num_augmentations}")
        print(f"  Validation ratio: {self.val_ratio*100:.0f}%")
        print(f"  Test ratio: {self.test_ratio*100:.0f}%")
        print(f"  Colors: {colors if colors else ['yellow', 'green', 'pink']}")

        # Step 1: Generate base dataset
        base_annotations = self.generate_base_dataset(num_base_images, colors)

        # Step 2: Split dataset
        val_data, test_data = self.split_dataset(base_annotations)

        # Step 3: Augment and save validation set
        val_annotations = self.augment_and_save(
            val_data,
            self.validation_dir,
            "validation",
            num_augmentations,
            use_light_augmentation=True
        )

        # Step 4: Augment and save test set
        test_annotations = self.augment_and_save(
            test_data,
            self.test_dir,
            "test",
            num_augmentations,
            use_light_augmentation=True
        )

        # Step 5: Generate statistics
        stats = self.generate_statistics(val_annotations, test_annotations)

        print(f"\n{'='*60}")
        print("✓ Dataset generation complete!")
        print(f"{'='*60}\n")
        print(f"Final dataset:")
        print(f"  Validation: {len(val_annotations)} images")
        print(f"  Test: {len(test_annotations)} images")
        print(f"  Total: {len(val_annotations) + len(test_annotations)} images")
        print(f"\nDataset saved to: {self.output_base_dir}")

        return {
            'validation_annotations': val_annotations,
            'test_annotations': test_annotations,
            'statistics': stats
        }


if __name__ == "__main__":
    # Build complete dataset
    builder = DatasetBuilder(output_base_dir="data")

    dataset_info = builder.build_complete_dataset(
        num_base_images=200,
        num_augmentations=2,
        colors=['yellow', 'green', 'pink']
    )

    print("\n✓ Dataset ready for use!")
