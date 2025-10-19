"""
Data Augmentation Module
Apply realistic augmentations to synthetic highlight images
"""

import os
import json
import random
from typing import List, Dict, Tuple
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm


class DataAugmentation:
    """Apply data augmentation with bounding box preservation"""

    def __init__(self, preserve_bbox: bool = True):
        """
        Initialize DataAugmentation

        Args:
            preserve_bbox: Whether to preserve bounding boxes during augmentation
        """
        self.preserve_bbox = preserve_bbox

        # Define augmentation pipeline
        self.transform = self._create_transform()

    def _create_transform(self) -> A.Compose:
        """Create Albumentations transform pipeline"""
        bbox_params = None
        if self.preserve_bbox:
            bbox_params = A.BboxParams(
                format='coco',  # [x, y, width, height]
                label_fields=['class_labels'],
                min_visibility=0.3
            )

        transform = A.Compose([
            # Brightness and contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),

            # Gaussian noise
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=0.3
            ),

            # Slight rotation (document scan)
            A.Rotate(
                limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
                p=0.5
            ),

            # Perspective transform (camera angle)
            A.Perspective(
                scale=(0.02, 0.05),
                p=0.3
            ),

            # Motion blur (camera shake)
            A.MotionBlur(
                blur_limit=5,
                p=0.2
            ),

            # JPEG compression artifacts
            A.ImageCompression(
                quality_lower=75,
                quality_upper=95,
                p=0.3
            ),

            # Color shift (different lighting)
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            ),

            # Shadow effect
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.2
            ),

        ], bbox_params=bbox_params)

        return transform

    def augment_image(
        self,
        image: np.ndarray,
        bboxes: List[List[int]] = None,
        labels: List[str] = None
    ) -> Tuple[np.ndarray, List[List[int]], List[str]]:
        """
        Apply augmentation to single image

        Args:
            image: Input image (numpy array)
            bboxes: List of bounding boxes [x, y, w, h]
            labels: List of labels corresponding to bboxes

        Returns:
            Tuple of (augmented_image, augmented_bboxes, labels)
        """
        if bboxes is None or not self.preserve_bbox:
            # No bboxes to preserve
            transformed = self.transform(image=image)
            return transformed['image'], bboxes, labels

        # Apply transform with bbox preservation
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=labels
        )

        return (
            transformed['image'],
            transformed['bboxes'],
            transformed['class_labels']
        )

    def augment_dataset(
        self,
        input_dir: str,
        output_dir: str,
        annotations_file: str,
        num_augmentations: int = 2
    ) -> Dict:
        """
        Augment entire dataset

        Args:
            input_dir: Directory with original images
            output_dir: Directory to save augmented images
            annotations_file: Path to annotations JSON
            num_augmentations: Number of augmentations per image

        Returns:
            Dictionary with augmented annotations
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load annotations
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        augmented_annotations = []
        augmented_id = 0

        for original_data in tqdm(annotations, desc="Augmenting dataset"):
            image_path = original_data['image_path']

            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read {image_path}")
                continue

            # Convert annotations to COCO format for Albumentations
            highlight_annots = original_data.get('highlight_annotations', [])
            bboxes = [annot['bbox'] for annot in highlight_annots]
            labels = [annot['color'] for annot in highlight_annots]

            # Generate multiple augmentations
            for aug_idx in range(num_augmentations):
                # Apply augmentation
                aug_image, aug_bboxes, aug_labels = self.augment_image(
                    image, bboxes, labels
                )

                # Save augmented image
                aug_image_name = f"aug_{original_data['image_id']:04d}_{aug_idx}.png"
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

                augmented_annotations.append({
                    'image_id': augmented_id,
                    'image_name': aug_image_name,
                    'image_path': aug_image_path,
                    'original_id': original_data['image_id'],
                    'augmentation_index': aug_idx,
                    'annotations': original_data['annotations'],
                    'highlight_annotations': aug_highlight_annots
                })

                augmented_id += 1

        # Save augmented annotations
        aug_annotations_path = os.path.join(output_dir, 'augmented_annotations.json')
        with open(aug_annotations_path, 'w', encoding='utf-8') as f:
            json.dump(augmented_annotations, f, ensure_ascii=False, indent=2)

        print(f"\nAugmentation complete:")
        print(f"  Original images: {len(annotations)}")
        print(f"  Augmented images: {len(augmented_annotations)}")
        print(f"  Total: {len(annotations) + len(augmented_annotations)}")
        print(f"  Saved to: {output_dir}")

        return {
            'original_count': len(annotations),
            'augmented_count': len(augmented_annotations),
            'annotations': augmented_annotations
        }


class LightAugmentation:
    """Lighter augmentation for validation/test data"""

    def __init__(self):
        """Initialize light augmentation"""
        self.transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
            A.Rotate(limit=2, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.3),
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['class_labels'],
            min_visibility=0.5
        ))

    def augment_image(
        self,
        image: np.ndarray,
        bboxes: List[List[int]] = None,
        labels: List[str] = None
    ) -> Tuple[np.ndarray, List[List[int]], List[str]]:
        """Apply light augmentation"""
        if bboxes is None:
            transformed = self.transform(image=image)
            return transformed['image'], bboxes, labels

        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=labels
        )

        return (
            transformed['image'],
            transformed['bboxes'],
            transformed['class_labels']
        )


if __name__ == "__main__":
    # Test augmentation
    augmentor = DataAugmentation()

    # Load test image
    test_image = cv2.imread("test_highlighted.png")
    if test_image is not None:
        # Apply augmentation
        aug_img, _, _ = augmentor.augment_image(test_image)
        cv2.imwrite("test_augmented.png", aug_img)
        print("Augmentation test complete")
