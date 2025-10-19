#!/usr/bin/env python3
"""
Debug OCR by Visualizing Highlight Regions
Extract and save highlight regions for manual inspection
"""

import sys
import os
import cv2
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from highlight_detector import HighlightDetector
from ocr import OCREngine


def debug_ocr_regions():
    """Extract and visualize highlight regions for OCR debugging"""

    print("\n" + "=" * 70)
    print("OCR REGION DEBUGGING")
    print("=" * 70 + "\n")

    # Load config
    with open('configs/optimized_hsv_ranges.json', 'r') as f:
        config = json.load(f)

    hsv_ranges = {
        color: {
            'lower': np.array(ranges['lower']),
            'upper': np.array(ranges['upper'])
        }
        for color, ranges in config['hsv_ranges'].items()
    }

    # Create detector
    detector = HighlightDetector(
        hsv_ranges=hsv_ranges,
        kernel_size=tuple(config['kernel_size']),
        min_area=config['min_area'],
        morph_iterations=config['morph_iterations']
    )

    # Create OCR engine
    ocr = OCREngine(lang='kor+eng', config='--psm 7 --oem 3', preprocessing=True)

    # Load validation data
    with open('data/validation/validation_annotations.json', 'r') as f:
        val_data = json.load(f)

    # Find sample with "OpenCV는"
    target_samples = [s for s in val_data if
                     any('OpenCV' in annot.get('text', '') for annot in s.get('annotations', []))]

    if not target_samples:
        print("No samples with 'OpenCV' found")
        return

    sample = target_samples[0]
    print(f"Using sample: {sample['image_name']}\n")

    # Load image
    image = cv2.imread(sample['image_path'])

    # Detect highlights
    detections = detector.detect(image)
    print(f"Detected {len(detections)} highlights\n")

    # Create output directory
    os.makedirs('outputs/ocr_debug', exist_ok=True)

    # Process each detection
    for i, det in enumerate(detections):
        bbox = det['bbox']
        color = det['color']
        x, y, w, h = bbox

        print(f"\nRegion {i+1}: {color} at {bbox}")

        # Extract region with padding
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)

        region = image[y_start:y_end, x_start:x_end]

        # Save original region
        cv2.imwrite(f'outputs/ocr_debug/region_{i}_original.png', region)
        print(f"  Saved: region_{i}_original.png")

        # Save preprocessed region
        preprocessed = ocr.preprocess_image(region)
        cv2.imwrite(f'outputs/ocr_debug/region_{i}_preprocessed.png', preprocessed)
        print(f"  Saved: region_{i}_preprocessed.png")

        # OCR on original (no preprocessing)
        text_orig, conf_orig = ocr.extract_text(region, bbox=None)
        print(f"  OCR (original): '{text_orig}' (conf: {conf_orig:.1f}%)")

        # OCR on preprocessed
        text_prep, conf_prep = ocr.extract_text(preprocessed, bbox=None)
        print(f"  OCR (preprocessed): '{text_prep}' (conf: {conf_prep:.1f}%)")

        # Try different PSM modes
        for psm in [3, 6, 7, 8]:
            ocr_test = OCREngine(lang='kor+eng', config=f'--psm {psm} --oem 3', preprocessing=False)
            text_test, conf_test = ocr_test.extract_text(region, bbox=None)
            print(f"  OCR (PSM {psm}): '{text_test}' (conf: {conf_test:.1f}%)")

    print(f"\n✓ Debug images saved to outputs/ocr_debug/")


if __name__ == "__main__":
    debug_ocr_regions()
