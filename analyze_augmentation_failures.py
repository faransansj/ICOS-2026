#!/usr/bin/env python3
"""
Analyze Augmentation Failure Cases
Identify which augmentation transforms cause false positives
"""

import sys
import os
import cv2
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from highlight_detector import HighlightDetector


def load_config():
    """Load optimized configuration"""
    with open('configs/optimized_hsv_ranges.json', 'r') as f:
        return json.load(f)


def analyze_failure_cases():
    """Analyze images with excessive false positives"""

    print("\n" + "=" * 70)
    print("ANALYZING AUGMENTATION FAILURE CASES")
    print("=" * 70 + "\n")

    # Load configuration
    config = load_config()
    hsv_ranges = {
        color: {
            'lower': np.array(ranges['lower']),
            'upper': np.array(ranges['upper'])
        }
        for color, ranges in config['hsv_ranges'].items()
    }

    detector = HighlightDetector(
        hsv_ranges=hsv_ranges,
        kernel_size=tuple(config['kernel_size']),
        min_area=config['min_area'],
        morph_iterations=config['morph_iterations']
    )

    # Load validation data
    with open('data/validation/validation_annotations.json', 'r') as f:
        val_data = json.load(f)

    # Known problematic cases from test output
    problem_images = [
        'validation_0004_aug0.png',  # 296 detections (GT: 4)
        'validation_0005_aug1.png',  # 321 detections
        'validation_0011_aug1.png',  # 77 detections
        'validation_0038_aug1.png',  # 423 detections
        'validation_0044_aug1.png',  # 322 detections
        'validation_0047_aug1.png',  # 440 detections
        'validation_0050_aug1.png',  # 201 detections
        'validation_0070_aug0.png',  # 496 detections (worst case)
        'validation_0073_aug0.png',  # 361 detections
        'validation_0091_aug0.png',  # 426 detections
    ]

    results = []

    for img_name in problem_images:
        # Find sample
        sample = next((s for s in val_data if s['image_name'] == img_name), None)
        if not sample:
            continue

        print(f"\n{'='*70}")
        print(f"Analyzing: {img_name}")
        print(f"{'='*70}")

        # Load image
        image = cv2.imread(sample['image_path'])
        if image is None:
            print("  ⚠ Could not load image")
            continue

        # Detect
        detections = detector.detect(image)
        ground_truths = sample['highlight_annotations']

        print(f"Detected: {len(detections)} highlights")
        print(f"Ground truth: {len(ground_truths)} highlights")
        print(f"False positives: ~{len(detections) - len(ground_truths)}")

        # Analyze color distribution of false positives
        color_counts = {}
        for det in detections:
            color = det['color']
            color_counts[color] = color_counts.get(color, 0) + 1

        print("\nColor distribution:")
        for color, count in sorted(color_counts.items(), key=lambda x: -x[1]):
            print(f"  {color}: {count} ({count/len(detections)*100:.1f}%)")

        # Analyze HSV statistics of detected regions
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        print("\nSample false positive HSV values:")
        for i, det in enumerate(detections[:5]):  # First 5 detections
            bbox = det['bbox']
            x, y, w, h = bbox
            region = hsv[y:y+h, x:x+w]

            h_mean = np.mean(region[:, :, 0])
            s_mean = np.mean(region[:, :, 1])
            v_mean = np.mean(region[:, :, 2])

            print(f"  {det['color']}: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")

        # Check if it's aug0 or aug1 to identify transform
        if 'aug0' in img_name:
            aug_type = "RandomBrightnessContrast"
        elif 'aug1' in img_name:
            aug_type = "HueSaturationValue + ImageCompression"
        else:
            aug_type = "Original"

        print(f"\nAugmentation type: {aug_type}")

        results.append({
            'image': img_name,
            'augmentation': aug_type,
            'detections': len(detections),
            'ground_truth': len(ground_truths),
            'false_positives': len(detections) - len(ground_truths),
            'color_distribution': color_counts
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70 + "\n")

    aug0_failures = [r for r in results if 'aug0' in r['image']]
    aug1_failures = [r for r in results if 'aug1' in r['image']]

    if aug0_failures:
        avg_fp_aug0 = np.mean([r['false_positives'] for r in aug0_failures])
        print(f"RandomBrightnessContrast (aug0): {len(aug0_failures)} failures")
        print(f"  Average false positives: {avg_fp_aug0:.0f}")

    if aug1_failures:
        avg_fp_aug1 = np.mean([r['false_positives'] for r in aug1_failures])
        print(f"HueSaturationValue + Compression (aug1): {len(aug1_failures)} failures")
        print(f"  Average false positives: {avg_fp_aug1:.0f}")

    # Dominant problematic color
    all_colors = {}
    for r in results:
        for color, count in r['color_distribution'].items():
            all_colors[color] = all_colors.get(color, 0) + count

    print("\nTotal false positives by color:")
    for color, count in sorted(all_colors.items(), key=lambda x: -x[1]):
        print(f"  {color}: {count}")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70 + "\n")

    dominant_color = max(all_colors.items(), key=lambda x: x[1])[0]
    print(f"1. {dominant_color} HSV range is too broad")
    print("2. Consider tightening Saturation/Value lower bounds")
    print("3. Increase min_area threshold")
    print("4. Add variance-based filtering for uniform regions")
    print()


if __name__ == "__main__":
    analyze_failure_cases()
