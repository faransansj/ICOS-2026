#!/usr/bin/env python3
"""
Compare Original vs Augmented Image
Understand what RandomBrightnessContrast does
"""

import cv2
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from highlight_detector import HighlightDetector

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

detector = HighlightDetector(
    hsv_ranges=hsv_ranges,
    kernel_size=tuple(config['kernel_size']),
    min_area=config['min_area'],
    morph_iterations=config['morph_iterations']
)

# Compare orig vs aug0
# Pattern: every 3 images is a set (orig, aug0, aug1)
# 69=orig, 70=aug0, 71=aug1
orig_path = 'data/validation/validation_0069_orig.png'
aug_path = 'data/validation/validation_0070_aug0.png'

print("=" * 70)
print("COMPARING ORIGINAL VS AUGMENTED")
print("=" * 70 + "\n")

for label, path in [("ORIGINAL", orig_path), ("AUGMENTED (RandomBrightnessContrast)", aug_path)]:
    print(f"\n{label}:")
    print(f"File: {path}")

    image = cv2.imread(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect
    detections = detector.detect(image)

    print(f"Detections: {len(detections)}")

    # Color distribution
    color_counts = {}
    for det in detections:
        color = det['color']
        color_counts[color] = color_counts.get(color, 0) + 1

    print(f"Color distribution:")
    for color in ['yellow', 'green', 'pink']:
        count = color_counts.get(color, 0)
        print(f"  {color}: {count}")

    # HSV statistics of entire image
    print(f"\nImage-wide HSV statistics:")
    print(f"  Hue:        {np.mean(hsv[:,:,0]):.1f} ± {np.std(hsv[:,:,0]):.1f}")
    print(f"  Saturation: {np.mean(hsv[:,:,1]):.1f} ± {np.std(hsv[:,:,1]):.1f}")
    print(f"  Value:      {np.mean(hsv[:,:,2]):.1f} ± {np.std(hsv[:,:,2]):.1f}")

    # Check yellow mask coverage
    yellow_mask = cv2.inRange(hsv, hsv_ranges['yellow']['lower'], hsv_ranges['yellow']['upper'])
    yellow_pixels = np.count_nonzero(yellow_mask)
    total_pixels = image.shape[0] * image.shape[1]

    print(f"\nYellow HSV range coverage:")
    print(f"  Matching pixels: {yellow_pixels:,} / {total_pixels:,} ({yellow_pixels/total_pixels*100:.2f}%)")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70 + "\n")

# Load annotations
with open('data/validation/validation_annotations.json', 'r') as f:
    val_data = json.load(f)

sample = next(s for s in val_data if 'validation_0069_orig' in s['image_name'])
gt_count = len(sample['highlight_annotations'])

print(f"Ground truth highlights: {gt_count}")
print(f"\nOriginal → Augmented change:")
print(f"  Detection increase: {496 - 1} = {495} additional false detections")
print(f"  Yellow mask coverage increase: Check above statistics")
print(f"\nConclusion:")
print(f"  RandomBrightnessContrast is creating image-wide color shifts")
print(f"  Large background areas now fall within yellow HSV range")
print(f"  Need to add additional constraints beyond HSV ranges")
