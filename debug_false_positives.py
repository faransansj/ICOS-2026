#!/usr/bin/env python3
"""
Debug False Positive Detections
Visualize what regions are being incorrectly detected
"""

import cv2
import numpy as np
import json

# Load config
with open('configs/optimized_hsv_ranges.json', 'r') as f:
    config = json.load(f)

# Load worst-case image
img_path = 'data/validation/validation_0070_aug0.png'
image = cv2.imread(img_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print(f"Image shape: {image.shape}")
print(f"HSV shape: {hsv.shape}")

# Check yellow range
yellow_lower = np.array(config['hsv_ranges']['yellow']['lower'])
yellow_upper = np.array(config['hsv_ranges']['yellow']['upper'])

print(f"\nConfigured Yellow range:")
print(f"  Lower: {yellow_lower} (H={yellow_lower[0]})")
print(f"  Upper: {yellow_upper} (H={yellow_upper[0]})")

# Create mask
yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

# Count matching pixels
matching_pixels = np.count_nonzero(yellow_mask)
total_pixels = image.shape[0] * image.shape[1]

print(f"\nMatching pixels: {matching_pixels:,} / {total_pixels:,} ({matching_pixels/total_pixels*100:.2f}%)")

# Sample some matching pixels
y_coords, x_coords = np.where(yellow_mask > 0)
num_samples = min(100, len(y_coords))
sample_indices = np.random.choice(len(y_coords), num_samples, replace=False)

h_values = []
s_values = []
v_values = []

for idx in sample_indices:
    y, x = y_coords[idx], x_coords[idx]
    h, s, v = hsv[y, x]
    h_values.append(h)
    s_values.append(s)
    v_values.append(v)

print(f"\nSample HSV values from 'yellow' detections:")
print(f"  Hue:        {np.mean(h_values):.1f} ± {np.std(h_values):.1f} (range: {np.min(h_values)}-{np.max(h_values)})")
print(f"  Saturation: {np.mean(s_values):.1f} ± {np.std(s_values):.1f} (range: {np.min(s_values)}-{np.max(s_values)})")
print(f"  Value:      {np.mean(v_values):.1f} ± {np.std(v_values):.1f} (range: {np.min(v_values)}-{np.max(v_values)})")

# Check if hue values are OUTSIDE expected range
outside_range = [h for h in h_values if h < 25 or h > 35]
print(f"\nHue values OUTSIDE yellow range (25-35): {len(outside_range)} / {num_samples}")
if outside_range:
    print(f"  Out-of-range hues: {sorted(set(outside_range))}")

# Visualize
output = image.copy()
output[yellow_mask > 0] = [0, 255, 255]  # Highlight in cyan

cv2.imwrite('outputs/debug_yellow_detections.png', output)
print(f"\n✓ Visualization saved to outputs/debug_yellow_detections.png")

# Also save the raw mask
cv2.imwrite('outputs/debug_yellow_mask.png', yellow_mask)
print(f"✓ Mask saved to outputs/debug_yellow_mask.png")
