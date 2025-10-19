#!/usr/bin/env python3
"""
Analyze Actual Highlight Colors
Measure real HSV values from synthesized highlights
"""

import sys
import os
import cv2
import numpy as np
import json
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def analyze_highlight_hsv(
    annotations_path: str,
    num_samples: int = 50,
    use_orig_only: bool = True
) -> Dict:
    """
    Analyze actual HSV values from highlight regions

    Args:
        annotations_path: Path to annotations JSON
        num_samples: Number of samples to analyze
        use_orig_only: Only use original (non-augmented) images

    Returns:
        Dictionary with color statistics
    """
    print("\n" + "=" * 60)
    print("ANALYZING ACTUAL HIGHLIGHT HSV VALUES")
    print("=" * 60 + "\n")

    # Load annotations
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    # Filter for original images only
    if use_orig_only:
        data = [d for d in data if not d.get('is_augmented', False)]
        print(f"Using {len(data)} original images (augmented excluded)")

    # Limit samples
    data = data[:num_samples]
    print(f"Analyzing {len(data)} samples\n")

    # Collect HSV values by color
    color_hsv_values = {
        'yellow': [],
        'green': [],
        'pink': []
    }

    for sample in data:
        image_path = sample['image_path']
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Could not load {image_path}")
            continue

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Process each highlight
        for highlight in sample.get('highlight_annotations', []):
            color = highlight['color']
            bbox = highlight['bbox']
            x, y, w, h = bbox

            # Extract highlight region
            region_hsv = hsv[y:y+h, x:x+w]

            # Collect all pixel values
            pixels = region_hsv.reshape(-1, 3)
            color_hsv_values[color].extend(pixels.tolist())

    # Calculate statistics
    results = {}

    for color, values in color_hsv_values.items():
        if not values:
            print(f"Warning: No samples for {color}")
            continue

        values_array = np.array(values)

        # Calculate percentiles for robust range
        h_values = values_array[:, 0]
        s_values = values_array[:, 1]
        v_values = values_array[:, 2]

        stats = {
            'num_samples': len(values),
            'h_mean': float(np.mean(h_values)),
            'h_std': float(np.std(h_values)),
            'h_min': float(np.min(h_values)),
            'h_max': float(np.max(h_values)),
            'h_p5': float(np.percentile(h_values, 5)),
            'h_p95': float(np.percentile(h_values, 95)),
            's_mean': float(np.mean(s_values)),
            's_std': float(np.std(s_values)),
            's_min': float(np.min(s_values)),
            's_max': float(np.max(s_values)),
            's_p5': float(np.percentile(s_values, 5)),
            's_p95': float(np.percentile(s_values, 95)),
            'v_mean': float(np.mean(v_values)),
            'v_std': float(np.std(v_values)),
            'v_min': float(np.min(v_values)),
            'v_max': float(np.max(v_values)),
            'v_p5': float(np.percentile(v_values, 5)),
            'v_p95': float(np.percentile(v_values, 95))
        }

        # Propose range (using 5th and 95th percentiles for robustness)
        proposed_range = {
            'lower': [
                max(0, int(stats['h_p5'] - 5)),
                max(0, int(stats['s_p5'] - 20)),
                max(0, int(stats['v_p5'] - 20))
            ],
            'upper': [
                min(180, int(stats['h_p95'] + 5)),
                255,
                255
            ]
        }

        stats['proposed_range'] = proposed_range

        results[color] = stats

        # Print results
        print(f"{color.upper()} Statistics:")
        print(f"  Samples: {stats['num_samples']:,} pixels")
        print(f"  Hue:        {stats['h_mean']:.1f} ± {stats['h_std']:.1f} (range: {stats['h_min']:.0f}-{stats['h_max']:.0f})")
        print(f"  Saturation: {stats['s_mean']:.1f} ± {stats['s_std']:.1f} (range: {stats['s_min']:.0f}-{stats['s_max']:.0f})")
        print(f"  Value:      {stats['v_mean']:.1f} ± {stats['v_std']:.1f} (range: {stats['v_min']:.0f}-{stats['v_max']:.0f})")
        print(f"  Proposed range:")
        print(f"    Lower: {proposed_range['lower']}")
        print(f"    Upper: {proposed_range['upper']}")
        print()

    # Save results
    output_path = 'outputs/hsv_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Analysis saved to: {output_path}\n")

    return results


def generate_optimized_config(analysis_results: Dict) -> Dict:
    """
    Generate optimized detector configuration

    Args:
        analysis_results: Results from HSV analysis

    Returns:
        Optimized configuration dictionary
    """
    config = {
        'hsv_ranges': {},
        'kernel_size': [5, 5],
        'min_area': 150,  # Increased from 100 to reduce noise
        'morph_iterations': 1
    }

    for color, stats in analysis_results.items():
        config['hsv_ranges'][color] = stats['proposed_range']

    # Save configuration
    os.makedirs('configs', exist_ok=True)
    config_path = 'configs/optimized_hsv_ranges.json'

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("OPTIMIZED CONFIGURATION")
    print("=" * 60 + "\n")

    for color, ranges in config['hsv_ranges'].items():
        print(f"{color.capitalize()}:")
        print(f"  Lower: {ranges['lower']}")
        print(f"  Upper: {ranges['upper']}")

    print(f"\nmin_area: {config['min_area']}")
    print(f"kernel_size: {config['kernel_size']}")
    print(f"\n✓ Configuration saved to: {config_path}\n")

    return config


if __name__ == "__main__":
    # Analyze validation data
    results = analyze_highlight_hsv(
        annotations_path='data/validation/validation_annotations.json',
        num_samples=50,
        use_orig_only=True
    )

    # Generate optimized configuration
    config = generate_optimized_config(results)

    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60 + "\n")
    print("1. Review the proposed HSV ranges above")
    print("2. Run test with optimized configuration:")
    print("   uv run python test_optimized_detection.py")
    print("3. If mIoU > 0.75, proceed to full dataset evaluation")
    print()
