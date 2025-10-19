#!/usr/bin/env python3
"""
Test Optimized Highlight Detection
Test with optimized HSV ranges from analysis
"""

import sys
import os
import cv2
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from highlight_detector import HighlightDetector, HighlightEvaluator


def load_optimized_config(config_path: str = 'configs/optimized_hsv_ranges.json'):
    """Load optimized configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)


def test_optimized_detection(use_orig_only: bool = True, num_samples: int = 50):
    """Test with optimized configuration"""
    print("\n" + "=" * 60)
    print("TESTING OPTIMIZED HIGHLIGHT DETECTION")
    print("=" * 60 + "\n")

    # Load optimized configuration
    config = load_optimized_config()
    print("Loaded optimized configuration:")
    for color, ranges in config['hsv_ranges'].items():
        print(f"  {color.capitalize()}: {ranges['lower']} -> {ranges['upper']}")
    print(f"  min_area: {config['min_area']}")
    print()

    # Create detector with optimized parameters
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

    evaluator = HighlightEvaluator(iou_threshold=0.5)

    # Load validation data
    with open('data/validation/validation_annotations.json', 'r') as f:
        val_data = json.load(f)

    # Filter for original images
    if use_orig_only:
        val_data = [d for d in val_data if not d.get('is_augmented', False)]
        print(f"Using {len(val_data)} original images (augmented excluded)")

    val_data = val_data[:num_samples]
    print(f"Testing on {len(val_data)} samples...\n")

    # Test
    all_preds = []
    all_gts = []

    for i, sample in enumerate(val_data):
        print(f"Image {i+1}/{len(val_data)}: {sample['image_name']}")

        # Load image
        image = cv2.imread(sample['image_path'])
        if image is None:
            print(f"  ⚠ Could not load image")
            continue

        # Detect highlights
        detections = detector.detect(image)
        ground_truths = sample['highlight_annotations']

        print(f"  Detected: {len(detections)} highlights")
        print(f"  Ground truth: {len(ground_truths)} highlights")

        # Evaluate
        metrics = evaluator.evaluate_single_image(detections, ground_truths)
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1: {metrics['f1_score']:.3f}")
        print(f"  mIoU: {metrics['mean_iou']:.3f}")

        all_preds.append(detections)
        all_gts.append(ground_truths)

        # Visualize first few images
        if i < 3:
            vis = detector.visualize_detections(image, detections)
            cv2.imwrite(f'outputs/optimized_detection_{i}.png', vis)
            print(f"  ✓ Saved visualization")

        print()

    # Overall metrics
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60 + "\n")

    overall_metrics = evaluator.evaluate_dataset(all_preds, all_gts)

    print("Overall Performance:")
    print(f"  mIoU:      {overall_metrics['overall']['mIoU']:.4f}")
    print(f"  Precision: {overall_metrics['overall']['precision']:.4f}")
    print(f"  Recall:    {overall_metrics['overall']['recall']:.4f}")
    print(f"  F1-Score:  {overall_metrics['overall']['f1_score']:.4f}")
    print()

    print("Detection Statistics:")
    print(f"  True Positives:  {overall_metrics['overall']['total_tp']}")
    print(f"  False Positives: {overall_metrics['overall']['total_fp']}")
    print(f"  False Negatives: {overall_metrics['overall']['total_fn']}")
    print()

    # Per-color evaluation
    print("Per-Color Performance:")
    color_metrics = evaluator.evaluate_by_color(all_preds, all_gts)
    for color, metrics in color_metrics.items():
        print(f"  {color.capitalize()}:")
        print(f"    mIoU:      {metrics['mIoU']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-Score:  {metrics['f1_score']:.4f}")

    print("\n" + "=" * 60)

    # Save metrics
    with open('outputs/optimized_test_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    print("\n✓ Metrics saved to outputs/optimized_test_metrics.json")

    # Check target
    miou = overall_metrics['overall']['mIoU']
    target_miou = 0.75

    print(f"\nTarget mIoU: {target_miou:.2f}")
    print(f"Current mIoU: {miou:.4f}")

    if miou >= target_miou:
        print("✅ TARGET ACHIEVED!")
        print("\nNext steps:")
        print("  1. Test on augmented images")
        print("  2. Test on full validation set (180 images)")
        print("  3. Proceed to OCR integration")
    else:
        gap = target_miou - miou
        print(f"❌ Target not met (gap: {gap:.4f})")
        print("\nNext steps:")
        print("  1. Analyze failure cases")
        print("  2. Fine-tune HSV ranges manually")
        print("  3. Adjust min_area or kernel_size")

    return overall_metrics


if __name__ == "__main__":
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)

    # Run test with optimized configuration
    metrics = test_optimized_detection(
        use_orig_only=True,
        num_samples=50
    )
