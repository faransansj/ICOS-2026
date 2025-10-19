#!/usr/bin/env python3
"""
Test Highlight Detection
Quick test of highlight detection on validation samples
"""

import sys
import os
import cv2
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from highlight_detector import HighlightDetector, HighlightEvaluator


def test_detection():
    """Test highlight detection on validation data"""
    print("\n" + "=" * 60)
    print("HIGHLIGHT DETECTION TEST")
    print("=" * 60 + "\n")

    # Load validation data
    with open('data/validation/validation_annotations.json', 'r') as f:
        val_data = json.load(f)

    print(f"Loaded {len(val_data)} validation images\n")

    # Create detector and evaluator
    detector = HighlightDetector()
    evaluator = HighlightEvaluator(iou_threshold=0.5)

    # Test on first 10 images
    num_samples = min(10, len(val_data))
    print(f"Testing on {num_samples} samples...\n")

    all_preds = []
    all_gts = []

    for i, sample in enumerate(val_data[:num_samples]):
        print(f"Image {i+1}/{num_samples}: {sample['image_name']}")

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

        # Show detections
        for j, det in enumerate(detections):
            print(f"    - {det['color']}: {det['bbox']}, conf={det['confidence']:.2f}")

        print()

        all_preds.append(detections)
        all_gts.append(ground_truths)

        # Visualize first image
        if i == 0:
            vis = detector.visualize_detections(image, detections)
            cv2.imwrite('outputs/test_detection_sample.png', vis)
            print("  ✓ Saved visualization to outputs/test_detection_sample.png\n")

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
    with open('outputs/initial_test_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    print("\n✓ Metrics saved to outputs/initial_test_metrics.json")

    return overall_metrics


if __name__ == "__main__":
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)

    # Run test
    metrics = test_detection()

    # Check if we meet target
    miou = metrics['overall']['mIoU']
    target_miou = 0.75

    print(f"\nTarget mIoU: {target_miou:.2f}")
    print(f"Current mIoU: {miou:.4f}")

    if miou >= target_miou:
        print("✅ Target achieved!")
    else:
        gap = target_miou - miou
        print(f"❌ Target not met (gap: {gap:.4f})")
        print("\nNext steps:")
        print("  1. Run parameter optimization: python src/highlight_detector/optimizer.py")
        print("  2. Adjust HSV ranges manually")
        print("  3. Tune morphology parameters")
