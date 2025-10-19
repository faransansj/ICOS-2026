#!/usr/bin/env python3
"""
Analyze OCR Failure Cases
Identify patterns in OCR errors to guide improvements
"""

import sys
import os
import cv2
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from highlight_detector import HighlightDetector
from ocr import OCREngine, OCREvaluator


def analyze_ocr_failures(num_samples: int = 50):
    """
    Analyze OCR failures to identify patterns

    Args:
        num_samples: Number of samples to analyze
    """
    print("\n" + "=" * 70)
    print("OCR FAILURE ANALYSIS")
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

    # Create detectors
    detector = HighlightDetector(
        hsv_ranges=hsv_ranges,
        kernel_size=tuple(config['kernel_size']),
        min_area=config['min_area'],
        morph_iterations=config['morph_iterations']
    )

    ocr_engine = OCREngine(lang='kor+eng', config='--psm 6 --oem 3', preprocessing=False)
    evaluator = OCREvaluator()

    # Load validation data
    with open('data/validation/validation_annotations.json', 'r') as f:
        val_data = json.load(f)

    val_data = [d for d in val_data if not d.get('is_augmented', False)][:num_samples]

    # Collect error patterns
    error_types = defaultdict(int)
    high_cer_samples = []
    error_examples = []

    for sample in val_data:
        image = cv2.imread(sample['image_path'])
        if image is None:
            continue

        detections = detector.detect(image)
        gt_texts = sample.get('annotations', [])

        # Extract OCR
        for det in detections:
            # Find matching GT
            best_match = None
            best_iou = 0

            for gt_text in gt_texts:
                iou = evaluator._calculate_bbox_iou(det['bbox'], gt_text['bbox'])
                if iou > best_iou and iou >= 0.5:
                    best_iou = iou
                    best_match = gt_text

            if best_match:
                gt_str = best_match['text']
                pred_text, conf = ocr_engine.extract_text(image, det['bbox'])

                metrics = evaluator.calculate_cer(gt_str, pred_text)

                # Categorize errors
                if metrics.cer > 0.5:
                    high_cer_samples.append({
                        'image': sample['image_name'],
                        'gt': gt_str,
                        'pred': pred_text,
                        'cer': metrics.cer,
                        'confidence': conf
                    })

                # Identify error patterns
                if metrics.insertions > 0:
                    error_types['insertions'] += metrics.insertions
                if metrics.deletions > 0:
                    error_types['deletions'] += metrics.deletions
                if metrics.substitutions > 0:
                    error_types['substitutions'] += metrics.substitutions

                # Check for specific patterns
                if ' ' in pred_text and ' ' not in gt_str:
                    error_types['extra_spaces'] += 1

                if any(c.isdigit() for c in pred_text) and not any(c.isdigit() for c in gt_str):
                    error_types['hallucinated_numbers'] += 1

                if len(pred_text) > len(gt_str) * 1.5:
                    error_types['over_segmentation'] += 1

                if len(pred_text) < len(gt_str) * 0.5:
                    error_types['under_segmentation'] += 1

                # Collect examples
                if metrics.cer > 0:
                    error_examples.append({
                        'gt': gt_str,
                        'pred': pred_text,
                        'cer': metrics.cer,
                        'ins': metrics.insertions,
                        'del': metrics.deletions,
                        'sub': metrics.substitutions
                    })

    # Analysis results
    print("Error Type Distribution:")
    print(f"  Insertions:           {error_types['insertions']}")
    print(f"  Deletions:            {error_types['deletions']}")
    print(f"  Substitutions:        {error_types['substitutions']}")
    print(f"  Extra spaces:         {error_types['extra_spaces']}")
    print(f"  Hallucinated numbers: {error_types['hallucinated_numbers']}")
    print(f"  Over-segmentation:    {error_types['over_segmentation']}")
    print(f"  Under-segmentation:   {error_types['under_segmentation']}")
    print()

    # High CER samples
    print(f"High CER Samples (CER > 0.5): {len(high_cer_samples)}")
    for i, sample in enumerate(high_cer_samples[:10]):
        print(f"\n  [{i+1}] {sample['image']}")
        print(f"      GT:   '{sample['gt']}'")
        print(f"      Pred: '{sample['pred']}'")
        print(f"      CER:  {sample['cer']:.3f} (conf: {sample['confidence']:.1f}%)")

    print("\n" + "=" * 70)
    print("IMPROVEMENT OPPORTUNITIES")
    print("=" * 70 + "\n")

    # Identify top issues
    total_errors = error_types['insertions'] + error_types['deletions'] + error_types['substitutions']

    if error_types['extra_spaces'] > 0:
        print(f"1. Extra Spaces ({error_types['extra_spaces']} cases)")
        print(f"   - Current: Korean space removal regex")
        print(f"   - Suggestion: Extend to all unnecessary spaces")

    if error_types['hallucinated_numbers'] > 0:
        print(f"\n2. Hallucinated Numbers ({error_types['hallucinated_numbers']} cases)")
        print(f"   - Issue: English chars misread as numbers")
        print(f"   - Suggestion: Confidence filtering or whitelist")

    if error_types['substitutions'] > total_errors * 0.3:
        print(f"\n3. Character Substitutions ({error_types['substitutions']} / {total_errors})")
        print(f"   - Suggestion: Try different PSM modes or preprocessing")

    if error_types['over_segmentation'] > 0:
        print(f"\n4. Over-segmentation ({error_types['over_segmentation']} cases)")
        print(f"   - Suggestion: Adjust bbox padding or use PSM 7 (single line)")

    # Save detailed analysis
    with open('outputs/ocr_failure_analysis.json', 'w') as f:
        json.dump({
            'error_types': dict(error_types),
            'high_cer_samples': high_cer_samples[:20],
            'error_examples': error_examples[:50]
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Detailed analysis saved to outputs/ocr_failure_analysis.json")


if __name__ == "__main__":
    analyze_ocr_failures(num_samples=50)
