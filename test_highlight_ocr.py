#!/usr/bin/env python3
"""
Test Integrated Highlight Detection + OCR Pipeline
End-to-end test with CER evaluation
"""

import sys
import os
import cv2
import json
import numpy as np
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from highlight_detector import HighlightDetector, HighlightEvaluator
from ocr import OCREngine, OCREvaluator


def load_optimized_config():
    """Load optimized highlight detection configuration"""
    with open('configs/optimized_hsv_ranges.json', 'r') as f:
        return json.load(f)


def test_integrated_pipeline(num_samples: int = 20):
    """
    Test integrated highlight detection + OCR pipeline

    Args:
        num_samples: Number of validation samples to test
    """
    print("\n" + "=" * 70)
    print("INTEGRATED HIGHLIGHT DETECTION + OCR TEST")
    print("=" * 70 + "\n")

    # Load configurations
    config = load_optimized_config()

    # Create detectors
    hsv_ranges = {
        color: {
            'lower': np.array(ranges['lower']),
            'upper': np.array(ranges['upper'])
        }
        for color, ranges in config['hsv_ranges'].items()
    }

    highlight_detector = HighlightDetector(
        hsv_ranges=hsv_ranges,
        kernel_size=tuple(config['kernel_size']),
        min_area=config['min_area'],
        morph_iterations=config['morph_iterations']
    )

    highlight_evaluator = HighlightEvaluator(iou_threshold=0.5)

    # Create OCR engine with PSM 7 (single line) for highlight regions
    ocr_engine = OCREngine(
        lang='kor+eng',
        config='--psm 7 --oem 3',
        preprocessing=False,
        min_confidence=60.0,
        use_multi_psm=True
    )

    ocr_evaluator = OCREvaluator()

    print("Loaded components:")
    print(f"  Highlight Detector: ✓")
    print(f"  OCR Engine: ✓ (Tesseract {ocr_engine.test_installation()['version']})")
    print()

    # Load validation data
    with open('data/validation/validation_annotations.json', 'r') as f:
        val_data = json.load(f)

    # Use only original images
    val_data = [d for d in val_data if not d.get('is_augmented', False)]
    print(f"Using {len(val_data)} original images")

    # Limit samples
    val_data = val_data[:num_samples]
    print(f"Testing on {len(val_data)} samples...\n")

    # Collect metrics
    highlight_metrics_list = []
    ocr_metrics_list = []
    all_highlight_preds = []
    all_highlight_gts = []

    for i, sample in enumerate(val_data):
        print(f"\n{'='*70}")
        print(f"Sample {i+1}/{len(val_data)}: {sample['image_name']}")
        print(f"{'='*70}")

        # Load image
        image = cv2.imread(sample['image_path'])
        if image is None:
            print("  ⚠ Could not load image")
            continue

        # Ground truth
        gt_highlights = sample['highlight_annotations']
        gt_texts = sample.get('annotations', [])

        print(f"\nGround Truth:")
        print(f"  Highlights: {len(gt_highlights)}")
        print(f"  Text regions: {len(gt_texts)}")

        # Step 1: Detect highlights
        detections = highlight_detector.detect(image)
        print(f"\nStep 1 - Highlight Detection:")
        print(f"  Detected: {len(detections)} highlights")

        # Evaluate highlight detection
        h_metrics = highlight_evaluator.evaluate_single_image(detections, gt_highlights)
        print(f"  Precision: {h_metrics['precision']:.3f}")
        print(f"  Recall: {h_metrics['recall']:.3f}")
        print(f"  mIoU: {h_metrics['mean_iou']:.3f}")

        highlight_metrics_list.append(h_metrics)
        all_highlight_preds.append(detections)
        all_highlight_gts.append(gt_highlights)

        # Step 2: OCR on detected highlights
        print(f"\nStep 2 - OCR Extraction:")

        ocr_results = ocr_engine.extract_from_detections(image, detections)

        # Display OCR results
        for j, result in enumerate(ocr_results[:5]):  # Show first 5
            print(f"  [{j+1}] {result.color}: '{result.text}' (conf: {result.confidence:.1f}%)")

        # Step 3: Evaluate OCR
        print(f"\nStep 3 - OCR Evaluation:")

        # Match OCR results to ground truth texts
        total_chars = 0
        total_errors = 0
        region_cer_values = []

        for result in ocr_results:
            # Find matching ground truth based on bbox IoU
            best_match = None
            best_iou = 0

            for gt_text in gt_texts:
                iou = ocr_evaluator._calculate_bbox_iou(result.bbox, gt_text['bbox'])
                if iou > best_iou and iou >= 0.5:
                    best_iou = iou
                    best_match = gt_text

            if best_match:
                gt_text_str = best_match['text']
                pred_text_str = result.text

                metrics = ocr_evaluator.calculate_cer(gt_text_str, pred_text_str)

                total_chars += metrics.total_chars
                total_errors += (metrics.insertions + metrics.deletions + metrics.substitutions)
                region_cer_values.append(metrics.cer)

                print(f"  Region: '{gt_text_str}' → '{pred_text_str}' (CER: {metrics.cer:.3f})")

        # Calculate sample CER
        if total_chars > 0:
            sample_cer = total_errors / total_chars
        else:
            sample_cer = 0.0

        print(f"\nSample CER: {sample_cer:.4f} ({total_errors}/{total_chars} errors)")

        ocr_metrics_list.append({
            'sample_cer': sample_cer,
            'total_chars': total_chars,
            'total_errors': total_errors,
            'num_regions': len(region_cer_values)
        })

    # Overall Results
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70 + "\n")

    # Highlight Detection Metrics
    print("Highlight Detection Performance:")
    overall_h_metrics = highlight_evaluator.evaluate_dataset(
        all_highlight_preds,
        all_highlight_gts
    )

    print(f"  mIoU:      {overall_h_metrics['overall']['mIoU']:.4f}")
    print(f"  Precision: {overall_h_metrics['overall']['precision']:.4f}")
    print(f"  Recall:    {overall_h_metrics['overall']['recall']:.4f}")
    print(f"  F1-Score:  {overall_h_metrics['overall']['f1_score']:.4f}")
    print()

    # OCR Performance
    print("OCR Performance:")
    overall_total_chars = sum(m['total_chars'] for m in ocr_metrics_list)
    overall_total_errors = sum(m['total_errors'] for m in ocr_metrics_list)

    if overall_total_chars > 0:
        overall_cer = overall_total_errors / overall_total_chars
    else:
        overall_cer = 0.0

    print(f"  CER (Character Error Rate): {overall_cer:.4f}")
    print(f"  Total Characters: {overall_total_chars}")
    print(f"  Total Errors: {overall_total_errors}")
    print(f"  Accuracy: {(1 - overall_cer) * 100:.2f}%")
    print()

    # Save results
    results = {
        'highlight_detection': {
            'miou': overall_h_metrics['overall']['mIoU'],
            'precision': overall_h_metrics['overall']['precision'],
            'recall': overall_h_metrics['overall']['recall'],
            'f1_score': overall_h_metrics['overall']['f1_score']
        },
        'ocr': {
            'cer': overall_cer,
            'total_characters': overall_total_chars,
            'total_errors': overall_total_errors,
            'accuracy': (1 - overall_cer) * 100
        },
        'num_samples': len(val_data)
    }

    with open('outputs/integrated_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to outputs/integrated_test_results.json")
    print()

    # Check targets
    print("=" * 70)
    print("TARGET ACHIEVEMENT")
    print("=" * 70 + "\n")

    print("Highlight Detection:")
    print(f"  Target mIoU: > 0.75")
    print(f"  Achieved:    {overall_h_metrics['overall']['mIoU']:.4f}")
    if overall_h_metrics['overall']['mIoU'] > 0.75:
        print(f"  ✅ TARGET ACHIEVED (+{(overall_h_metrics['overall']['mIoU'] - 0.75):.4f})")
    else:
        print(f"  ❌ Gap: {0.75 - overall_h_metrics['overall']['mIoU']:.4f}")
    print()

    print("OCR:")
    print(f"  Target CER: < 0.05 (5%)")
    print(f"  Achieved:   {overall_cer:.4f} ({overall_cer*100:.2f}%)")
    if overall_cer < 0.05:
        print(f"  ✅ TARGET ACHIEVED")
    else:
        print(f"  ❌ Gap: {(overall_cer - 0.05)*100:.2f}%")
    print()

    return results


if __name__ == "__main__":
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)

    # Run integrated test
    results = test_integrated_pipeline(num_samples=50)
