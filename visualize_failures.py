#!/usr/bin/env python3
"""
Visualize OCR Failure Cases
Extract and analyze high-CER samples to identify root causes
"""

import sys
import os
import cv2
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from highlight_detector import HighlightDetector
from ocr import OCREngine, OCREvaluator


def visualize_failure_cases():
    """
    Visualize high-CER OCR failures to diagnose root causes
    """
    print("\n" + "=" * 70)
    print("OCR FAILURE VISUALIZATION")
    print("=" * 70 + "\n")

    # Load failure analysis
    with open('outputs/ocr_failure_analysis.json', 'r') as f:
        failure_data = json.load(f)

    high_cer_samples = failure_data['high_cer_samples']
    print(f"Analyzing {len(high_cer_samples)} high-CER samples\n")

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

    # Load validation annotations
    with open('data/validation/validation_annotations.json', 'r') as f:
        val_data = json.load(f)

    # Create output directory
    output_dir = Path('outputs/failure_visualizations')
    output_dir.mkdir(exist_ok=True)

    # Process each high-CER sample
    for i, sample_info in enumerate(high_cer_samples[:10]):
        image_name = sample_info['image']
        gt_text = sample_info['gt']
        pred_text = sample_info['pred']
        cer = sample_info['cer']
        confidence = sample_info['confidence']

        print(f"\n{'='*70}")
        print(f"Sample {i+1}: {image_name}")
        print(f"{'='*70}")
        print(f"Ground Truth: '{gt_text}'")
        print(f"Predicted:    '{pred_text}'")
        print(f"CER:          {cer:.3f}")
        print(f"Confidence:   {confidence:.1f}%")

        # Find matching sample in validation data
        matching_sample = None
        for sample in val_data:
            if sample['image_name'] == image_name:
                matching_sample = sample
                break

        if not matching_sample:
            print("  ⚠ Could not find sample in validation data")
            continue

        # Load image
        image = cv2.imread(matching_sample['image_path'])
        if image is None:
            print("  ⚠ Could not load image")
            continue

        # Detect highlights
        detections = detector.detect(image)

        # Find matching detection for this ground truth
        gt_texts = matching_sample.get('annotations', [])
        matching_gt = None
        for gt in gt_texts:
            if gt['text'] == gt_text:
                matching_gt = gt
                break

        if not matching_gt:
            print("  ⚠ Could not find matching ground truth annotation")
            continue

        # Find best matching detection
        best_det = None
        best_iou = 0
        for det in detections:
            iou = evaluator._calculate_bbox_iou(det['bbox'], matching_gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_det = det

        if not best_det:
            print("  ⚠ No detection found")
            continue

        print(f"\nDetection Analysis:")
        print(f"  Bbox IoU:  {best_iou:.3f}")
        print(f"  Color:     {best_det['color']}")
        print(f"  Bbox:      {best_det['bbox']}")
        print(f"  GT Bbox:   {matching_gt['bbox']}")

        # Extract regions
        det_bbox = best_det['bbox']
        gt_bbox = matching_gt['bbox']

        x, y, w, h = det_bbox
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)

        det_region = image[y_start:y_end, x_start:x_end].copy()

        # Create visualization
        vis_image = image.copy()

        # Draw ground truth bbox (green)
        gt_x, gt_y, gt_w, gt_h = gt_bbox
        cv2.rectangle(vis_image, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (0, 255, 0), 2)
        cv2.putText(vis_image, "GT", (gt_x, gt_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw detection bbox (blue)
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(vis_image, "DET", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save visualizations
        base_name = image_name.replace('.png', '')

        # Full image with bboxes
        cv2.imwrite(str(output_dir / f"{base_name}_full.png"), vis_image)

        # Detected region
        cv2.imwrite(str(output_dir / f"{base_name}_region.png"), det_region)

        # Try different preprocessing on the region
        gray = cv2.cvtColor(det_region, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(output_dir / f"{base_name}_gray.png"), gray)

        # Otsu thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(str(output_dir / f"{base_name}_otsu.png"), otsu)

        # Adaptive thresholding (different from original)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(str(output_dir / f"{base_name}_adaptive.png"), adaptive)

        # Test OCR on different versions
        ocr_results = {}

        # Original
        text_orig, conf_orig = ocr_engine.extract_text(image, det_bbox)
        ocr_results['original'] = (text_orig, conf_orig)

        # Grayscale
        text_gray, conf_gray = ocr_engine.extract_text(gray, None)
        ocr_results['gray'] = (text_gray, conf_gray)

        # Otsu
        text_otsu, conf_otsu = ocr_engine.extract_text(otsu, None)
        ocr_results['otsu'] = (text_otsu, conf_otsu)

        # Adaptive
        text_adaptive, conf_adaptive = ocr_engine.extract_text(adaptive, None)
        ocr_results['adaptive'] = (text_adaptive, conf_adaptive)

        print(f"\nOCR Comparison:")
        for method, (text, conf) in ocr_results.items():
            metrics = evaluator.calculate_cer(gt_text, text)
            print(f"  {method:10s}: '{text}' (CER: {metrics.cer:.3f}, conf: {conf:.1f}%)")

        # Analyze bbox quality
        bbox_overlap_x = min(gt_x + gt_w, x + w) - max(gt_x, x)
        bbox_overlap_y = min(gt_y + gt_h, y + h) - max(gt_y, y)

        if bbox_overlap_x > 0 and bbox_overlap_y > 0:
            overlap_area = bbox_overlap_x * bbox_overlap_y
            gt_area = gt_w * gt_h
            det_area = w * h
            overlap_ratio = overlap_area / max(gt_area, det_area)
            print(f"\nBbox Overlap Analysis:")
            print(f"  Overlap ratio: {overlap_ratio:.3f}")
            print(f"  Det width/GT width: {w/gt_w:.3f}")
            print(f"  Det height/GT height: {h/gt_h:.3f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70 + "\n")
    print(f"Visualizations saved to: {output_dir}")
    print(f"Files generated:")
    print(f"  *_full.png     - Full image with GT (green) and detection (blue) bboxes")
    print(f"  *_region.png   - Extracted detection region")
    print(f"  *_gray.png     - Grayscale conversion")
    print(f"  *_otsu.png     - Otsu thresholding")
    print(f"  *_adaptive.png - Adaptive thresholding")
    print()


if __name__ == "__main__":
    visualize_failure_cases()
