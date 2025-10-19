#!/usr/bin/env python3
"""
Analyze Remaining 25 Errors to Reach 95% Accuracy
Detailed analysis of each remaining error with root cause identification
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


def analyze_remaining_errors():
    """
    Analyze each of the 25 remaining errors in detail
    """
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS OF REMAINING 25 ERRORS")
    print("Target: 95% Accuracy (CER < 5%, max 15 errors from 298 chars)")
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

    ocr_engine = OCREngine(
        lang='kor+eng',
        config='--psm 7 --oem 3',
        preprocessing=False,
        min_confidence=60.0,
        use_multi_psm=True
    )

    evaluator = OCREvaluator()

    # Load validation data
    with open('data/validation/validation_annotations.json', 'r') as f:
        val_data = json.load(f)

    val_data = [d for d in val_data if not d.get('is_augmented', False)][:50]

    # Collect all errors
    all_errors = []
    error_by_type = defaultdict(list)
    error_by_confidence = defaultdict(list)
    error_by_color = defaultdict(list)

    total_chars = 0
    total_errors = 0

    for sample in val_data:
        image = cv2.imread(sample['image_path'])
        if image is None:
            continue

        detections = detector.detect(image)
        gt_texts = sample.get('annotations', [])

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
                pred_text, conf = ocr_engine.extract_text(image, det['bbox'], det['color'])

                metrics = evaluator.calculate_cer(gt_str, pred_text)

                total_chars += metrics.total_chars
                total_errors += (metrics.insertions + metrics.deletions + metrics.substitutions)

                if metrics.cer > 0:
                    error_info = {
                        'image': sample['image_name'],
                        'gt': gt_str,
                        'pred': pred_text,
                        'cer': metrics.cer,
                        'confidence': conf,
                        'color': det['color'],
                        'insertions': metrics.insertions,
                        'deletions': metrics.deletions,
                        'substitutions': metrics.substitutions,
                        'bbox_iou': best_iou
                    }

                    all_errors.append(error_info)

                    # Categorize by error type
                    if metrics.insertions > 0:
                        error_by_type['insertions'].append(error_info)
                    if metrics.deletions > 0:
                        error_by_type['deletions'].append(error_info)
                    if metrics.substitutions > 0:
                        error_by_type['substitutions'].append(error_info)

                    # Categorize by confidence
                    if conf < 70:
                        error_by_confidence['low_conf_<70'].append(error_info)
                    elif conf < 85:
                        error_by_confidence['medium_conf_70-85'].append(error_info)
                    else:
                        error_by_confidence['high_conf_>85'].append(error_info)

                    # Categorize by color
                    error_by_color[det['color']].append(error_info)

    print(f"Total Errors: {total_errors} / {total_chars} characters")
    print(f"Current CER: {total_errors/total_chars:.4f} ({total_errors/total_chars*100:.2f}%)")
    print(f"Current Accuracy: {(1-total_errors/total_chars)*100:.2f}%")
    print(f"\nTarget: ≤15 errors for 95% accuracy")
    print(f"Need to fix: {total_errors - 15} more errors\n")

    print("=" * 70)
    print("ERROR BREAKDOWN BY TYPE")
    print("=" * 70 + "\n")

    for error_type, errors in sorted(error_by_type.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"{error_type.upper()}: {len(errors)} cases")
        for err in errors[:3]:
            print(f"  '{err['gt']}' → '{err['pred']}' (CER: {err['cer']:.3f}, conf: {err['confidence']:.1f}%)")
        if len(errors) > 3:
            print(f"  ... and {len(errors)-3} more")
        print()

    print("=" * 70)
    print("ERROR BREAKDOWN BY CONFIDENCE")
    print("=" * 70 + "\n")

    for conf_range, errors in sorted(error_by_confidence.items()):
        print(f"{conf_range}: {len(errors)} cases")
        total_err_chars = sum(e['insertions'] + e['deletions'] + e['substitutions'] for e in errors)
        print(f"  Total error characters: {total_err_chars}")
        for err in errors[:2]:
            print(f"  '{err['gt']}' → '{err['pred']}' (CER: {err['cer']:.3f}, conf: {err['confidence']:.1f}%)")
        print()

    print("=" * 70)
    print("ERROR BREAKDOWN BY HIGHLIGHT COLOR")
    print("=" * 70 + "\n")

    for color, errors in sorted(error_by_color.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"{color.upper()}: {len(errors)} error cases")
        total_err_chars = sum(e['insertions'] + e['deletions'] + e['substitutions'] for e in errors)
        print(f"  Total error characters: {total_err_chars}")
        avg_cer = sum(e['cer'] for e in errors) / len(errors) if errors else 0
        print(f"  Average CER: {avg_cer:.3f}")
        for err in errors[:2]:
            print(f"  '{err['gt']}' → '{err['pred']}' (CER: {err['cer']:.3f})")
        print()

    print("=" * 70)
    print("TOP 10 HIGHEST CER ERRORS")
    print("=" * 70 + "\n")

    top_errors = sorted(all_errors, key=lambda x: x['cer'], reverse=True)[:10]
    for i, err in enumerate(top_errors, 1):
        print(f"{i}. Image: {err['image']}")
        print(f"   GT:   '{err['gt']}'")
        print(f"   Pred: '{err['pred']}'")
        print(f"   CER: {err['cer']:.3f} | Conf: {err['confidence']:.1f}% | Color: {err['color']}")
        print(f"   Errors: +{err['insertions']} -{err['deletions']} ~{err['substitutions']}")
        print()

    print("=" * 70)
    print("IMPROVEMENT OPPORTUNITIES")
    print("=" * 70 + "\n")

    # Identify fixable patterns
    low_conf_errors = error_by_confidence.get('low_conf_<70', [])
    if low_conf_errors:
        low_conf_chars = sum(e['insertions'] + e['deletions'] + e['substitutions'] for e in low_conf_errors)
        print(f"1. LOW CONFIDENCE FILTERING (< 70%)")
        print(f"   Cases: {len(low_conf_errors)}")
        print(f"   Error chars: {low_conf_chars}")
        print(f"   Potential gain: {low_conf_chars/total_chars*100:.2f}% accuracy")
        print()

    # Check for Korean particle errors
    particle_errors = [e for e in all_errors if any(p in e['gt'] for p in ['는', '은', '를', '을', '에서', '이가'])]
    if particle_errors:
        particle_chars = sum(e['deletions'] for e in particle_errors)
        print(f"2. KOREAN PARTICLE DELETIONS")
        print(f"   Cases: {len(particle_errors)}")
        print(f"   Deleted chars: {particle_chars}")
        print(f"   Common: Missing final character (는, 은, 를, 을)")
        for err in particle_errors[:3]:
            print(f"   '{err['gt']}' → '{err['pred']}'")
        print()

    # Check for substitution errors
    sub_errors = error_by_type.get('substitutions', [])
    if sub_errors:
        sub_chars = sum(e['substitutions'] for e in sub_errors)
        print(f"3. CHARACTER SUBSTITUTIONS")
        print(f"   Cases: {len(sub_errors)}")
        print(f"   Substituted chars: {sub_chars}")
        for err in sub_errors[:3]:
            print(f"   '{err['gt']}' → '{err['pred']}' (conf: {err['confidence']:.1f}%)")
        print()

    # Save detailed analysis
    with open('outputs/remaining_errors_analysis.json', 'w') as f:
        json.dump({
            'total_errors': total_errors,
            'total_chars': total_chars,
            'current_cer': total_errors / total_chars,
            'target_errors': 15,
            'errors_to_fix': total_errors - 15,
            'error_by_type': {k: len(v) for k, v in error_by_type.items()},
            'error_by_confidence': {k: len(v) for k, v in error_by_confidence.items()},
            'error_by_color': {k: len(v) for k, v in error_by_color.items()},
            'top_10_errors': top_errors[:10],
            'all_errors': all_errors
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Detailed analysis saved to outputs/remaining_errors_analysis.json")


if __name__ == "__main__":
    analyze_remaining_errors()
