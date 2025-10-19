"""
OCR Evaluator Module
Calculate Character Error Rate (CER) and other OCR metrics
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class OCRMetrics:
    """OCR evaluation metrics"""
    cer: float  # Character Error Rate
    total_chars: int
    correct_chars: int
    insertions: int
    deletions: int
    substitutions: int

    def __repr__(self):
        return (f"OCRMetrics(CER={self.cer:.4f}, "
                f"correct={self.correct_chars}/{self.total_chars})")


class OCREvaluator:
    """Evaluator for OCR performance"""

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> Tuple[int, int, int, int]:
        """
        Calculate Levenshtein distance and edit operations

        Args:
            s1: Reference string
            s2: Hypothesis string

        Returns:
            Tuple of (distance, insertions, deletions, substitutions)
        """
        len1 = len(s1)
        len2 = len(s2)

        # Create distance matrix
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        # Initialize first row and column
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        # Fill matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    cost = 0
                else:
                    cost = 1

                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )

        distance = dp[len1][len2]

        # Backtrack to count operation types
        insertions = 0
        deletions = 0
        substitutions = 0

        i, j = len1, len2
        while i > 0 or j > 0:
            if i == 0:
                insertions += j
                break
            elif j == 0:
                deletions += i
                break
            else:
                if s1[i-1] == s2[j-1]:
                    i -= 1
                    j -= 1
                else:
                    min_val = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

                    if dp[i-1][j-1] == min_val:
                        substitutions += 1
                        i -= 1
                        j -= 1
                    elif dp[i-1][j] == min_val:
                        deletions += 1
                        i -= 1
                    else:
                        insertions += 1
                        j -= 1

        return distance, insertions, deletions, substitutions

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for comparison

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Remove extra whitespace
        normalized = ' '.join(text.split())

        # Convert to lowercase (if needed)
        # normalized = normalized.lower()

        return normalized

    def calculate_cer(
        self,
        reference: str,
        hypothesis: str,
        normalize: bool = True
    ) -> OCRMetrics:
        """
        Calculate Character Error Rate

        CER = (I + D + S) / N
        where:
            I = insertions
            D = deletions
            S = substitutions
            N = total characters in reference

        Args:
            reference: Ground truth text
            hypothesis: OCR predicted text
            normalize: Whether to normalize texts before comparison

        Returns:
            OCRMetrics object
        """
        if normalize:
            ref = self.normalize_text(reference)
            hyp = self.normalize_text(hypothesis)
        else:
            ref = reference
            hyp = hypothesis

        # Calculate Levenshtein distance
        distance, insertions, deletions, substitutions = \
            self.levenshtein_distance(ref, hyp)

        # Calculate metrics
        total_chars = len(ref)

        if total_chars == 0:
            cer = 0.0 if len(hyp) == 0 else 1.0
            correct_chars = 0
        else:
            correct_chars = max(0, total_chars - (deletions + substitutions))
            cer = distance / total_chars

        return OCRMetrics(
            cer=cer,
            total_chars=total_chars,
            correct_chars=correct_chars,
            insertions=insertions,
            deletions=deletions,
            substitutions=substitutions
        )

    def evaluate_single_region(
        self,
        predicted_text: str,
        ground_truth_text: str
    ) -> Dict:
        """
        Evaluate OCR for a single highlight region

        Args:
            predicted_text: Text from OCR
            ground_truth_text: Ground truth text

        Returns:
            Dictionary with metrics
        """
        metrics = self.calculate_cer(ground_truth_text, predicted_text)

        return {
            'cer': metrics.cer,
            'total_chars': metrics.total_chars,
            'correct_chars': metrics.correct_chars,
            'insertions': metrics.insertions,
            'deletions': metrics.deletions,
            'substitutions': metrics.substitutions,
            'predicted_text': predicted_text,
            'ground_truth_text': ground_truth_text
        }

    def evaluate_detections(
        self,
        ocr_results: List[Dict],
        ground_truth_annotations: List[Dict]
    ) -> Dict:
        """
        Evaluate OCR results against ground truth

        Args:
            ocr_results: List of OCR results with 'text' and 'bbox'
            ground_truth_annotations: List of GT annotations with 'text' and 'bbox'

        Returns:
            Dictionary with overall metrics
        """
        # Match OCR results to ground truth based on bbox IoU
        matches = self._match_results_to_gt(ocr_results, ground_truth_annotations)

        # Calculate metrics for each match
        all_metrics = []
        total_chars = 0
        total_errors = 0

        for ocr_result, gt_annot in matches:
            if gt_annot is None:
                # False positive - no ground truth match
                continue

            pred_text = ocr_result.get('text', '')
            gt_text = gt_annot.get('text', '')

            metrics = self.calculate_cer(gt_text, pred_text)

            all_metrics.append({
                'cer': metrics.cer,
                'chars': metrics.total_chars,
                'errors': metrics.insertions + metrics.deletions + metrics.substitutions,
                'predicted': pred_text,
                'ground_truth': gt_text
            })

            total_chars += metrics.total_chars
            total_errors += (metrics.insertions + metrics.deletions +
                           metrics.substitutions)

        # Calculate overall CER
        if total_chars > 0:
            overall_cer = total_errors / total_chars
        else:
            overall_cer = 0.0

        return {
            'overall_cer': overall_cer,
            'total_characters': total_chars,
            'total_errors': total_errors,
            'num_regions': len(all_metrics),
            'region_metrics': all_metrics
        }

    def _match_results_to_gt(
        self,
        ocr_results: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5
    ) -> List[Tuple[Dict, Dict]]:
        """
        Match OCR results to ground truth based on bbox IoU

        Args:
            ocr_results: OCR results
            ground_truth: Ground truth annotations
            iou_threshold: Minimum IoU for matching

        Returns:
            List of (ocr_result, gt_annotation) tuples
        """
        matches = []
        used_gt = set()

        for ocr_result in ocr_results:
            ocr_bbox = ocr_result.get('bbox', [])

            best_iou = 0
            best_gt = None

            for i, gt_annot in enumerate(ground_truth):
                if i in used_gt:
                    continue

                gt_bbox = gt_annot.get('bbox', [])
                iou = self._calculate_bbox_iou(ocr_bbox, gt_bbox)

                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt = (i, gt_annot)

            if best_gt is not None:
                idx, gt_annot = best_gt
                used_gt.add(idx)
                matches.append((ocr_result, gt_annot))
            else:
                matches.append((ocr_result, None))

        return matches

    @staticmethod
    def _calculate_bbox_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate IoU between two bounding boxes"""
        if not bbox1 or not bbox2:
            return 0.0

        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


if __name__ == "__main__":
    # Test evaluator
    evaluator = OCREvaluator()

    # Test CER calculation
    print("=" * 60)
    print("OCR EVALUATOR TEST")
    print("=" * 60 + "\n")

    # Test case 1: Perfect match
    ref1 = "안녕하세요"
    hyp1 = "안녕하세요"
    metrics1 = evaluator.calculate_cer(ref1, hyp1)
    print(f"Test 1 - Perfect match:")
    print(f"  Reference: '{ref1}'")
    print(f"  Hypothesis: '{hyp1}'")
    print(f"  {metrics1}\n")

    # Test case 2: One substitution
    ref2 = "안녕하세요"
    hyp2 = "안녕히세요"
    metrics2 = evaluator.calculate_cer(ref2, hyp2)
    print(f"Test 2 - One substitution:")
    print(f"  Reference: '{ref2}'")
    print(f"  Hypothesis: '{hyp2}'")
    print(f"  {metrics2}\n")

    # Test case 3: Multiple errors
    ref3 = "하이라이트 텍스트"
    hyp3 = "하라이트 텍트"
    metrics3 = evaluator.calculate_cer(ref3, hyp3)
    print(f"Test 3 - Multiple errors:")
    print(f"  Reference: '{ref3}'")
    print(f"  Hypothesis: '{hyp3}'")
    print(f"  {metrics3}\n")

    print("✓ Evaluator test complete")
