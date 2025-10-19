"""
Highlight Detection Evaluator
Performance evaluation using IoU, mIoU, Precision, Recall, F1
"""

import numpy as np
from typing import List, Dict, Tuple
import json


class HighlightEvaluator:
    """Evaluate highlight detection performance"""

    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize evaluator

        Args:
            iou_threshold: IoU threshold for positive match
        """
        self.iou_threshold = iou_threshold

    def calculate_iou(
        self,
        bbox1: List[int],
        bbox2: List[int]
    ) -> float:
        """
        Calculate Intersection over Union between two bounding boxes

        Args:
            bbox1: [x, y, w, h]
            bbox2: [x, y, w, h]

        Returns:
            IoU score (0.0 to 1.0)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to corner coordinates
        x1_min, y1_min = x1, y1
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_min, y2_min = x2, y2
        x2_max, y2_max = x2 + w2, y2 + h2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # Check if there's an intersection
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        # Calculate areas
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - inter_area

        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0.0

        return iou

    def match_detections(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ) -> Tuple[List[Tuple], List[int], List[int]]:
        """
        Match predictions to ground truths using IoU

        Args:
            predictions: List of predicted detections
            ground_truths: List of ground truth annotations

        Returns:
            Tuple of (matches, unmatched_preds, unmatched_gts)
            matches: [(pred_idx, gt_idx, iou), ...]
            unmatched_preds: [pred_idx, ...]
            unmatched_gts: [gt_idx, ...]
        """
        if len(predictions) == 0 or len(ground_truths) == 0:
            return [], list(range(len(predictions))), list(range(len(ground_truths)))

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(predictions), len(ground_truths)))

        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truths):
                # Only match same colors
                if pred['color'] == gt['color']:
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    iou_matrix[i, j] = iou

        # Greedy matching (highest IoU first)
        matches = []
        matched_preds = set()
        matched_gts = set()

        # Sort by IoU (descending)
        iou_pairs = []
        for i in range(len(predictions)):
            for j in range(len(ground_truths)):
                if iou_matrix[i, j] >= self.iou_threshold:
                    iou_pairs.append((i, j, iou_matrix[i, j]))

        iou_pairs.sort(key=lambda x: x[2], reverse=True)

        # Match greedily
        for pred_idx, gt_idx, iou in iou_pairs:
            if pred_idx not in matched_preds and gt_idx not in matched_gts:
                matches.append((pred_idx, gt_idx, iou))
                matched_preds.add(pred_idx)
                matched_gts.add(gt_idx)

        # Find unmatched
        unmatched_preds = [i for i in range(len(predictions)) if i not in matched_preds]
        unmatched_gts = [j for j in range(len(ground_truths)) if j not in matched_gts]

        return matches, unmatched_preds, unmatched_gts

    def evaluate_single_image(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ) -> Dict:
        """
        Evaluate detections for a single image

        Args:
            predictions: Predicted detections
            ground_truths: Ground truth annotations

        Returns:
            Evaluation metrics
        """
        matches, unmatched_preds, unmatched_gts = self.match_detections(
            predictions, ground_truths
        )

        # Calculate metrics
        tp = len(matches)  # True Positives
        fp = len(unmatched_preds)  # False Positives
        fn = len(unmatched_gts)  # False Negatives

        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Mean IoU (only for matched detections)
        mean_iou = np.mean([iou for _, _, iou in matches]) if matches else 0.0

        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_iou': mean_iou,
            'num_predictions': len(predictions),
            'num_ground_truths': len(ground_truths)
        }

    def evaluate_dataset(
        self,
        all_predictions: List[List[Dict]],
        all_ground_truths: List[List[Dict]]
    ) -> Dict:
        """
        Evaluate detections for entire dataset

        Args:
            all_predictions: List of predictions for each image
            all_ground_truths: List of ground truths for each image

        Returns:
            Aggregated metrics
        """
        assert len(all_predictions) == len(all_ground_truths), \
            "Number of predictions and ground truths must match"

        # Aggregate counts
        total_tp = 0
        total_fp = 0
        total_fn = 0
        all_ious = []

        # Per-image metrics
        image_metrics = []

        for preds, gts in zip(all_predictions, all_ground_truths):
            metrics = self.evaluate_single_image(preds, gts)

            total_tp += metrics['true_positives']
            total_fp += metrics['false_positives']
            total_fn += metrics['false_negatives']

            # Collect IoUs
            matches, _, _ = self.match_detections(preds, gts)
            all_ious.extend([iou for _, _, iou in matches])

            image_metrics.append(metrics)

        # Calculate overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        miou = np.mean(all_ious) if all_ious else 0.0

        # Per-image averages
        avg_precision = np.mean([m['precision'] for m in image_metrics])
        avg_recall = np.mean([m['recall'] for m in image_metrics])
        avg_f1 = np.mean([m['f1_score'] for m in image_metrics])

        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mIoU': miou,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn,
                'total_predictions': total_tp + total_fp,
                'total_ground_truths': total_tp + total_fn
            },
            'per_image_avg': {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1
            },
            'image_metrics': image_metrics,
            'num_images': len(all_predictions)
        }

    def evaluate_by_color(
        self,
        all_predictions: List[List[Dict]],
        all_ground_truths: List[List[Dict]],
        colors: List[str] = ['yellow', 'green', 'pink']
    ) -> Dict:
        """
        Evaluate performance per color

        Args:
            all_predictions: All predictions
            all_ground_truths: All ground truths
            colors: List of colors to evaluate

        Returns:
            Per-color metrics
        """
        color_metrics = {}

        for color in colors:
            # Filter by color
            color_preds = [
                [p for p in preds if p['color'] == color]
                for preds in all_predictions
            ]
            color_gts = [
                [gt for gt in gts if gt['color'] == color]
                for gts in all_ground_truths
            ]

            # Evaluate
            metrics = self.evaluate_dataset(color_preds, color_gts)
            color_metrics[color] = metrics['overall']

        return color_metrics

    def generate_report(
        self,
        metrics: Dict,
        output_path: str = None
    ) -> str:
        """
        Generate evaluation report

        Args:
            metrics: Evaluation metrics
            output_path: Path to save report (optional)

        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("HIGHLIGHT DETECTION EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Overall metrics
        overall = metrics['overall']
        report_lines.append("Overall Performance:")
        report_lines.append(f"  mIoU:      {overall['mIoU']:.4f}")
        report_lines.append(f"  Precision: {overall['precision']:.4f}")
        report_lines.append(f"  Recall:    {overall['recall']:.4f}")
        report_lines.append(f"  F1-Score:  {overall['f1_score']:.4f}")
        report_lines.append("")

        report_lines.append("Detection Statistics:")
        report_lines.append(f"  True Positives:  {overall['total_tp']}")
        report_lines.append(f"  False Positives: {overall['total_fp']}")
        report_lines.append(f"  False Negatives: {overall['total_fn']}")
        report_lines.append(f"  Total Predictions: {overall['total_predictions']}")
        report_lines.append(f"  Total Ground Truths: {overall['total_ground_truths']}")
        report_lines.append("")

        # Per-image averages
        per_image = metrics['per_image_avg']
        report_lines.append("Per-Image Averages:")
        report_lines.append(f"  Precision: {per_image['precision']:.4f}")
        report_lines.append(f"  Recall:    {per_image['recall']:.4f}")
        report_lines.append(f"  F1-Score:  {per_image['f1_score']:.4f}")
        report_lines.append("")

        report_lines.append(f"Number of Images: {metrics['num_images']}")
        report_lines.append("=" * 60)

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report


if __name__ == "__main__":
    # Test evaluator
    evaluator = HighlightEvaluator(iou_threshold=0.5)

    # Example predictions and ground truths
    preds = [
        {'bbox': [50, 50, 80, 17], 'color': 'pink'},
        {'bbox': [140, 50, 50, 17], 'color': 'yellow'}
    ]

    gts = [
        {'bbox': [50, 50, 80, 17], 'color': 'pink'},
        {'bbox': [135, 50, 48, 17], 'color': 'yellow'}
    ]

    # Evaluate
    metrics = evaluator.evaluate_single_image(preds, gts)
    print(json.dumps(metrics, indent=2))
