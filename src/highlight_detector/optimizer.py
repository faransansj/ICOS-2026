"""
Hyperparameter Optimizer
Grid search for optimal HSV ranges and detector parameters
"""

import cv2
import numpy as np
import json
from typing import List, Dict, Tuple
from itertools import product
from tqdm import tqdm

from .highlight_detector import HighlightDetector
from .evaluator import HighlightEvaluator


class ParameterOptimizer:
    """Optimize highlight detector parameters using grid search"""

    def __init__(
        self,
        val_annotations_path: str,
        metric: str = 'mIoU'
    ):
        """
        Initialize optimizer

        Args:
            val_annotations_path: Path to validation annotations
            metric: Metric to optimize ('mIoU', 'f1_score', 'precision', 'recall')
        """
        self.val_annotations_path = val_annotations_path
        self.metric = metric

        # Load validation data
        with open(val_annotations_path, 'r') as f:
            self.val_data = json.load(f)

        print(f"Loaded {len(self.val_data)} validation images")

    def grid_search_hsv(
        self,
        color: str,
        h_range: List[int] = None,
        s_range: List[int] = None,
        v_range: List[int] = None,
        step: int = 5
    ) -> Dict:
        """
        Grid search for optimal HSV range for a color

        Args:
            color: Color to optimize
            h_range: Hue range to search [min, max]
            s_range: Saturation range to search [min, max]
            v_range: Value range to search [min, max]
            step: Step size for search

        Returns:
            Best HSV range and metrics
        """
        # Default ranges based on color
        if h_range is None:
            if color == 'yellow':
                h_range = [15, 35]
            elif color == 'green':
                h_range = [35, 85]
            elif color == 'pink':
                h_range = [135, 175]

        if s_range is None:
            s_range = [40, 100]

        if v_range is None:
            v_range = [40, 100]

        # Generate parameter combinations
        h_lower_vals = list(range(h_range[0], h_range[1], step))
        h_upper_vals = list(range(h_range[0], h_range[1], step))
        s_lower_vals = list(range(s_range[0], s_range[1], step * 2))
        v_lower_vals = list(range(v_range[0], v_range[1], step * 2))

        best_score = 0.0
        best_params = None
        best_metrics = None

        print(f"\nOptimizing HSV range for {color}...")
        print(f"Search space: {len(h_lower_vals) * len(h_upper_vals) * len(s_lower_vals) * len(v_lower_vals)} combinations")

        # Grid search
        for h_lower, h_upper, s_lower, v_lower in tqdm(
            product(h_lower_vals, h_upper_vals, s_lower_vals, v_lower_vals),
            desc=f"Searching {color}"
        ):
            # Skip invalid ranges
            if h_lower >= h_upper:
                continue

            # Create HSV range
            lower = np.array([h_lower, s_lower, v_lower])
            upper = np.array([h_upper, 255, 255])

            # Test this configuration
            detector = HighlightDetector()
            detector.update_hsv_range(color, lower, upper)

            evaluator = HighlightEvaluator(iou_threshold=0.5)

            # Evaluate on validation set
            all_preds = []
            all_gts = []

            for sample in self.val_data[:50]:  # Use subset for speed
                image = cv2.imread(sample['image_path'])
                detections = detector.detect(image, colors=[color])

                # Filter ground truths for this color
                gts = [
                    gt for gt in sample['highlight_annotations']
                    if gt['color'] == color
                ]

                all_preds.append(detections)
                all_gts.append(gts)

            # Calculate metrics
            metrics = evaluator.evaluate_dataset(all_preds, all_gts)
            score = metrics['overall'][self.metric]

            # Update best
            if score > best_score:
                best_score = score
                best_params = {
                    'lower': lower,
                    'upper': upper
                }
                best_metrics = metrics['overall']

        print(f"\nBest {color} HSV range:")
        print(f"  Lower: {best_params['lower']}")
        print(f"  Upper: {best_params['upper']}")
        print(f"  {self.metric}: {best_score:.4f}")

        return {
            'color': color,
            'best_params': {
                'lower': best_params['lower'].tolist(),
                'upper': best_params['upper'].tolist()
            },
            'best_score': best_score,
            'metrics': best_metrics
        }

    def grid_search_morph(
        self,
        kernel_sizes: List[Tuple[int, int]] = None,
        min_areas: List[int] = None,
        iterations: List[int] = None
    ) -> Dict:
        """
        Grid search for morphology parameters

        Args:
            kernel_sizes: List of kernel sizes to try
            min_areas: List of minimum areas to try
            iterations: List of iteration counts to try

        Returns:
            Best parameters and metrics
        """
        if kernel_sizes is None:
            kernel_sizes = [(3, 3), (5, 5), (7, 7)]

        if min_areas is None:
            min_areas = [50, 100, 150, 200]

        if iterations is None:
            iterations = [1, 2]

        best_score = 0.0
        best_params = None
        best_metrics = None

        print(f"\nOptimizing morphology parameters...")
        print(f"Search space: {len(kernel_sizes) * len(min_areas) * len(iterations)} combinations")

        for kernel_size, min_area, iters in tqdm(
            product(kernel_sizes, min_areas, iterations),
            desc="Searching morph params"
        ):
            # Create detector
            detector = HighlightDetector(
                kernel_size=kernel_size,
                min_area=min_area,
                morph_iterations=iters
            )

            evaluator = HighlightEvaluator(iou_threshold=0.5)

            # Evaluate
            all_preds = []
            all_gts = []

            for sample in self.val_data[:50]:
                image = cv2.imread(sample['image_path'])
                detections = detector.detect(image)

                all_preds.append(detections)
                all_gts.append(sample['highlight_annotations'])

            metrics = evaluator.evaluate_dataset(all_preds, all_gts)
            score = metrics['overall'][self.metric]

            if score > best_score:
                best_score = score
                best_params = {
                    'kernel_size': kernel_size,
                    'min_area': min_area,
                    'morph_iterations': iters
                }
                best_metrics = metrics['overall']

        print(f"\nBest morphology parameters:")
        print(f"  Kernel size: {best_params['kernel_size']}")
        print(f"  Min area: {best_params['min_area']}")
        print(f"  Iterations: {best_params['morph_iterations']}")
        print(f"  {self.metric}: {best_score:.4f}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'metrics': best_metrics
        }

    def full_optimization(
        self,
        colors: List[str] = ['yellow', 'green', 'pink'],
        save_path: str = 'configs/optimized_params.json'
    ) -> Dict:
        """
        Run full optimization pipeline

        Args:
            colors: Colors to optimize
            save_path: Path to save optimized parameters

        Returns:
            Optimized configuration
        """
        print("\n" + "=" * 60)
        print("FULL PARAMETER OPTIMIZATION")
        print("=" * 60)

        # Step 1: Optimize morphology parameters first
        print("\nStep 1: Optimizing morphology parameters...")
        morph_result = self.grid_search_morph()

        # Step 2: Optimize HSV ranges for each color
        hsv_results = {}
        for color in colors:
            hsv_result = self.grid_search_hsv(color)
            hsv_results[color] = hsv_result

        # Combine results
        optimized_config = {
            'hsv_ranges': {
                color: {
                    'lower': res['best_params']['lower'],
                    'upper': res['best_params']['upper']
                }
                for color, res in hsv_results.items()
            },
            'kernel_size': morph_result['best_params']['kernel_size'],
            'min_area': morph_result['best_params']['min_area'],
            'morph_iterations': morph_result['best_params']['morph_iterations'],
            'optimization_metric': self.metric,
            'results': {
                'morphology': morph_result,
                'hsv_ranges': hsv_results
            }
        }

        # Save configuration
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(optimized_config, f, indent=2)

        print(f"\n✓ Optimized configuration saved to: {save_path}")

        return optimized_config


if __name__ == "__main__":
    # Run optimization
    optimizer = ParameterOptimizer(
        val_annotations_path='data/validation/validation_annotations.json',
        metric='mIoU'
    )

    # Full optimization
    config = optimizer.full_optimization(
        colors=['yellow', 'green', 'pink'],
        save_path='configs/optimized_params.json'
    )

    print("\nOptimization complete!")
