"""
Highlight Detector Module
HSV color space-based highlight region detection
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class HighlightDetector:
    """Detect highlighted regions using HSV color space filtering"""

    # Default HSV color ranges for highlights
    DEFAULT_HSV_RANGES = {
        'yellow': {
            'lower': np.array([20, 100, 100]),
            'upper': np.array([30, 255, 255])
        },
        'green': {
            'lower': np.array([40, 40, 40]),
            'upper': np.array([80, 255, 255])
        },
        'pink': {
            'lower': np.array([140, 50, 50]),
            'upper': np.array([170, 255, 255])
        }
    }

    def __init__(
        self,
        hsv_ranges: Dict[str, Dict[str, np.ndarray]] = None,
        kernel_size: Tuple[int, int] = (5, 5),
        min_area: int = 100,
        morph_iterations: int = 1
    ):
        """
        Initialize HighlightDetector

        Args:
            hsv_ranges: Custom HSV color ranges for each color
            kernel_size: Morphology kernel size (width, height)
            min_area: Minimum contour area to filter noise
            morph_iterations: Number of morphology operation iterations
        """
        self.hsv_ranges = hsv_ranges if hsv_ranges else self.DEFAULT_HSV_RANGES
        self.kernel_size = kernel_size
        self.min_area = min_area
        self.morph_iterations = morph_iterations

        # Create morphology kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.kernel_size
        )

    def detect(
        self,
        image: np.ndarray,
        colors: List[str] = None,
        return_masks: bool = False
    ) -> List[Dict]:
        """
        Detect all highlight regions in image

        Args:
            image: Input image (BGR format)
            colors: List of colors to detect (default: all)
            return_masks: Whether to return binary masks

        Returns:
            List of detection results:
            [
                {
                    'bbox': [x, y, w, h],
                    'color': 'yellow',
                    'area': 1000,
                    'confidence': 0.95,
                    'mask': np.ndarray (if return_masks=True)
                },
                ...
            ]
        """
        if colors is None:
            colors = list(self.hsv_ranges.keys())

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        all_detections = []

        for color in colors:
            if color not in self.hsv_ranges:
                print(f"Warning: Color '{color}' not in HSV ranges, skipping")
                continue

            # Detect this color
            detections = self._detect_single_color(
                hsv,
                color,
                return_masks=return_masks
            )
            all_detections.extend(detections)

        return all_detections

    def _detect_single_color(
        self,
        hsv: np.ndarray,
        color: str,
        return_masks: bool = False
    ) -> List[Dict]:
        """
        Detect highlights of a single color

        Args:
            hsv: HSV image
            color: Color name
            return_masks: Whether to return masks

        Returns:
            List of detections for this color
        """
        # Get color range
        lower = self.hsv_ranges[color]['lower']
        upper = self.hsv_ranges[color]['upper']

        # Create color mask
        mask = cv2.inRange(hsv, lower, upper)

        # Apply morphology operations
        mask = self._apply_morphology(mask)

        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Extract bounding boxes
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter small contours (noise)
            if area < self.min_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate confidence (area ratio of contour to bbox)
            bbox_area = w * h
            confidence = area / bbox_area if bbox_area > 0 else 0

            detection = {
                'bbox': [x, y, w, h],
                'color': color,
                'area': int(area),
                'confidence': float(confidence)
            }

            if return_masks:
                # Create individual mask for this detection
                individual_mask = np.zeros_like(mask)
                cv2.drawContours(individual_mask, [contour], -1, 255, -1)
                detection['mask'] = individual_mask

            detections.append(detection)

        return detections

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphology operations to clean up mask

        Args:
            mask: Binary mask

        Returns:
            Cleaned mask
        """
        # Closing: remove small holes
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=self.morph_iterations
        )

        # Opening: remove small noise
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel,
            iterations=self.morph_iterations
        )

        return mask

    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        color_map: Dict[str, Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on image

        Args:
            image: Input image
            detections: List of detections
            color_map: BGR colors for each highlight color

        Returns:
            Image with drawn bounding boxes
        """
        if color_map is None:
            color_map = {
                'yellow': (0, 255, 255),
                'green': (0, 255, 0),
                'pink': (255, 0, 255)
            }

        result = image.copy()

        for det in detections:
            x, y, w, h = det['bbox']
            color_name = det['color']
            bgr_color = color_map.get(color_name, (255, 255, 255))

            # Draw rectangle
            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                bgr_color,
                2
            )

            # Draw label
            label = f"{color_name}: {det['confidence']:.2f}"
            cv2.putText(
                result,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                bgr_color,
                1
            )

        return result

    def update_hsv_range(
        self,
        color: str,
        lower: np.ndarray,
        upper: np.ndarray
    ):
        """
        Update HSV range for a specific color

        Args:
            color: Color name
            lower: Lower HSV bound
            upper: Upper HSV bound
        """
        self.hsv_ranges[color] = {
            'lower': lower,
            'upper': upper
        }

    def get_config(self) -> Dict:
        """Get current detector configuration"""
        return {
            'hsv_ranges': {
                color: {
                    'lower': ranges['lower'].tolist(),
                    'upper': ranges['upper'].tolist()
                }
                for color, ranges in self.hsv_ranges.items()
            },
            'kernel_size': self.kernel_size,
            'min_area': self.min_area,
            'morph_iterations': self.morph_iterations
        }


if __name__ == "__main__":
    # Test highlight detection
    import json

    # Load validation data
    with open('data/validation/validation_annotations.json', 'r') as f:
        val_data = json.load(f)

    # Test on first image
    sample = val_data[0]
    image = cv2.imread(sample['image_path'])

    # Create detector
    detector = HighlightDetector()

    # Detect highlights
    detections = detector.detect(image)

    print(f"Detected {len(detections)} highlights")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['color']}: bbox={det['bbox']}, confidence={det['confidence']:.2f}")

    # Visualize
    vis = detector.visualize_detections(image, detections)
    cv2.imwrite('test_detection.png', vis)
    print("Saved visualization to test_detection.png")
