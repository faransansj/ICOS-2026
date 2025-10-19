"""
Highlight Overlay Module
Simulate highlighter marker effects on text images
"""

import random
from typing import List, Dict, Tuple
import numpy as np
import cv2
from PIL import Image


class HighlightOverlay:
    """Apply highlighter overlay simulation to text images"""

    # Highlight color definitions (BGR format for OpenCV)
    HIGHLIGHT_COLORS = {
        'yellow': (0, 255, 255),
        'green': (0, 255, 0),
        'pink': (203, 192, 255)
    }

    def __init__(
        self,
        alpha: float = 0.3,
        highlight_ratio: Tuple[float, float] = (0.2, 0.4),
        irregularity: float = 0.15
    ):
        """
        Initialize HighlightOverlay

        Args:
            alpha: Transparency of highlight (0.0-1.0)
            highlight_ratio: Min/max ratio of text to highlight
            irregularity: Edge irregularity factor (0.0-1.0)
        """
        self.alpha = alpha
        self.highlight_ratio = highlight_ratio
        self.irregularity = irregularity

    def apply_highlights(
        self,
        image: np.ndarray,
        annotations: List[Dict],
        colors: List[str] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Apply random highlights to image

        Args:
            image: Input image (PIL Image or numpy array)
            annotations: List of word annotations with bbox
            colors: List of colors to use (random if None)

        Returns:
            Tuple of (highlighted_image, highlight_annotations)
            highlight_annotations: List of {text, bbox, color}
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Copy image
        result = image.copy()

        # Determine how many words to highlight
        total_words = len(annotations)
        min_highlights = int(total_words * self.highlight_ratio[0])
        max_highlights = int(total_words * self.highlight_ratio[1])
        num_highlights = random.randint(min_highlights, max_highlights)

        # Randomly select words to highlight
        selected_indices = random.sample(range(total_words), num_highlights)

        # Use specified colors or random
        if colors is None:
            colors = list(self.HIGHLIGHT_COLORS.keys())

        highlight_annotations = []

        for idx in selected_indices:
            annotation = annotations[idx]
            color_name = random.choice(colors)
            color = self.HIGHLIGHT_COLORS[color_name]

            # Apply highlight
            bbox = annotation['bbox']
            result = self._apply_single_highlight(result, bbox, color)

            # Record highlight annotation
            highlight_annotations.append({
                'text': annotation['text'],
                'bbox': bbox,
                'color': color_name
            })

        return result, highlight_annotations

    def _apply_single_highlight(
        self,
        image: np.ndarray,
        bbox: List[int],
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Apply single highlight with transparency and irregularity

        Args:
            image: Input image
            bbox: Bounding box [x, y, w, h]
            color: Highlight color (BGR)

        Returns:
            Image with highlight applied
        """
        x, y, w, h = bbox

        # Add padding to highlight area (highlighter is usually larger than text)
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        # Create highlight overlay
        overlay = image.copy()

        # Apply irregular edges if enabled
        if self.irregularity > 0:
            # Generate noise for edge irregularity
            mask = self._create_irregular_mask(y2 - y1, x2 - x1)

            # Apply highlight with mask
            highlight_region = overlay[y1:y2, x1:x2]
            color_layer = np.full_like(highlight_region, color, dtype=np.uint8)

            # Blend with mask
            for c in range(3):
                highlight_region[:, :, c] = (
                    highlight_region[:, :, c] * (1 - mask * self.alpha) +
                    color_layer[:, :, c] * mask * self.alpha
                ).astype(np.uint8)

            overlay[y1:y2, x1:x2] = highlight_region
        else:
            # Simple rectangle highlight
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            # Blend with original
            cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0, image)
            return image

        return overlay

    def _create_irregular_mask(self, height: int, width: int) -> np.ndarray:
        """
        Create irregular edge mask to simulate real highlighter effect

        Args:
            height: Mask height
            width: Mask width

        Returns:
            Mask with values 0.0-1.0
        """
        # Create base mask (all ones)
        mask = np.ones((height, width), dtype=np.float32)

        # Add Gaussian noise to edges
        noise_strength = self.irregularity

        # Top and bottom edge noise
        edge_width = max(2, int(height * 0.2))

        # Top edge
        top_noise = np.random.normal(0, noise_strength, (edge_width, width))
        top_gradient = np.linspace(0, 1, edge_width).reshape(-1, 1)
        mask[:edge_width, :] *= np.clip(top_gradient + top_noise, 0, 1)

        # Bottom edge
        bottom_noise = np.random.normal(0, noise_strength, (edge_width, width))
        bottom_gradient = np.linspace(1, 0, edge_width).reshape(-1, 1)
        mask[-edge_width:, :] *= np.clip(bottom_gradient + bottom_noise, 0, 1)

        # Left and right edge noise
        edge_height = max(2, int(width * 0.1))

        # Left edge
        left_noise = np.random.normal(0, noise_strength, (height, edge_height))
        left_gradient = np.linspace(0, 1, edge_height)
        mask[:, :edge_height] *= np.clip(left_gradient + left_noise, 0, 1)

        # Right edge
        right_noise = np.random.normal(0, noise_strength, (height, edge_height))
        right_gradient = np.linspace(1, 0, edge_height)
        mask[:, -edge_height:] *= np.clip(right_gradient + right_noise, 0, 1)

        # Smooth mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        return mask

    def apply_consecutive_highlights(
        self,
        image: np.ndarray,
        annotations: List[Dict],
        consecutive_prob: float = 0.3,
        colors: List[str] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Apply highlights with tendency for consecutive words

        Args:
            image: Input image
            annotations: Word annotations
            consecutive_prob: Probability of highlighting consecutive words
            colors: Available colors

        Returns:
            Tuple of (highlighted_image, highlight_annotations)
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        result = image.copy()

        # Determine colors
        if colors is None:
            colors = list(self.HIGHLIGHT_COLORS.keys())

        total_words = len(annotations)
        min_highlights = int(total_words * self.highlight_ratio[0])
        max_highlights = int(total_words * self.highlight_ratio[1])
        num_highlights = random.randint(min_highlights, max_highlights)

        highlight_annotations = []
        highlighted_indices = set()

        i = 0
        while len(highlighted_indices) < num_highlights and i < total_words:
            if i in highlighted_indices:
                i += 1
                continue

            # Select color
            color_name = random.choice(colors)
            color = self.HIGHLIGHT_COLORS[color_name]

            # Add current word
            highlighted_indices.add(i)
            annotation = annotations[i]
            bbox = annotation['bbox']
            result = self._apply_single_highlight(result, bbox, color)

            highlight_annotations.append({
                'text': annotation['text'],
                'bbox': bbox,
                'color': color_name
            })

            # Potentially highlight consecutive words
            j = i + 1
            while (j < total_words and
                   len(highlighted_indices) < num_highlights and
                   random.random() < consecutive_prob):
                highlighted_indices.add(j)
                annotation = annotations[j]
                bbox = annotation['bbox']
                result = self._apply_single_highlight(result, bbox, color)

                highlight_annotations.append({
                    'text': annotation['text'],
                    'bbox': bbox,
                    'color': color_name
                })
                j += 1

            i = j

        return result, highlight_annotations


if __name__ == "__main__":
    # Test highlight overlay
    from text_image_generator import TextImageGenerator

    # Generate text image
    generator = TextImageGenerator()
    img, annotations = generator.generate_text_image()

    # Apply highlights
    overlay = HighlightOverlay(alpha=0.3, irregularity=0.2)
    highlighted_img, highlight_annot = overlay.apply_consecutive_highlights(
        np.array(img), annotations
    )

    # Save result
    cv2.imwrite("test_highlighted.png", highlighted_img)
    print(f"Applied {len(highlight_annot)} highlights")
    for annot in highlight_annot[:5]:
        print(f"  - {annot['text']}: {annot['color']}")
