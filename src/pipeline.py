"""
End-to-End Highlight Text Extraction Pipeline
Integrates highlight detection and OCR for complete text extraction
"""

import cv2
import numpy as np
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

from highlight_detector import HighlightDetector
from ocr import OCREngine, OCRResult


@dataclass
class HighlightTextResult:
    """Complete result for a single highlight region"""
    text: str
    color: str
    confidence: float
    bbox: List[int]  # [x, y, w, h]

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'text': self.text,
            'color': self.color,
            'confidence': round(self.confidence, 2),
            'bbox': {
                'x': self.bbox[0],
                'y': self.bbox[1],
                'width': self.bbox[2],
                'height': self.bbox[3]
            }
        }


@dataclass
class ExtractionResult:
    """Complete extraction result for an image"""
    image_path: str
    total_highlights: int
    highlights_by_color: Dict[str, int]
    results: List[HighlightTextResult]

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'image_path': self.image_path,
            'total_highlights': self.total_highlights,
            'highlights_by_color': self.highlights_by_color,
            'results': [r.to_dict() for r in self.results]
        }

    def get_texts_by_color(self) -> Dict[str, List[str]]:
        """Get all texts grouped by highlight color"""
        texts_by_color = {
            'yellow': [],
            'green': [],
            'pink': []
        }

        for result in self.results:
            if result.text:  # Only include non-empty texts
                texts_by_color[result.color].append(result.text)

        return texts_by_color


class HighlightTextExtractor:
    """
    End-to-end pipeline for extracting text from highlighted regions

    Usage:
        extractor = HighlightTextExtractor()
        result = extractor.process_image('path/to/image.jpg')
        extractor.save_json(result, 'output.json')
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        ocr_lang: str = 'kor+eng',
        ocr_config: str = '--psm 7 --oem 3',
        min_confidence: float = 60.0
    ):
        """
        Initialize the extraction pipeline

        Args:
            config_path: Path to HSV configuration file
            ocr_lang: Tesseract language setting
            ocr_config: Tesseract configuration string
            min_confidence: Minimum OCR confidence threshold
        """
        # Load HSV configuration
        if config_path is None:
            config_path = 'configs/optimized_hsv_ranges.json'

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Initialize highlight detector
        hsv_ranges = {
            color: {
                'lower': np.array(ranges['lower']),
                'upper': np.array(ranges['upper'])
            }
            for color, ranges in config['hsv_ranges'].items()
        }

        self.highlight_detector = HighlightDetector(
            hsv_ranges=hsv_ranges,
            kernel_size=tuple(config['kernel_size']),
            min_area=config['min_area'],
            morph_iterations=config['morph_iterations']
        )

        # Initialize OCR engine
        self.ocr_engine = OCREngine(
            lang=ocr_lang,
            config=ocr_config,
            preprocessing=False,
            min_confidence=min_confidence,
            use_multi_psm=True
        )

        print(f"✓ HighlightTextExtractor initialized")
        print(f"  - Highlight colors: {list(hsv_ranges.keys())}")
        print(f"  - OCR language: {ocr_lang}")
        print(f"  - Min confidence: {min_confidence}%")

    def process_image(
        self,
        image_path: str,
        visualize: bool = False,
        output_path: Optional[str] = None
    ) -> ExtractionResult:
        """
        Process a single image and extract all highlighted text

        Args:
            image_path: Path to input image
            visualize: Whether to create visualization
            output_path: Path to save visualization (if visualize=True)

        Returns:
            ExtractionResult containing all detected highlights and texts
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Step 1: Detect highlights
        detections = self.highlight_detector.detect(image)

        # Step 2: Extract text from each highlight
        results = []
        for det in detections:
            text, confidence = self.ocr_engine.extract_text(
                image,
                det['bbox'],
                det['color']
            )

            result = HighlightTextResult(
                text=text,
                color=det['color'],
                confidence=confidence,
                bbox=det['bbox']
            )
            results.append(result)

        # Count highlights by color
        highlights_by_color = {'yellow': 0, 'green': 0, 'pink': 0}
        for det in detections:
            highlights_by_color[det['color']] += 1

        extraction_result = ExtractionResult(
            image_path=image_path,
            total_highlights=len(detections),
            highlights_by_color=highlights_by_color,
            results=results
        )

        # Create visualization if requested
        if visualize:
            vis_image = self._create_visualization(image, results)
            if output_path:
                cv2.imwrite(output_path, vis_image)

        return extraction_result

    def _create_visualization(
        self,
        image: np.ndarray,
        results: List[HighlightTextResult]
    ) -> np.ndarray:
        """
        Create visualization with bounding boxes and text labels

        Args:
            image: Original image
            results: Extraction results

        Returns:
            Annotated image
        """
        vis_image = image.copy()

        # Color mapping for visualization
        color_map = {
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'pink': (255, 0, 255)
        }

        for result in results:
            x, y, w, h = result.bbox
            color = color_map.get(result.color, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

            # Draw text label
            label = f"{result.text} ({result.confidence:.0f}%)"

            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis_image,
                (x, y - text_height - 5),
                (x + text_width, y),
                color,
                -1
            )

            # Text
            cv2.putText(
                vis_image,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

        return vis_image

    def save_json(
        self,
        result: ExtractionResult,
        output_path: str,
        pretty: bool = True
    ):
        """
        Save extraction result as JSON

        Args:
            result: ExtractionResult to save
            output_path: Path to output JSON file
            pretty: Whether to use pretty printing
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                result.to_dict(),
                f,
                ensure_ascii=False,
                indent=2 if pretty else None
            )

        print(f"✓ Saved JSON to: {output_path}")

    def save_csv(
        self,
        result: ExtractionResult,
        output_path: str
    ):
        """
        Save extraction result as CSV

        Args:
            result: ExtractionResult to save
            output_path: Path to output CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'Color', 'Text', 'Confidence', 'X', 'Y', 'Width', 'Height'
            ])

            # Write data
            for r in result.results:
                writer.writerow([
                    r.color,
                    r.text,
                    f"{r.confidence:.2f}",
                    r.bbox[0],
                    r.bbox[1],
                    r.bbox[2],
                    r.bbox[3]
                ])

        print(f"✓ Saved CSV to: {output_path}")

    def save_summary(
        self,
        result: ExtractionResult,
        output_path: str
    ):
        """
        Save human-readable summary

        Args:
            result: ExtractionResult to save
            output_path: Path to output text file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        texts_by_color = result.get_texts_by_color()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("HIGHLIGHT TEXT EXTRACTION SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Image: {result.image_path}\n")
            f.write(f"Total Highlights: {result.total_highlights}\n\n")

            f.write("Highlights by Color:\n")
            for color, count in result.highlights_by_color.items():
                f.write(f"  {color.capitalize()}: {count}\n")
            f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("EXTRACTED TEXT BY COLOR\n")
            f.write("=" * 70 + "\n\n")

            for color in ['yellow', 'green', 'pink']:
                texts = texts_by_color[color]
                if texts:
                    f.write(f"{color.upper()} HIGHLIGHTS ({len(texts)}):\n")
                    for i, text in enumerate(texts, 1):
                        f.write(f"  {i}. {text}\n")
                    f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("=" * 70 + "\n\n")

            for i, r in enumerate(result.results, 1):
                f.write(f"{i}. [{r.color.upper()}] \"{r.text}\"\n")
                f.write(f"   Confidence: {r.confidence:.1f}%\n")
                f.write(f"   Location: x={r.bbox[0]}, y={r.bbox[1]}, "
                       f"w={r.bbox[2]}, h={r.bbox[3]}\n\n")

        print(f"✓ Saved summary to: {output_path}")

    def process_batch(
        self,
        image_paths: List[str],
        output_dir: str = 'outputs/batch',
        formats: List[str] = ['json', 'csv', 'txt']
    ) -> List[ExtractionResult]:
        """
        Process multiple images in batch

        Args:
            image_paths: List of image paths to process
            output_dir: Directory to save outputs
            formats: List of output formats ('json', 'csv', 'txt', 'vis')

        Returns:
            List of ExtractionResults
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING: {len(image_paths)} images")
        print(f"{'='*70}\n")

        for i, image_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] Processing: {image_path}")

            try:
                # Process image
                result = self.process_image(
                    image_path,
                    visualize='vis' in formats,
                    output_path=str(output_dir / f"{Path(image_path).stem}_vis.jpg")
                    if 'vis' in formats else None
                )

                results.append(result)

                # Save in requested formats
                base_name = Path(image_path).stem

                if 'json' in formats:
                    self.save_json(result, str(output_dir / f"{base_name}.json"))

                if 'csv' in formats:
                    self.save_csv(result, str(output_dir / f"{base_name}.csv"))

                if 'txt' in formats:
                    self.save_summary(result, str(output_dir / f"{base_name}.txt"))

                print(f"  ✓ Found {result.total_highlights} highlights\n")

            except Exception as e:
                print(f"  ✗ Error: {e}\n")

        print(f"{'='*70}")
        print(f"BATCH COMPLETE: {len(results)}/{len(image_paths)} successful")
        print(f"{'='*70}\n")

        return results


if __name__ == "__main__":
    # Quick test
    extractor = HighlightTextExtractor()
    print("\n✓ Pipeline module ready")
