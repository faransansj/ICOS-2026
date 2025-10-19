#!/usr/bin/env python3
"""
Highlight Text Extractor - Command Line Interface

Extract text from highlighted regions in images with color-based classification.

Usage:
    # Single image
    python extract_highlights.py image.jpg

    # With visualization
    python extract_highlights.py image.jpg --visualize

    # Batch processing
    python extract_highlights.py image1.jpg image2.jpg image3.jpg --batch

    # Specify output directory and formats
    python extract_highlights.py image.jpg -o outputs/results --format json csv txt vis
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import HighlightTextExtractor


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from highlighted regions in images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image with all outputs
  python extract_highlights.py sample.jpg --visualize --format json csv txt

  # Batch process multiple images
  python extract_highlights.py img1.jpg img2.jpg img3.jpg --batch

  # Custom output directory
  python extract_highlights.py image.jpg -o my_results

  # Only JSON output
  python extract_highlights.py image.jpg --format json
        """
    )

    parser.add_argument(
        'images',
        nargs='+',
        help='Input image path(s)'
    )

    parser.add_argument(
        '-o', '--output',
        default='outputs/extracted',
        help='Output directory (default: outputs/extracted)'
    )

    parser.add_argument(
        '--format',
        nargs='+',
        choices=['json', 'csv', 'txt', 'vis'],
        default=['json', 'txt'],
        help='Output format(s) (default: json txt)'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization (shortcut for --format ... vis)'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode: save all outputs to output directory'
    )

    parser.add_argument(
        '--config',
        default='configs/optimized_hsv_ranges.json',
        help='Path to HSV configuration file'
    )

    parser.add_argument(
        '--lang',
        default='kor+eng',
        help='OCR language (default: kor+eng)'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=60.0,
        help='Minimum OCR confidence threshold (default: 60.0)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    # Add visualization to formats if requested
    if args.visualize and 'vis' not in args.format:
        args.format.append('vis')

    # Validate input files
    valid_images = []
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Warning: File not found: {img_path}")
        else:
            valid_images.append(img_path)

    if not valid_images:
        print("Error: No valid input images found")
        return 1

    # Initialize extractor
    if not args.quiet:
        print("\n" + "=" * 70)
        print("HIGHLIGHT TEXT EXTRACTOR")
        print("=" * 70 + "\n")

    try:
        extractor = HighlightTextExtractor(
            config_path=args.config,
            ocr_lang=args.lang,
            min_confidence=args.confidence
        )
    except Exception as e:
        print(f"Error initializing extractor: {e}")
        return 1

    # Process images
    if len(valid_images) == 1 and not args.batch:
        # Single image mode
        image_path = valid_images[0]

        if not args.quiet:
            print(f"\nProcessing: {image_path}\n")

        try:
            # Process image
            result = extractor.process_image(
                image_path,
                visualize='vis' in args.format,
                output_path=None  # Will save separately
            )

            # Print summary
            if not args.quiet:
                print(f"\n{'='*70}")
                print("EXTRACTION RESULTS")
                print(f"{'='*70}\n")

                print(f"Total Highlights: {result.total_highlights}")
                print(f"\nBy Color:")
                for color, count in result.highlights_by_color.items():
                    print(f"  {color.capitalize()}: {count}")

                print(f"\nExtracted Texts:")
                texts_by_color = result.get_texts_by_color()
                for color in ['yellow', 'green', 'pink']:
                    texts = texts_by_color[color]
                    if texts:
                        print(f"\n  {color.upper()}:")
                        for i, text in enumerate(texts, 1):
                            print(f"    {i}. {text}")

                print()

            # Save outputs
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            base_name = Path(image_path).stem

            if 'json' in args.format:
                extractor.save_json(result, str(output_dir / f"{base_name}.json"))

            if 'csv' in args.format:
                extractor.save_csv(result, str(output_dir / f"{base_name}.csv"))

            if 'txt' in args.format:
                extractor.save_summary(result, str(output_dir / f"{base_name}.txt"))

            if 'vis' in args.format:
                import cv2
                vis_image = extractor._create_visualization(
                    cv2.imread(image_path),
                    result.results
                )
                vis_path = output_dir / f"{base_name}_annotated.jpg"
                cv2.imwrite(str(vis_path), vis_image)
                print(f"✓ Saved visualization to: {vis_path}")

            if not args.quiet:
                print(f"\n{'='*70}")
                print("COMPLETE")
                print(f"{'='*70}\n")

        except Exception as e:
            print(f"Error processing image: {e}")
            return 1

    else:
        # Batch mode
        try:
            results = extractor.process_batch(
                valid_images,
                output_dir=args.output,
                formats=args.format
            )

            if not args.quiet:
                print(f"\n✓ Successfully processed {len(results)}/{len(valid_images)} images")
                print(f"✓ Results saved to: {args.output}\n")

        except Exception as e:
            print(f"Error in batch processing: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
