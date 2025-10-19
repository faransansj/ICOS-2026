#!/usr/bin/env python3
"""
Dataset Generation Script
Run this script to generate the complete synthetic highlight dataset
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator.dataset_builder import DatasetBuilder


def main():
    """Main dataset generation pipeline"""
    print("\n" + "="*70)
    print(" SYNTHETIC HIGHLIGHT TEXT DATASET GENERATOR")
    print("="*70 + "\n")

    # Configuration
    config = {
        'output_base_dir': 'data',
        'num_base_images': 200,
        'num_augmentations': 2,
        'val_ratio': 0.3,
        'test_ratio': 0.7,
        'colors': ['yellow', 'green', 'pink']
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Initialize builder
    builder = DatasetBuilder(
        output_base_dir=config['output_base_dir'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio']
    )

    # Build dataset
    try:
        dataset_info = builder.build_complete_dataset(
            num_base_images=config['num_base_images'],
            num_augmentations=config['num_augmentations'],
            colors=config['colors']
        )

        print("\n" + "="*70)
        print(" DATASET GENERATION SUCCESSFUL!")
        print("="*70 + "\n")

        print("Dataset structure:")
        print(f"  {config['output_base_dir']}/")
        print(f"  ├── synthetic/        (base {config['num_base_images']} images)")
        print(f"  ├── validation/       ({len(dataset_info['validation_annotations'])} images)")
        print(f"  ├── test/            ({len(dataset_info['test_annotations'])} images)")
        print(f"  └── dataset_statistics.json")
        print()

        print("Next steps:")
        print("  1. Review dataset statistics: data/dataset_statistics.json")
        print("  2. Inspect sample images in data/validation/ and data/test/")
        print("  3. Begin implementing highlight detection module (Week 3)")
        print()

        return 0

    except Exception as e:
        print(f"\n❌ Error during dataset generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
