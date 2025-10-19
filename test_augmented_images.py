#!/usr/bin/env python3
"""
Test Highlight Detection on Augmented Images
Verify robustness to data augmentation
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from test_optimized_detection import test_optimized_detection


if __name__ == "__main__":
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)

    print("\n" + "=" * 70)
    print("TESTING ON AUGMENTED IMAGES")
    print("=" * 70 + "\n")

    # Test on augmented images only
    metrics = test_optimized_detection(
        use_orig_only=False,  # Include augmented images
        num_samples=100       # Test more samples
    )

    print("\n" + "=" * 70)
    print("AUGMENTED IMAGE TEST COMPLETE")
    print("=" * 70 + "\n")
