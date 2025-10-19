"""
Highlight Detector Package
HSV-based highlight region detection and evaluation
"""

from .highlight_detector import HighlightDetector
from .evaluator import HighlightEvaluator

__all__ = ['HighlightDetector', 'HighlightEvaluator']
