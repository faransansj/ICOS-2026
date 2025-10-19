"""
Data Generator Package
Synthetic dataset generation for highlight text extraction research
"""

from .text_image_generator import TextImageGenerator
from .highlight_overlay import HighlightOverlay

__all__ = ['TextImageGenerator', 'HighlightOverlay']
