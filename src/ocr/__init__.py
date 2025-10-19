"""
OCR Module
Tesseract-based text extraction from highlight regions
"""

from .ocr_engine import OCREngine, OCRResult
from .evaluator import OCREvaluator

__all__ = ['OCREngine', 'OCRResult', 'OCREvaluator']
