"""
Phorcys: A Turkish NLP Project

This package provides advanced text processing and AI-based analysis tools for Turkish language.
"""

__version__ = "0.1.0"
__author__ = "Mesut Aktaş"

from .Abyss import Abyss
from .AIProcessText import ProcessWithAI
from .GenerateText import TextGenerator
from .Model import ModelGen
from .ProcessMedia import MediaToText, OcrProcessor, PdfProcessor
from .ProcessText import Normalize, ProcessText

__all__ = [
    'Abyss',
    'ProcessWithAI',
    'TextGenerator',
    'ModelGen',
    'MediaToText',
    'OcrProcessor',
    'PdfProcessor',
    'Normalize',
    'ProcessText'
]
