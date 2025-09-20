"""
ARENA Chapter 1.1: Transformer from Scratch

Educational implementation of transformer architecture for mechanistic interpretability.
"""

__version__ = "1.0.0"
__author__ = "ARENA Student"
__description__ = "Educational Transformer Implementation for Mechanistic Interpretability"

# Make imports available at package level
from .models.config import TransformerConfig
from .models.transformer import EducationalTransformer
from .interpretability.hooks import HookManager, ActivationPatcher, AblationAnalyzer

__all__ = [
    'TransformerConfig',
    'EducationalTransformer',
    'HookManager',
    'ActivationPatcher',
    'AblationAnalyzer'
]