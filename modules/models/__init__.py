"""
Models __init__.py
"""

from .base import GenerationBackend
from .registry import get_backend_class

__all__ = ["GenerationBackend"]