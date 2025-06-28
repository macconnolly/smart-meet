"""
Embedding and vector management for cognitive vectors.

This module provides:
- ONNX encoder for semantic embeddings (384D)
- Vector manager for composing/decomposing 400D vectors
- Vector validation utilities
- Embedding engine for batch processing
"""

from .onnx_encoder import ONNXEncoder, get_encoder
from .vector_manager import VectorManager, get_vector_manager, VectorStats
from .vector_validation import VectorValidator, get_vector_validator, ValidationResult
from .engine import EmbeddingEngine

__all__ = [
    # Encoder
    "ONNXEncoder",
    "get_onnx_encoder",
    # Vector Manager
    "VectorManager",
    "get_vector_manager",
    "VectorStats",
    # Validator
    "VectorValidator",
    "get_vector_validator",
    "ValidationResult",
    # Engine
    "EmbeddingEngine",
]