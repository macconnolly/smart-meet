"""
Activation spreading engines for cognitive memory retrieval.

This module contains:
- BasicActivationEngine: BFS-based activation spreading
- ConsultingActivationEngine: Enhanced activation with project context (to be implemented)
"""

from .basic_activation_engine import BasicActivationEngine, ActivationResult

__all__ = ["BasicActivationEngine", "ActivationResult"]
