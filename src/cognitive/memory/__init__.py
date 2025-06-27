"""
Advanced memory systems with episodic/semantic separation and consolidation.

This module contains:
- DualMemorySystem: Complete dual memory system with consolidation
- EpisodicMemoryStore: Fast-decaying episodic memories
- SemanticMemoryStore: Slow-decaying semantic memories
- MemoryConsolidation: Automatic consolidation from episodic to semantic
"""

from .dual_memory_system import (
    DualMemorySystem,
    EpisodicMemoryStore,
    SemanticMemoryStore,
    MemoryConsolidation,
    MemoryType,
    MemoryAccessPattern,
    ContentTypeDecayProfile
)

__all__ = [
    "DualMemorySystem",
    "EpisodicMemoryStore", 
    "SemanticMemoryStore",
    "MemoryConsolidation",
    "MemoryType",
    "MemoryAccessPattern",
    "ContentTypeDecayProfile"
]
