"""
Evolutionary dimension extractor for cognitive features.

This module extracts 3 evolutionary dimensions from memory content:
- Change rate: How quickly things are changing (0-1)
- Innovation level: Degree of new ideas/approaches (0-1)
- Adaptation need: Need for adaptation/evolution (0-1)

NOTE: This is a placeholder implementation that returns default values.
Full implementation will analyze patterns of change, innovation indicators, etc.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvolutionaryFeatures:
    """Extracted evolutionary features."""
    change_rate: float  # 0-1: Rate of change
    innovation_level: float  # 0-1: Degree of innovation
    adaptation_need: float  # 0-1: Need for adaptation
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.change_rate,
            self.innovation_level,
            self.adaptation_need
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'EvolutionaryFeatures':
        """Create from numpy array."""
        if array.shape != (3,):
            raise ValueError(f"Array must be 3D, got {array.shape}")
        return cls(
            change_rate=float(array[0]),
            innovation_level=float(array[1]),
            adaptation_need=float(array[2])
        )


class EvolutionaryDimensionExtractor:
    """
    Extracts evolutionary dimensions from memory content.
    
    PLACEHOLDER IMPLEMENTATION: Returns default values of 0.5 for all dimensions.
    
    Future implementation will analyze:
    - Change indicators and patterns
    - Innovation signals
    - Adaptation requirements
    - Evolution of ideas over time
    - Transformation indicators
    """
    
    def __init__(self):
        """Initialize the evolutionary extractor."""
        logger.info("EvolutionaryDimensionExtractor initialized (placeholder implementation)")
    
    def extract(
        self,
        content: str,
        content_type: Optional[str] = None,
        timestamp_ms: Optional[int] = None,
        previous_versions: Optional[List[str]] = None
    ) -> EvolutionaryFeatures:
        """
        Extract evolutionary dimensions from memory content.
        
        Args:
            content: Memory content text
            content_type: Type of content
            timestamp_ms: When this was created
            previous_versions: Previous versions of similar content
            
        Returns:
            EvolutionaryFeatures with 3 dimensions
        """
        content_lower = content.lower()

        # Change Rate
        change_rate_score = 0.5
        change_keywords = ["changed to", "updated", "new version", "revision", "modification", "shift", "transition", "evolved"]
        for keyword in change_keywords:
            if keyword in content_lower:
                change_rate_score += 0.1
        
        if content_type == "change" or content_type == "update":
            change_rate_score += 0.1
        if previous_versions and len(previous_versions) > 0:
            change_rate_score += 0.1 # Indicates a history of changes

        # Innovation Level
        innovation_level_score = 0.5
        innovation_keywords = ["new idea", "innovative", "breakthrough", "unique approach", "novel", "creative solution", "pioneer", "disruptive"]
        for keyword in innovation_keywords:
            if keyword in content_lower:
                innovation_level_score += 0.1
        
        if content_type == "innovation" or content_type == "idea":
            innovation_level_score += 0.1

        # Adaptation Need
        adaptation_need_score = 0.5
        adaptation_keywords = ["adapt to", "adjust to", "respond to feedback", "evolve", "pivot", "re-evaluate", "flexible", "learn from"]
        for keyword in adaptation_keywords:
            if keyword in content_lower:
                adaptation_need_score += 0.1
        
        if content_type == "issue" or content_type == "feedback":
            adaptation_need_score += 0.1

        # Normalize scores to be within [0, 1]
        change_rate_score = np.clip(change_rate_score, 0.0, 1.0)
        innovation_level_score = np.clip(innovation_level_score, 0.0, 1.0)
        adaptation_need_score = np.clip(adaptation_need_score, 0.0, 1.0)
            
        return EvolutionaryFeatures(
            change_rate=float(change_rate_score),
            innovation_level=float(innovation_level_score),
            adaptation_need=float(adaptation_need_score)
        )
    
    def batch_extract(
        self,
        contents: List[str],
        content_types: Optional[List[str]] = None,
        timestamps_ms: Optional[List[int]] = None,
        previous_versions_lists: Optional[List[List[str]]] = None
    ) -> np.ndarray:
        """
        Extract evolutionary features for multiple memories.
        
        Args:
            contents: List of memory contents
            content_types: List of content types
            timestamps_ms: List of timestamps
            previous_versions_lists: List of previous version lists
            
        Returns:
            Array of shape (n_memories, 3)
        """
        n_memories = len(contents)
        
        # Prepare lists with None if not provided
        if content_types is None:
            content_types = [None] * n_memories
        if timestamps_ms is None:
            timestamps_ms = [None] * n_memories
        if previous_versions_lists is None:
            previous_versions_lists = [None] * n_memories
        
        # Extract features
        features = []
        for i in range(n_memories):
            evolutionary_features = self.extract(
                content=contents[i],
                content_type=content_types[i],
                timestamp_ms=timestamps_ms[i],
                previous_versions=previous_versions_lists[i]
            )
            features.append(evolutionary_features.to_array())
        
        return np.vstack(features)