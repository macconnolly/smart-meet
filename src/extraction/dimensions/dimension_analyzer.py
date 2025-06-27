"""
Dimension analyzer that orchestrates all dimension extractors.

This module combines all dimension extractors to produce the complete
16D cognitive feature vector for memories.

Currently implements temporal (4D) and emotional (3D) extractors,
with placeholder implementations for social (3D), causal (3D), and strategic (3D).
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .temporal_extractor import TemporalDimensionExtractor, TemporalFeatures
from .emotional_extractor import EmotionalDimensionExtractor, EmotionalFeatures

logger = logging.getLogger(__name__)


@dataclass
class SocialFeatures:
    """Placeholder social dimensions (3D)."""

    authority: float = 0.5
    influence: float = 0.5
    team_dynamics: float = 0.5

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.authority, self.influence, self.team_dynamics])

    @classmethod
    def from_array(cls, array: np.ndarray) -> "SocialFeatures":
        """Create from numpy array."""
        return cls(authority=array[0], influence=array[1], team_dynamics=array[2])


@dataclass
class CausalFeatures:
    """Placeholder causal dimensions (3D)."""

    dependencies: float = 0.5
    impact: float = 0.5
    risk_factors: float = 0.5

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.dependencies, self.impact, self.risk_factors])

    @classmethod
    def from_array(cls, array: np.ndarray) -> "CausalFeatures":
        """Create from numpy array."""
        return cls(dependencies=array[0], impact=array[1], risk_factors=array[2])


@dataclass
class StrategicFeatures:
    """Placeholder strategic dimensions (3D)."""

    alignment: float = 0.5
    innovation: float = 0.5
    value: float = 0.5

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.alignment, self.innovation, self.value])

    @classmethod
    def from_array(cls, array: np.ndarray) -> "StrategicFeatures":
        """Create from numpy array."""
        return cls(alignment=array[0], innovation=array[1], value=array[2])


@dataclass
class CognitiveDimensions:
    """Complete 16D cognitive dimensions."""

    temporal: TemporalFeatures  # 4D
    emotional: EmotionalFeatures  # 3D
    social: SocialFeatures  # 3D
    causal: CausalFeatures  # 3D
    strategic: StrategicFeatures  # 3D

    def to_array(self) -> np.ndarray:
        """
        Convert to 16D numpy array.

        Order: temporal(4) + emotional(3) + social(3) + causal(3) + strategic(3)
        """
        return np.concatenate(
            [
                self.temporal.to_array(),
                self.emotional.to_array(),
                self.social.to_array(),
                self.causal.to_array(),
                self.strategic.to_array(),
            ]
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with named dimensions."""
        array = self.to_array()
        return {
            # Temporal
            "urgency": array[0],
            "deadline_proximity": array[1],
            "sequence_position": array[2],
            "duration_relevance": array[3],
            # Emotional
            "polarity": array[4],
            "intensity": array[5],
            "confidence": array[6],
            # Social (placeholder)
            "authority": array[7],
            "influence": array[8],
            "team_dynamics": array[9],
            # Causal (placeholder)
            "dependencies": array[10],
            "impact": array[11],
            "risk_factors": array[12],
            # Strategic (placeholder)
            "alignment": array[13],
            "innovation": array[14],
            "value": array[15],
        }

    @classmethod
    def from_array(cls, array: np.ndarray) -> "CognitiveDimensions":
        """Create from 16D numpy array."""
        if array.shape != (16,):
            raise ValueError(f"Array must be 16D, got {array.shape}")

        return cls(
            temporal=TemporalFeatures.from_array(array[:4]),
            emotional=EmotionalFeatures.from_array(array[4:7]),
            social=SocialFeatures.from_array(array[7:10]),
            causal=CausalFeatures.from_array(array[10:13]),
            strategic=StrategicFeatures.from_array(array[13:16]),
        )


@dataclass
class DimensionExtractionContext:
    """Context information for dimension extraction."""

    timestamp_ms: Optional[int] = None
    meeting_duration_ms: Optional[int] = None
    speaker: Optional[str] = None
    speaker_role: Optional[str] = None
    participants: Optional[List[str]] = None
    content_type: Optional[str] = None
    project_id: Optional[str] = None
    project_type: Optional[str] = None
    linked_memories: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class DimensionAnalyzer:
    """
    Orchestrates all dimension extractors to produce 16D cognitive features.

    Currently implemented:
    - Temporal (4D): Fully implemented with urgency, deadlines, sequence, duration
    - Emotional (3D): Fully implemented with VADER + custom patterns
    - Social (3D): Placeholder implementation (returns 0.5 for all)
    - Causal (3D): Placeholder implementation (returns 0.5 for all)
    - Strategic (3D): Placeholder implementation (returns 0.5 for all)
    """

    def __init__(self, use_parallel: bool = True):
        """
        Initialize the dimension analyzer.

        Args:
            use_parallel: Whether to extract dimensions in parallel
        """
        # Initialize implemented extractors
        self.temporal_extractor = TemporalDimensionExtractor()
        self.emotional_extractor = EmotionalDimensionExtractor()

        # Thread pool for parallel extraction (only 2 real extractors for now)
        self.use_parallel = use_parallel
        if use_parallel:
            self._executor = ThreadPoolExecutor(max_workers=2)

        logger.info("DimensionAnalyzer initialized with temporal and emotional extractors")

    async def analyze(
        self, content: str, context: Optional[DimensionExtractionContext] = None
    ) -> CognitiveDimensions:
        """
        Analyze content to extract all cognitive dimensions.

        Args:
            content: Memory content text
            context: Extraction context with metadata

        Returns:
            CognitiveDimensions with all 16 features
        """
        if context is None:
            context = DimensionExtractionContext()

        if self.use_parallel:
            # Extract real dimensions in parallel
            dimensions = await self._analyze_parallel(content, context)
        else:
            # Extract dimensions sequentially
            dimensions = self._analyze_sequential(content, context)

        # Validate dimensions
        self._validate_dimensions(dimensions)

        return dimensions

    async def _analyze_parallel(
        self, content: str, context: DimensionExtractionContext
    ) -> CognitiveDimensions:
        """Extract dimensions in parallel for performance."""
        loop = asyncio.get_event_loop()

        # Create extraction tasks for implemented extractors
        tasks = [
            loop.run_in_executor(
                self._executor,
                self.temporal_extractor.extract,
                content,
                context.timestamp_ms,
                context.meeting_duration_ms,
                context.speaker,
                context.content_type,
            ),
            loop.run_in_executor(
                self._executor,
                self.emotional_extractor.extract,
                content,
                context.speaker,
                context.content_type,
            ),
        ]

        # Wait for real extractions
        results = await asyncio.gather(*tasks)

        # Add placeholder dimensions
        return CognitiveDimensions(
            temporal=results[0],
            emotional=results[1],
            social=SocialFeatures(),  # Placeholder
            causal=CausalFeatures(),  # Placeholder
            strategic=StrategicFeatures(),  # Placeholder
        )

    def _analyze_sequential(
        self, content: str, context: DimensionExtractionContext
    ) -> CognitiveDimensions:
        """Extract dimensions sequentially."""
        temporal = self.temporal_extractor.extract(
            content,
            context.timestamp_ms,
            context.meeting_duration_ms,
            context.speaker,
            context.content_type,
        )

        emotional = self.emotional_extractor.extract(content, context.speaker, context.content_type)

        # Add placeholder dimensions
        return CognitiveDimensions(
            temporal=temporal,
            emotional=emotional,
            social=SocialFeatures(),  # Placeholder
            causal=CausalFeatures(),  # Placeholder
            strategic=StrategicFeatures(),  # Placeholder
        )

    def _validate_dimensions(self, dimensions: CognitiveDimensions) -> None:
        """Validate that all dimensions are in valid range."""
        array = dimensions.to_array()

        # Check range [0, 1]
        if np.any(array < 0) or np.any(array > 1):
            invalid_dims = []
            dim_dict = dimensions.to_dict()
            for name, value in dim_dict.items():
                if value < 0 or value > 1:
                    invalid_dims.append(f"{name}={value}")

            logger.warning(f"Dimensions outside [0,1] range: {invalid_dims}")

            # Clip to valid range
            array = np.clip(array, 0, 1)

            # Update dimensions
            clipped = CognitiveDimensions.from_array(array)
            dimensions.temporal = clipped.temporal
            dimensions.emotional = clipped.emotional
            dimensions.social = clipped.social
            dimensions.causal = clipped.causal
            dimensions.strategic = clipped.strategic

    async def batch_analyze(
        self, contents: List[str], contexts: Optional[List[DimensionExtractionContext]] = None
    ) -> np.ndarray:
        """
        Analyze multiple contents in batch.

        Args:
            contents: List of memory contents
            contexts: List of extraction contexts

        Returns:
            Array of shape (n_memories, 16)
        """
        n_memories = len(contents)

        if contexts is None:
            contexts = [DimensionExtractionContext() for _ in range(n_memories)]
        elif len(contexts) != n_memories:
            raise ValueError("Number of contents and contexts must match")

        # Process in batches for efficiency
        all_dimensions = []

        for content, context in zip(contents, contexts):
            dimensions = await self.analyze(content, context)
            all_dimensions.append(dimensions.to_array())

        return np.vstack(all_dimensions)

    def get_dimension_statistics(self, dimensions_array: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each dimension.

        Args:
            dimensions_array: Array of shape (n_memories, 16)

        Returns:
            Dictionary with stats for each dimension
        """
        if dimensions_array.shape[1] != 16:
            raise ValueError("Array must have 16 columns")

        dimension_names = [
            # Temporal (implemented)
            "urgency",
            "deadline_proximity",
            "sequence_position",
            "duration_relevance",
            # Emotional (implemented)
            "polarity",
            "intensity",
            "confidence",
            # Social (placeholder)
            "authority",
            "influence",
            "team_dynamics",
            # Causal (placeholder)
            "dependencies",
            "impact",
            "risk_factors",
            # Strategic (placeholder)
            "alignment",
            "innovation",
            "value",
        ]

        stats = {}
        for i, name in enumerate(dimension_names):
            values = dimensions_array[:, i]
            stats[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }

        return stats

    def close(self):
        """Clean up resources."""
        if self.use_parallel and hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)


# Singleton instance
_analyzer_instance = None


def get_dimension_analyzer() -> DimensionAnalyzer:
    """Get singleton dimension analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = DimensionAnalyzer()
    return _analyzer_instance


# Example usage and testing
async def test_analyzer():
    """Test the dimension analyzer."""
    analyzer = get_dimension_analyzer()

    test_cases = [
        "We need to implement this feature ASAP for the client demo tomorrow!",
        "I think the long-term strategy should focus on sustainable growth.",
        "The quarterly results show positive trends across all metrics.",
    ]

    for i, content in enumerate(test_cases):
        print(f"\nTest case {i+1}: {content}")

        context = DimensionExtractionContext(
            timestamp_ms=i * 60000,  # Every minute
            meeting_duration_ms=180000,  # 3 minutes
            content_type="insight" if i == 1 else "decision",
        )

        dimensions = await analyzer.analyze(content, context)
        print(f"Dimensions: {dimensions.to_dict()}")


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_analyzer())
