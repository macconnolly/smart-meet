from typing import Dict, Any, Optional
import numpy as np
import logging
from dataclasses import dataclass
import json

from .temporal import TemporalDimensionExtractor
from .emotional import EmotionalDimensionExtractor
from .social import SocialDimensionExtractor
from .causal import CausalDimensionExtractor
from .evolutionary import EvolutionaryDimensionExtractor

logger = logging.getLogger(__name__)


@dataclass
class CognitiveDimensions:
    """
    Represents the 16D cognitive dimensions.

    Structure:
    - Temporal (4D): urgency, deadline_proximity, sequence_position, duration_relevance
    - Emotional (3D): polarity, intensity, confidence
    - Social (3D): authority, influence, team_dynamics
    - Causal (3D): dependencies, impact, risk_factors
    - Evolutionary (3D): change_rate, innovation_level, adaptation_need
    """
    # Temporal dimensions
    urgency: float
    deadline_proximity: float
    sequence_position: float
    duration_relevance: float

    # Emotional dimensions
    polarity: float
    intensity: float
    confidence: float

    # Social dimensions
    authority: float
    influence: float
    team_dynamics: float

    # Causal dimensions
    dependencies: float
    impact: float
    risk_factors: float

    # Evolutionary dimensions
    change_rate: float
    innovation_level: float
    adaptation_need: float

    def to_array(self) -> np.ndarray:
        """Convert to 16D numpy array."""
        return np.array([
            # Temporal (4D)
            self.urgency,
            self.deadline_proximity,
            self.sequence_position,
            self.duration_relevance,
            # Emotional (3D)
            self.polarity,
            self.intensity,
            self.confidence,
            # Social (3D)
            self.authority,
            self.influence,
            self.team_dynamics,
            # Causal (3D)
            self.dependencies,
            self.impact,
            self.risk_factors,
            # Evolutionary (3D)
            self.change_rate,
            self.innovation_level,
            self.adaptation_need
        ])

    @classmethod
    def from_array(cls, array: np.ndarray) -> 'CognitiveDimensions':
        """Create from 16D numpy array."""
        if array.shape != (16,):
            raise ValueError(f"Array must be 16D, got {array.shape}")
        return cls(
            # Temporal
            urgency=float(array[0]),
            deadline_proximity=float(array[1]),
            sequence_position=float(array[2]),
            duration_relevance=float(array[3]),
            # Emotional
            polarity=float(array[4]),
            intensity=float(array[5]),
            confidence=float(array[6]),
            # Social
            authority=float(array[7]),
            influence=float(array[8]),
            team_dynamics=float(array[9]),
            # Causal
            dependencies=float(array[10]),
            impact=float(array[11]),
            risk_factors=float(array[12]),
            # Evolutionary
            change_rate=float(array[13]),
            innovation_level=float(array[14]),
            adaptation_need=float(array[15])
        )

    def to_dict(self) -> str:
        """Convert to JSON string for storage."""
        data = {
            'temporal': {
                'urgency': self.urgency,
                'deadline_proximity': self.deadline_proximity,
                'sequence_position': self.sequence_position,
                'duration_relevance': self.duration_relevance
            },
            'emotional': {
                'polarity': self.polarity,
                'intensity': self.intensity,
                'confidence': self.confidence
            },
            'social': {
                'authority': self.authority,
                'influence': self.influence,
                'team_dynamics': self.team_dynamics
            },
            'causal': {
                'dependencies': self.dependencies,
                'impact': self.impact,
                'risk_factors': self.risk_factors
            },
            'evolutionary': {
                'change_rate': self.change_rate,
                'innovation_level': self.innovation_level,
                'adaptation_need': self.adaptation_need
            }
        }
        return json.dumps(data)

    @classmethod
    def from_dict(cls, json_str: str) -> 'CognitiveDimensions':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(
            # Temporal
            urgency=data['temporal']['urgency'],
            deadline_proximity=data['temporal']['deadline_proximity'],
            sequence_position=data['temporal']['sequence_position'],
            duration_relevance=data['temporal']['duration_relevance'],
            # Emotional
            polarity=data['emotional']['polarity'],
            intensity=data['emotional']['intensity'],
            confidence=data['emotional']['confidence'],
            # Social
            authority=data['social']['authority'],
            influence=data['social']['influence'],
            team_dynamics=data['social']['team_dynamics'],
            # Causal
            dependencies=data['causal']['dependencies'],
            impact=data['causal']['impact'],
            risk_factors=data['causal']['risk_factors'],
            # Evolutionary
            change_rate=data['evolutionary']['change_rate'],
            innovation_level=data['evolutionary']['innovation_level'],
            adaptation_need=data['evolutionary']['adaptation_need']
        )


@dataclass
class DimensionExtractionContext:
    """Context for dimension extraction."""
    timestamp_ms: Optional[int] = None
    speaker: Optional[str] = None
    speaker_role: Optional[str] = None
    content_type: Optional[str] = None
    project_id: Optional[str] = None
    meeting_type: Optional[str] = None
    linked_memories: Optional[list] = None
    previous_versions: Optional[list] = None
    current_memory_index: Optional[int] = None
    total_memories: Optional[int] = None


class DimensionAnalyzer:
    """
    Orchestrates the extraction of all 16 cognitive dimensions.

    Dimensions (16D):
    - Temporal (4D)
    - Emotional (3D)
    - Social (3D)
    - Causal (3D)
    - Evolutionary (3D)
    """

    def __init__(self):
        self.temporal_extractor = TemporalDimensionExtractor()
        self.emotional_extractor = EmotionalDimensionExtractor()
        self.social_extractor = SocialDimensionExtractor()
        self.causal_extractor = CausalDimensionExtractor()
        self.evolutionary_extractor = EvolutionaryDimensionExtractor()

    async def analyze(self, text: str, context: DimensionExtractionContext) -> CognitiveDimensions:
        """
        Analyzes text and extracts all 16 cognitive dimensions.

        Args:
            text: The input text to analyze.
            context: DimensionExtractionContext with extraction parameters.

        Returns:
            CognitiveDimensions object containing all 16 dimensions.
        """
        # Convert context to dict for extractors
        context_dict = {
            'timestamp_ms': context.timestamp_ms,
            'speaker': context.speaker,
            'speaker_role': context.speaker_role,
            'content_type': context.content_type,
            'project_id': context.project_id,
            'meeting_type': context.meeting_type,
            'linked_memories': context.linked_memories,
            'previous_versions': context.previous_versions,
            'current_memory_index': context.current_memory_index,
            'total_memories': context.total_memories
        }

        # Extract dimensions from each category
        temporal_dims = self.temporal_extractor.extract(text, context_dict)
        emotional_dims = self.emotional_extractor.extract(text, context_dict)
        social_dims = self.social_extractor.extract(text, context_dict)
        causal_dims = self.causal_extractor.extract(text, context_dict)
        evolutionary_dims = self.evolutionary_extractor.extract(text, context_dict)

        # Create CognitiveDimensions object
        return CognitiveDimensions(
            # Temporal
            urgency=float(temporal_dims[0]),
            deadline_proximity=float(temporal_dims[1]),
            sequence_position=float(temporal_dims[2]),
            duration_relevance=float(temporal_dims[3]),
            # Emotional
            polarity=float(emotional_dims[0]),
            intensity=float(emotional_dims[1]),
            confidence=float(emotional_dims[2]),
            # Social
            authority=float(social_dims[0]),
            influence=float(social_dims[1]),
            team_dynamics=float(social_dims[2]),
            # Causal
            dependencies=float(causal_dims[0]),
            impact=float(causal_dims[1]),
            risk_factors=float(causal_dims[2]),
            # Evolutionary
            change_rate=float(evolutionary_dims[0]),
            innovation_level=float(evolutionary_dims[1]),
            adaptation_need=float(evolutionary_dims[2])
        )


# Singleton instance
_dimension_analyzer_instance: Optional[DimensionAnalyzer] = None


def get_dimension_analyzer() -> DimensionAnalyzer:
    """Get or create the global dimension analyzer instance."""
    global _dimension_analyzer_instance

    if _dimension_analyzer_instance is None:
        _dimension_analyzer_instance = DimensionAnalyzer()

    return _dimension_analyzer_instance


if __name__ == "__main__":
    import asyncio
    from datetime import datetime

    async def test_analyzer():
        analyzer = DimensionAnalyzer()

        # Test cases
        print("--- Dimension Analyzer Tests ---")

        # Test 1: Basic analysis
        text1 = "This is an urgent and positive message about the project deadline."
        context1 = DimensionExtractionContext(content_type="context")
        dims1 = await analyzer.analyze(text1, context1)
        print(f"'{text1}' -> Urgency: {dims1.urgency}, Polarity: {dims1.polarity}")
        assert dims1.urgency == 1.0  # Urgency
        assert dims1.polarity > 0.5  # Positive polarity

        # Test 2: With context for temporal
        text2 = "Final discussion point for today."
        context2 = DimensionExtractionContext(
            current_memory_index=9,
            total_memories=10,
            content_type="context"
        )
        dims2 = await analyzer.analyze(text2, context2)
        print(f"'{text2}' (with context) -> Sequence: {dims2.sequence_position}")
        assert dims2.sequence_position == 0.9  # Sequence position

        # Test 3: Empty text
        text3 = ""
        context3 = DimensionExtractionContext(content_type="context")
        dims3 = await analyzer.analyze(text3, context3)
        print(f"'{text3}' -> Urgency: {dims3.urgency}, Polarity: {dims3.polarity}")

        # Test 4: JSON serialization
        json_str = dims1.to_dict()
        print(f"JSON: {json_str[:100]}...")
        dims_restored = CognitiveDimensions.from_dict(json_str)
        assert dims_restored.urgency == dims1.urgency

        print("\nAll dimension analyzer tests passed!")

    asyncio.run(test_analyzer())
