"""
Dimension extractors for cognitive features.

This module provides extractors for the 16D cognitive dimensions:
- Temporal (4D): urgency, deadline_proximity, sequence_position, duration_relevance
- Emotional (3D): polarity, intensity, confidence
- Social (3D): authority, influence, team_dynamics
- Causal (3D): dependencies, impact, risk_factors
- Evolutionary (3D): change_rate, innovation_level, adaptation_need
"""

from .temporal_extractor import TemporalDimensionExtractor, TemporalFeatures
from .emotional_extractor import EmotionalDimensionExtractor, EmotionalFeatures
from .social_extractor import SocialDimensionExtractor, SocialFeatures
from .causal_extractor import CausalDimensionExtractor, CausalFeatures
from .evolutionary_extractor import EvolutionaryDimensionExtractor, EvolutionaryFeatures
from .dimension_analyzer import (
    DimensionAnalyzer,
    CognitiveDimensions,
    DimensionExtractionContext,
    get_dimension_analyzer
)
from .dimension_cache import DimensionCache, get_dimension_cache

__all__ = [
    # Extractors
    "TemporalDimensionExtractor",
    "EmotionalDimensionExtractor",
    "SocialDimensionExtractor",
    "CausalDimensionExtractor",
    "EvolutionaryDimensionExtractor",
    # Feature classes
    "TemporalFeatures",
    "EmotionalFeatures",
    "SocialFeatures",
    "CausalFeatures",
    "EvolutionaryFeatures",
    # Analyzer
    "DimensionAnalyzer",
    "CognitiveDimensions",
    "DimensionExtractionContext",
    "get_dimension_analyzer",
    # Cache
    "DimensionCache",
    "get_dimension_cache",
]