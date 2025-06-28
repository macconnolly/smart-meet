from typing import Dict, Any, Optional
import numpy as np
import logging

from .temporal import TemporalDimensionExtractor
from .emotional import EmotionalDimensionExtractor

logger = logging.getLogger(__name__)


class DimensionAnalyzer:
    """
    Orchestrates the extraction of all 16 cognitive dimensions.

    Dimensions (16D):
    - Temporal (4D)
    - Emotional (3D)
    - Social (3D) - Placeholder
    - Causal (3D) - Placeholder
    - Strategic (3D) - Placeholder
    """

    def __init__(self):
        self.temporal_extractor = TemporalDimensionExtractor()
        self.emotional_extractor = EmotionalDimensionExtractor()

    def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Analyzes text and extracts all 16 cognitive dimensions.

        Args:
            text: The input text to analyze.
            context: Optional dictionary with additional context for extractors.

        Returns:
            A 16D numpy array representing the cognitive dimensions.
        """
        temporal_dims = self.temporal_extractor.extract(text, context)
        emotional_dims = self.emotional_extractor.extract(text, context)

        # Placeholder for Social, Causal, Strategic dimensions (9D)
        # As per IMPLEMENTATION_GUIDE.md, these should return 0.5 for now.
        social_dims = np.array([0.5, 0.5, 0.5])
        causal_dims = np.array([0.5, 0.5, 0.5])
        strategic_dims = np.array([0.5, 0.5, 0.5])

        # Concatenate all dimensions to form a 16D vector
        cognitive_dimensions = np.concatenate(
            [temporal_dims, emotional_dims, social_dims, causal_dims, strategic_dims]
        )

        if cognitive_dimensions.shape != (16,):
            logger.error(f"Dimension mismatch: Expected 16D, got {cognitive_dimensions.shape}")
            # Attempt to resize or pad to 16D if possible, or raise error
            # For now, raise an error to indicate a critical issue
            raise ValueError(
                f"Concatenated cognitive dimensions are not 16D: {cognitive_dimensions.shape}"
            )

        return cognitive_dimensions


if __name__ == "__main__":
    analyzer = DimensionAnalyzer()

    # Test cases
    print("--- Dimension Analyzer Tests ---")

    # Test 1: Basic analysis
    text1 = "This is an urgent and positive message about the project deadline."
    dims1 = analyzer.analyze(text1)
    print(f"'{text1}' -> Dimensions: {dims1}")
    assert dims1.shape == (16,)
    assert dims1[0] == 1.0  # Urgency
    assert dims1[4] > 0.5  # Positive polarity
    assert np.allclose(dims1[7:], 0.5)  # Placeholders

    # Test 2: With context for temporal
    text2 = "Final discussion point for today."
    context2 = {"current_time": datetime.now(), "current_memory_index": 9, "total_memories": 10}
    dims2 = analyzer.analyze(text2, context2)
    print(f"'{text2}' (with context) -> Dimensions: {dims2}")
    assert dims2.shape == (16,)
    assert dims2[2] == 0.9  # Sequence position

    # Test 3: Empty text
    text3 = ""
    dims3 = analyzer.analyze(text3)
    print(f"'{text3}' -> Dimensions: {dims3}")
    assert dims3.shape == (16,)
    assert np.allclose(dims3[[0, 1, 3, 5]], [0.0, 0.0, 0.0, 0.0]) # Urgency, deadline, duration, intensity should be 0
    assert np.allclose(dims3[[2, 4, 6]], [0.5, 0.5, 0.5]) # Sequence, polarity, confidence should be 0.5
    assert np.allclose(dims3[7:], 0.5) # Placeholders

    print("\nAll dimension analyzer tests passed!")
