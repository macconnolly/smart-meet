"""
Unit tests for emotional dimension extractor.
"""

import pytest
import numpy as np

from src.extraction.dimensions.emotional_extractor import (
    EmotionalDimensionExtractor,
    EmotionalFeatures
)


class TestEmotionalDimensionExtractor:
    """Test emotional dimension extraction."""
    
    @pytest.fixture
    def extractor(self):
        """Create emotional extractor instance."""
        return EmotionalDimensionExtractor()
    
    def test_extract_polarity(self, extractor):
        """Test sentiment polarity extraction."""
        # Positive sentiment
        content = "I absolutely love this amazing idea!"
        features = extractor.extract(content)
        assert features.polarity > 0.7  # Positive
        
        # Negative sentiment
        content = "This is terrible and completely disappointing."
        features = extractor.extract(content)
        assert features.polarity < 0.3  # Negative
        
        # Neutral sentiment
        content = "The meeting is scheduled for 3 PM."
        features = extractor.extract(content)
        assert 0.4 <= features.polarity <= 0.6  # Neutral
    
    def test_extract_intensity(self, extractor):
        """Test emotional intensity extraction."""
        # High intensity
        content = "I'm EXTREMELY excited and absolutely thrilled!!!"
        features = extractor.extract(content)
        assert features.intensity >= 0.8
        
        # Low intensity
        content = "Okay, that's fine."
        features = extractor.extract(content)
        assert features.intensity <= 0.3
        
        # Medium intensity with modifiers
        content = "I'm quite happy with the results."
        features = extractor.extract(content)
        assert 0.4 <= features.intensity <= 0.7
    
    def test_extract_confidence(self, extractor):
        """Test confidence level extraction."""
        # High confidence
        content = "I'm absolutely certain this will work."
        features = extractor.extract(content)
        assert features.confidence >= 0.8
        
        # Low confidence
        content = "Maybe we could possibly try this approach?"
        features = extractor.extract(content)
        assert features.confidence <= 0.4
        
        # Medium confidence
        content = "I think this is a good solution."
        features = extractor.extract(content)
        assert 0.5 <= features.confidence <= 0.7
    
    def test_content_type_influence(self, extractor):
        """Test how content type affects confidence."""
        content = "This is our conclusion."
        
        # Decisions have higher default confidence
        features_decision = extractor.extract(content, content_type="decision")
        features_normal = extractor.extract(content)
        assert features_decision.confidence >= features_normal.confidence
        
        # Questions have lower default confidence
        features_question = extractor.extract(content, content_type="question")
        assert features_question.confidence < features_normal.confidence
    
    def test_hedging_language(self, extractor):
        """Test detection of hedging language."""
        # Strong hedging reduces confidence
        content = "If we assume everything goes well, this might work."
        features = extractor.extract(content)
        assert features.confidence < 0.5
        
        # Definitive language increases confidence
        content = "This will definitely improve performance."
        features = extractor.extract(content)
        assert features.confidence > 0.7
    
    def test_feature_ranges(self, extractor):
        """Test that all features are in valid range [0, 1]."""
        test_contents = [
            "I absolutely HATE this terrible idea!!!",
            "This is fantastic and amazing!",
            "I'm not sure about this.",
            "The data clearly shows improvement.",
            "Maybe we should consider alternatives?"
        ]
        
        for content in test_contents:
            features = extractor.extract(content)
            array = features.to_array()
            
            # Check all values are in [0, 1]
            assert np.all(array >= 0)
            assert np.all(array <= 1)
            assert array.shape == (3,)
    
    def test_to_from_array(self, extractor):
        """Test array conversion methods."""
        features = EmotionalFeatures(
            polarity=0.8,
            intensity=0.6,
            confidence=0.7
        )
        
        # Convert to array
        array = features.to_array()
        assert array.shape == (3,)
        assert array[0] == 0.8
        assert array[1] == 0.6
        assert array[2] == 0.7
        
        # Convert back from array
        features_restored = EmotionalFeatures.from_array(array)
        assert features_restored.polarity == features.polarity
        assert features_restored.intensity == features.intensity
        assert features_restored.confidence == features.confidence
    
    def test_vader_integration(self, extractor):
        """Test VADER sentiment analysis integration."""
        # VADER should handle complex sentiment
        content = "The project isn't bad, but it could be much better."
        features = extractor.extract(content)
        
        # Mixed sentiment should be near neutral
        assert 0.3 <= features.polarity <= 0.7
        
        # Should have some intensity due to comparison
        assert features.intensity > 0.2
    
    def test_batch_extract(self, extractor):
        """Test batch extraction."""
        contents = [
            "This is amazing!",
            "I'm not sure about this.",
            "Terrible idea, absolutely hate it."
        ]
        
        features_array = extractor.batch_extract(contents)
        
        assert features_array.shape == (3, 3)
        assert np.all(features_array >= 0)
        assert np.all(features_array <= 1)
        
        # First should be positive
        assert features_array[0, 0] > 0.6
        # Last should be negative
        assert features_array[2, 0] < 0.4