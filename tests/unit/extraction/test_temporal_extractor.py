"""
Unit tests for temporal dimension extractor.
"""

import pytest
import numpy as np

from src.extraction.dimensions.temporal_extractor import (
    TemporalDimensionExtractor,
    TemporalFeatures
)


class TestTemporalDimensionExtractor:
    """Test temporal dimension extraction."""
    
    @pytest.fixture
    def extractor(self):
        """Create temporal extractor instance."""
        return TemporalDimensionExtractor()
    
    def test_extract_urgency(self, extractor):
        """Test urgency extraction from content."""
        # High urgency
        content = "This is CRITICAL and needs to be done ASAP!"
        features = extractor.extract(content)
        assert features.urgency >= 0.9
        
        # Medium urgency
        content = "Please complete this task by tomorrow."
        features = extractor.extract(content)
        assert 0.5 <= features.urgency <= 0.7
        
        # Low urgency
        content = "No rush, whenever you have time."
        features = extractor.extract(content)
        assert features.urgency <= 0.3
    
    def test_extract_deadline_proximity(self, extractor):
        """Test deadline proximity extraction."""
        # Close deadline
        content = "This needs to be done by end of day today."
        features = extractor.extract(content)
        assert features.deadline_proximity >= 0.7
        
        # Medium deadline
        content = "Due within 2 weeks from now."
        features = extractor.extract(content)
        assert 0.3 <= features.deadline_proximity <= 0.7
        
        # No deadline
        content = "This is a general discussion about the project."
        features = extractor.extract(content)
        assert features.deadline_proximity == 0.0
    
    def test_extract_sequence_position(self, extractor):
        """Test sequence position extraction."""
        # Beginning of meeting
        features = extractor.extract(
            "Let's get started",
            timestamp_ms=1000,
            meeting_duration_ms=60000
        )
        assert features.sequence_position < 0.1
        
        # Middle of meeting
        features = extractor.extract(
            "Moving on to the next topic",
            timestamp_ms=30000,
            meeting_duration_ms=60000
        )
        assert 0.4 <= features.sequence_position <= 0.6
        
        # End of meeting
        features = extractor.extract(
            "To summarize our discussion",
            timestamp_ms=55000,
            meeting_duration_ms=60000
        )
        assert features.sequence_position > 0.9
    
    def test_extract_duration_relevance(self, extractor):
        """Test duration relevance extraction."""
        # Long-term relevance
        content = "This will be our strategic framework going forward."
        features = extractor.extract(content)
        assert features.duration_relevance >= 0.7
        
        # Short-term relevance
        content = "Quick temporary fix for today's demo."
        features = extractor.extract(content)
        assert features.duration_relevance <= 0.3
        
        # Medium-term relevance
        content = "Let's implement this feature next sprint."
        features = extractor.extract(content)
        assert 0.4 <= features.duration_relevance <= 0.7
    
    def test_content_type_influence(self, extractor):
        """Test how content type affects extraction."""
        content = "We need to address this issue."
        
        # Action items are more urgent
        features_action = extractor.extract(content, content_type="action")
        features_normal = extractor.extract(content)
        assert features_action.urgency >= features_normal.urgency
        
        # Policies have longer duration relevance
        features_policy = extractor.extract(content, content_type="policy")
        assert features_policy.duration_relevance >= 0.7
    
    def test_feature_ranges(self, extractor):
        """Test that all features are in valid range [0, 1]."""
        test_contents = [
            "URGENT: Critical bug in production!!!",
            "Long-term strategic planning session",
            "Quick chat about lunch plans",
            "Deadline: Submit report by January 15",
            "No specific timeline for this task"
        ]
        
        for content in test_contents:
            features = extractor.extract(content)
            array = features.to_array()
            
            # Check all values are in [0, 1]
            assert np.all(array >= 0)
            assert np.all(array <= 1)
            assert array.shape == (4,)
    
    def test_to_from_array(self, extractor):
        """Test array conversion methods."""
        features = TemporalFeatures(
            urgency=0.8,
            deadline_proximity=0.6,
            sequence_position=0.5,
            duration_relevance=0.7
        )
        
        # Convert to array
        array = features.to_array()
        assert array.shape == (4,)
        assert array[0] == 0.8
        assert array[1] == 0.6
        assert array[2] == 0.5
        assert array[3] == 0.7
        
        # Convert back from array
        features_restored = TemporalFeatures.from_array(array)
        assert features_restored.urgency == features.urgency
        assert features_restored.deadline_proximity == features.deadline_proximity
        assert features_restored.sequence_position == features.sequence_position
        assert features_restored.duration_relevance == features.duration_relevance
    
    def test_batch_extract(self, extractor):
        """Test batch extraction."""
        contents = [
            "This is urgent!",
            "Long-term strategy discussion",
            "Meeting next week"
        ]
        timestamps = [1000, 30000, 55000]
        meeting_duration = 60000
        
        features_array = extractor.batch_extract(
            contents,
            timestamps_ms=timestamps,
            meeting_duration_ms=meeting_duration
        )
        
        assert features_array.shape == (3, 4)
        assert np.all(features_array >= 0)
        assert np.all(features_array <= 1)