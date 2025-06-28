"""
Comprehensive unit tests for temporal dimension extraction.

Tests cover:
- Urgency detection
- Deadline proximity calculation  
- Sequence position tracking
- Duration relevance
- Edge cases and error handling
"""

import pytest
from datetime import datetime, timedelta
from src.extraction.dimensions.temporal_extractor import (
    TemporalDimensionExtractor, TemporalFeatures
)


class TestTemporalDimensionExtractor:
    """Test suite for temporal dimension extraction."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return TemporalDimensionExtractor()
    
    # ===== Urgency Detection Tests =====
    
    @pytest.mark.parametrize("content,expected_urgency,description", [
        # High urgency (1.0)
        ("This is URGENT!", 1.0, "Explicit urgency keyword"),
        ("We need this ASAP", 1.0, "ASAP keyword"),
        ("Critical issue requiring immediate attention", 1.0, "Critical + immediate"),
        ("Emergency meeting needed NOW", 1.0, "Emergency keyword"),
        
        # Medium-high urgency (0.7-0.9)
        ("This is quite important", 0.7, "Important keyword"),
        ("High priority task", 0.8, "High priority"),
        ("We should address this soon", 0.6, "Soon keyword"),
        
        # Low urgency (0.0-0.3)
        ("Regular weekly update", 0.0, "No urgency indicators"),
        ("For your information", 0.0, "FYI context"),
        ("When you have time", 0.0, "Casual timing"),
        
        # Context-dependent
        ("The deadline is tomorrow", 0.8, "Near deadline"),
        ("Due by end of quarter", 0.3, "Distant deadline"),
        ("Already overdue!", 1.0, "Overdue status"),
    ])
    def test_urgency_detection(self, extractor, content, expected_urgency, description):
        """Test urgency detection across various scenarios."""
        result = extractor.extract(content)
        
        assert isinstance(result.urgency, float)
        assert 0 <= result.urgency <= 1
        assert abs(result.urgency - expected_urgency) < 0.15, (
            f"Failed for '{description}': "
            f"expected ~{expected_urgency}, got {result.urgency}"
        )
    
    def test_urgency_accumulation(self, extractor):
        """Test that multiple urgency indicators accumulate properly."""
        # Single indicator
        result1 = extractor.extract("This is urgent")
        
        # Multiple indicators should increase urgency
        result2 = extractor.extract("This is URGENT and CRITICAL - need ASAP!")
        
        assert result2.urgency >= result1.urgency
        assert result2.urgency == 1.0  # Should max out
    
    def test_urgency_case_insensitive(self, extractor):
        """Test that urgency detection is case-insensitive."""
        result1 = extractor.extract("URGENT task")
        result2 = extractor.extract("urgent task")
        result3 = extractor.extract("Urgent task")
        
        assert result1.urgency == result2.urgency == result3.urgency
    
    # ===== Deadline Proximity Tests =====
    
    @pytest.mark.parametrize("content,min_proximity,max_proximity", [
        ("Due tomorrow", 0.8, 1.0),
        ("Deadline is today", 0.9, 1.0),
        ("Due in 2 days", 0.7, 0.9),
        ("Due next week", 0.4, 0.6),
        ("Due next month", 0.2, 0.4),
        ("Due next quarter", 0.1, 0.3),
        ("No deadline mentioned", 0.0, 0.1),
    ])
    def test_deadline_proximity(self, extractor, content, min_proximity, max_proximity):
        """Test deadline proximity calculation."""
        result = extractor.extract(content)
        
        assert isinstance(result.deadline_proximity, float)
        assert 0 <= result.deadline_proximity <= 1
        assert min_proximity <= result.deadline_proximity <= max_proximity, (
            f"Deadline proximity {result.deadline_proximity} "
            f"not in range [{min_proximity}, {max_proximity}] for: {content}"
        )
    
    def test_deadline_formats(self, extractor):
        """Test various deadline format recognitions."""
        deadline_phrases = [
            "by tomorrow 5pm",
            "before end of day",
            "by COB today",
            "due Monday morning",
            "deadline: next Friday",
            "must be done by 15th",
            "complete by month end"
        ]
        
        for phrase in deadline_phrases:
            result = extractor.extract(phrase)
            assert result.deadline_proximity > 0, (
                f"Failed to detect deadline in: {phrase}"
            )
    
    # ===== Sequence Position Tests =====
    
    def test_sequence_position_calculation(self, extractor):
        """Test sequence position within meeting context."""
        meeting_duration = 3600000  # 1 hour in ms
        
        # Beginning of meeting
        result1 = extractor.extract(
            "Let's get started",
            timestamp_ms=0,
            meeting_duration_ms=meeting_duration
        )
        assert result1.sequence_position < 0.2
        
        # Middle of meeting  
        result2 = extractor.extract(
            "Moving on to the next topic",
            timestamp_ms=1800000,  # 30 min
            meeting_duration_ms=meeting_duration
        )
        assert 0.4 < result2.sequence_position < 0.6
        
        # End of meeting
        result3 = extractor.extract(
            "Let's wrap up",
            timestamp_ms=3300000,  # 55 min
            meeting_duration_ms=meeting_duration
        )
        assert result3.sequence_position > 0.9
    
    def test_sequence_position_defaults(self, extractor):
        """Test sequence position with missing context."""
        # No timestamp or duration
        result = extractor.extract("Some content")
        assert result.sequence_position == 0.5  # Default middle
        
        # Only timestamp, no duration
        result = extractor.extract("Some content", timestamp_ms=1000)
        assert result.sequence_position == 0.5
    
    # ===== Duration Relevance Tests =====
    
    def test_duration_relevance(self, extractor):
        """Test duration relevance extraction."""
        # Short duration mentions
        result1 = extractor.extract("This will take 5 minutes")
        assert result1.duration_relevance < 0.3
        
        # Medium duration
        result2 = extractor.extract("We need 2 hours for this")
        assert 0.3 < result2.duration_relevance < 0.7
        
        # Long duration
        result3 = extractor.extract("This is a 3-month project")
        assert result3.duration_relevance > 0.7
    
    @pytest.mark.parametrize("content,has_duration", [
        ("Sprint lasts 2 weeks", True),
        ("Quarter-long initiative", True),
        ("Day-long workshop", True),
        ("Multi-year program", True),
        ("No duration mentioned", False),
        ("Regular status update", False),
    ])
    def test_duration_detection(self, extractor, content, has_duration):
        """Test detection of duration mentions."""
        result = extractor.extract(content)
        
        if has_duration:
            assert result.duration_relevance > 0
        else:
            assert result.duration_relevance == 0
    
    # ===== Integration Tests =====
    
    def test_combined_temporal_features(self, extractor):
        """Test extraction with multiple temporal indicators."""
        content = "URGENT: Project deadline tomorrow, this 2-week sprint is critical!"
        result = extractor.extract(content, timestamp_ms=0, meeting_duration_ms=3600000)
        
        # Should have high urgency
        assert result.urgency > 0.8
        
        # Should have high deadline proximity
        assert result.deadline_proximity > 0.8
        
        # Should detect duration
        assert result.duration_relevance > 0
        
        # All values should be valid
        assert all(0 <= val <= 1 for val in result.to_array())
    
    # ===== Edge Cases and Error Handling =====
    
    def test_empty_content(self, extractor):
        """Test handling of empty content."""
        result = extractor.extract("")
        
        assert result.urgency == 0
        assert result.deadline_proximity == 0
        assert result.sequence_position == 0.5
        assert result.duration_relevance == 0
    
    def test_very_long_content(self, extractor):
        """Test handling of very long content."""
        long_content = "urgent " * 1000  # Very repetitive
        result = extractor.extract(long_content)
        
        # Should still cap at 1.0
        assert result.urgency == 1.0
        
        # Should complete in reasonable time
        import time
        start = time.time()
        extractor.extract(long_content)
        duration = time.time() - start
        assert duration < 0.1  # Should be fast
    
    def test_special_characters(self, extractor):
        """Test handling of special characters."""
        special_contents = [
            "URGENT!!!!!!",
            ">>> ASAP <<<",
            "**CRITICAL**",
            "~important~",
            "deadline: ****"
        ]
        
        for content in special_contents:
            # Should not crash
            result = extractor.extract(content)
            assert isinstance(result, TemporalFeatures)
    
    def test_null_safety(self, extractor):
        """Test handling of None values."""
        # Should handle None content
        result = extractor.extract(None)
        assert isinstance(result, TemporalFeatures)
        
        # Should handle None context parameters
        result = extractor.extract(
            "Test content",
            timestamp_ms=None,
            meeting_duration_ms=None,
            speaker=None,
            content_type=None
        )
        assert isinstance(result, TemporalFeatures)
    
    # ===== Array Conversion Tests =====
    
    def test_to_array_conversion(self, extractor):
        """Test conversion to numpy array."""
        result = extractor.extract("Urgent deadline tomorrow")
        array = result.to_array()
        
        assert array.shape == (4,)
        assert array.dtype == np.float32
        assert all(0 <= val <= 1 for val in array)
        
        # Check order matches expectation
        assert array[0] == result.urgency
        assert array[1] == result.deadline_proximity
        assert array[2] == result.sequence_position
        assert array[3] == result.duration_relevance
    
    def test_from_array_conversion(self, extractor):
        """Test creation from numpy array."""
        import numpy as np
        
        array = np.array([0.8, 0.6, 0.3, 0.5], dtype=np.float32)
        features = TemporalFeatures.from_array(array)
        
        assert features.urgency == 0.8
        assert features.deadline_proximity == 0.6
        assert features.sequence_position == 0.3
        assert features.duration_relevance == 0.5
    
    def test_array_roundtrip(self, extractor):
        """Test array conversion roundtrip."""
        original = extractor.extract("Urgent task due tomorrow")
        array = original.to_array()
        reconstructed = TemporalFeatures.from_array(array)
        
        assert original.urgency == reconstructed.urgency
        assert original.deadline_proximity == reconstructed.deadline_proximity
        assert original.sequence_position == reconstructed.sequence_position
        assert original.duration_relevance == reconstructed.duration_relevance
    
    # ===== Performance Tests =====
    
    @pytest.mark.performance
    def test_extraction_performance(self, extractor, benchmark_timer):
        """Test extraction performance meets requirements."""
        test_contents = [
            "Regular meeting notes",
            "URGENT: Critical issue needs immediate attention!",
            "Project deadline is next Friday, 2-week sprint",
            "Long content " * 50
        ]
        
        # Warm up
        for content in test_contents:
            extractor.extract(content)
        
        # Measure
        with benchmark_timer:
            for _ in range(100):
                for content in test_contents:
                    extractor.extract(content)
        
        # Should be fast (< 1ms per extraction average)
        avg_time = benchmark_timer.last / (100 * len(test_contents))
        assert avg_time < 0.001, f"Extraction too slow: {avg_time*1000:.2f}ms per extract"


class TestTemporalFeatures:
    """Test the TemporalFeatures dataclass itself."""
    
    def test_value_validation(self):
        """Test that values are properly validated."""
        # Should accept valid values
        features = TemporalFeatures(
            urgency=0.5,
            deadline_proximity=0.8,
            sequence_position=0.2,
            duration_relevance=0.0
        )
        assert features.urgency == 0.5
        
    def test_default_values(self):
        """Test default values for features."""
        # Most should default to 0, sequence_position to 0.5
        features = TemporalFeatures()
        
        assert features.urgency == 0.0
        assert features.deadline_proximity == 0.0  
        assert features.sequence_position == 0.5
        assert features.duration_relevance == 0.0