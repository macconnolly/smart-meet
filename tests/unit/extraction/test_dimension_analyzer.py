"""
Unit tests for dimension analyzer.
"""

import pytest
import asyncio
import numpy as np

from src.extraction.dimensions.dimension_analyzer import (
    DimensionAnalyzer,
    CognitiveDimensions,
    DimensionExtractionContext,
    get_dimension_analyzer
)
from src.extraction.dimensions import (
    TemporalFeatures,
    EmotionalFeatures,
    SocialFeatures,
    CausalFeatures,
    EvolutionaryFeatures
)


class TestDimensionAnalyzer:
    """Test dimension analyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create dimension analyzer instance."""
        return DimensionAnalyzer(use_parallel=False, use_cache=False)
    
    @pytest.fixture
    def analyzer_with_cache(self):
        """Create dimension analyzer with caching enabled."""
        return DimensionAnalyzer(use_parallel=False, use_cache=True)
    
    @pytest.mark.asyncio
    async def test_analyze_basic(self, analyzer):
        """Test basic dimension analysis."""
        content = "We need to complete this urgent task by tomorrow!"
        context = DimensionExtractionContext(
            content_type="action",
            timestamp_ms=30000,
            meeting_duration_ms=60000
        )
        
        dimensions = await analyzer.analyze(content, context)
        
        # Check structure
        assert isinstance(dimensions, CognitiveDimensions)
        assert isinstance(dimensions.temporal, TemporalFeatures)
        assert isinstance(dimensions.emotional, EmotionalFeatures)
        assert isinstance(dimensions.social, SocialFeatures)
        assert isinstance(dimensions.causal, CausalFeatures)
        assert isinstance(dimensions.evolutionary, EvolutionaryFeatures)
        
        # Check array conversion
        array = dimensions.to_array()
        assert array.shape == (16,)
        assert np.all(array >= 0)
        assert np.all(array <= 1)
    
    @pytest.mark.asyncio
    async def test_analyze_no_context(self, analyzer):
        """Test analysis without context."""
        content = "This is a simple statement."
        dimensions = await analyzer.analyze(content)
        
        # Should still work with default context
        array = dimensions.to_array()
        assert array.shape == (16,)
    
    @pytest.mark.asyncio
    async def test_temporal_extraction(self, analyzer):
        """Test temporal dimension extraction."""
        content = "Critical: Complete by end of day today!"
        context = DimensionExtractionContext(
            content_type="action",
            timestamp_ms=50000,
            meeting_duration_ms=60000
        )
        
        dimensions = await analyzer.analyze(content, context)
        
        # Should have high urgency and deadline proximity
        assert dimensions.temporal.urgency > 0.7
        assert dimensions.temporal.deadline_proximity > 0.7
        assert dimensions.temporal.sequence_position > 0.8  # Near end
    
    @pytest.mark.asyncio
    async def test_emotional_extraction(self, analyzer):
        """Test emotional dimension extraction."""
        content = "I'm absolutely thrilled with these amazing results!"
        
        dimensions = await analyzer.analyze(content)
        
        # Should have positive polarity and high intensity
        assert dimensions.emotional.polarity > 0.7
        assert dimensions.emotional.intensity > 0.7
    
    @pytest.mark.asyncio
    async def test_dimension_validation(self, analyzer):
        """Test dimension validation and clipping."""
        # This shouldn't happen in normal operation, but test the validation
        content = "Test content"
        dimensions = await analyzer.analyze(content)
        
        # Manually set invalid values
        dimensions.temporal.urgency = 1.5  # Out of range
        
        # Validation should clip values
        analyzer._validate_dimensions(dimensions)
        assert dimensions.temporal.urgency == 1.0
    
    @pytest.mark.asyncio
    async def test_to_dict_conversion(self, analyzer):
        """Test conversion to dictionary."""
        content = "Strategic planning for next quarter."
        dimensions = await analyzer.analyze(content)
        
        dim_dict = dimensions.to_dict()
        
        # Check all 16 dimensions are present
        assert len(dim_dict) == 16
        
        # Check specific dimensions
        assert "urgency" in dim_dict
        assert "polarity" in dim_dict
        assert "authority" in dim_dict
        assert "dependencies" in dim_dict
        assert "change_rate" in dim_dict
        
        # All values should be floats in [0, 1]
        for value in dim_dict.values():
            assert isinstance(value, float)
            assert 0 <= value <= 1
    
    @pytest.mark.asyncio
    async def test_from_array_conversion(self, analyzer):
        """Test creating dimensions from array."""
        # Create a valid 16D array
        array = np.array([
            # Temporal
            0.8, 0.7, 0.5, 0.6,
            # Emotional
            0.9, 0.8, 0.7,
            # Social
            0.6, 0.7, 0.8,
            # Causal
            0.5, 0.6, 0.7,
            # Evolutionary
            0.8, 0.9, 0.7
        ])
        
        dimensions = CognitiveDimensions.from_array(array)
        
        # Check all components
        assert dimensions.temporal.urgency == 0.8
        assert dimensions.emotional.polarity == 0.9
        assert dimensions.social.authority == 0.6
        assert dimensions.causal.dependencies == 0.5
        assert dimensions.evolutionary.change_rate == 0.8
    
    @pytest.mark.asyncio
    async def test_batch_analyze(self, analyzer):
        """Test batch analysis."""
        contents = [
            "Urgent: Fix the critical bug!",
            "Long-term strategic planning.",
            "I'm concerned about the timeline."
        ]
        
        contexts = [
            DimensionExtractionContext(content_type="action"),
            DimensionExtractionContext(content_type="strategy"),
            DimensionExtractionContext(content_type="concern")
        ]
        
        features_array = await analyzer.batch_analyze(contents, contexts)
        
        assert features_array.shape == (3, 16)
        assert np.all(features_array >= 0)
        assert np.all(features_array <= 1)
    
    @pytest.mark.asyncio
    async def test_caching(self, analyzer_with_cache):
        """Test dimension caching functionality."""
        content = "This content will be cached."
        context = DimensionExtractionContext(content_type="test")
        
        # First call - should compute
        dimensions1 = await analyzer_with_cache.analyze(content, context)
        
        # Second call - should use cache
        dimensions2 = await analyzer_with_cache.analyze(content, context)
        
        # Results should be identical
        np.testing.assert_array_equal(
            dimensions1.to_array(),
            dimensions2.to_array()
        )
        
        # Check cache stats
        if analyzer_with_cache._cache:
            stats = analyzer_with_cache._cache.get_stats()
            assert stats["hits"] >= 1
    
    @pytest.mark.asyncio
    async def test_parallel_extraction(self):
        """Test parallel dimension extraction."""
        analyzer_parallel = DimensionAnalyzer(use_parallel=True, use_cache=False)
        
        content = "Important decision about project timeline."
        dimensions = await analyzer_parallel.analyze(content)
        
        # Should produce valid results
        array = dimensions.to_array()
        assert array.shape == (16,)
        assert np.all(array >= 0)
        assert np.all(array <= 1)
        
        # Cleanup
        analyzer_parallel.close()
    
    def test_dimension_statistics(self, analyzer):
        """Test dimension statistics calculation."""
        # Create test arrays
        dimensions_array = np.array([
            [0.8, 0.7, 0.5, 0.6, 0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.9, 0.7],
            [0.2, 0.3, 0.5, 0.4, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3, 0.2, 0.1, 0.3],
        ])
        
        stats = analyzer.get_dimension_statistics(dimensions_array)
        
        # Check structure
        assert len(stats) == 16
        assert "urgency" in stats
        
        # Check statistics
        urgency_stats = stats["urgency"]
        assert "mean" in urgency_stats
        assert "std" in urgency_stats
        assert "min" in urgency_stats
        assert "max" in urgency_stats
        assert urgency_stats["mean"] == 0.5
        assert urgency_stats["min"] == 0.2
        assert urgency_stats["max"] == 0.8
    
    def test_singleton_instance(self):
        """Test singleton pattern."""
        analyzer1 = get_dimension_analyzer()
        analyzer2 = get_dimension_analyzer()
        
        assert analyzer1 is analyzer2