
import pytest
import numpy as np
from datetime import datetime, timedelta

from src.embedding.vector_manager import VectorManager
from src.extraction.dimensions.temporal import TemporalDimensionExtractor
from src.extraction.dimensions.emotional import EmotionalDimensionExtractor
from src.extraction.dimensions.analyzer import DimensionAnalyzer
from src.models.entities import Vector


class TestVectorManager:
    """Test suite for VectorManager."""

    @pytest.fixture
    def manager(self):
        return VectorManager()

    def test_compose_vector(self, manager):
        """Test successful vector composition."""
        semantic = np.random.rand(384)
        semantic = semantic / np.linalg.norm(semantic)  # Normalize
        cognitive = np.random.rand(16)

        vector = manager.compose_vector(semantic, cognitive)

        assert vector.semantic.shape == (384,)
        assert vector.dimensions.shape == (16,)
        assert np.allclose(np.linalg.norm(vector.semantic), 1.0)
        assert np.all((vector.dimensions >= 0) & (vector.dimensions <= 1))

    def test_compose_vector_invalid_dimensions(self, manager):
        """Test composition with invalid input dimensions."""
        with pytest.raises(ValueError):
            manager.compose_vector(np.random.rand(300), np.random.rand(16))
        with pytest.raises(ValueError):
            manager.compose_vector(np.random.rand(384), np.random.rand(10))

    def test_decompose_vector(self, manager):
        """Test vector decomposition."""
        semantic = np.random.rand(384)
        semantic = semantic / np.linalg.norm(semantic)
        cognitive = np.random.rand(16)
        vector = manager.compose_vector(semantic, cognitive)

        s, c = manager.decompose_vector(vector)

        assert np.allclose(s, semantic)
        assert np.array_equal(c, cognitive)

    def test_validate_vector(self, manager):
        """Test vector validation."""
        semantic = np.random.rand(384)
        semantic = semantic / np.linalg.norm(semantic)
        cognitive = np.random.rand(16)
        vector = manager.compose_vector(semantic, cognitive)

        is_valid, error = manager.validate_vector(vector)
        assert is_valid is True
        assert error is None

        # Test invalid semantic dimension (by modifying a valid vector)
        invalid_semantic_vector = Vector(np.random.rand(384), cognitive)
        invalid_semantic_vector.semantic = np.random.rand(300)
        is_valid, error = manager.validate_vector(invalid_semantic_vector)
        assert is_valid is False
        assert "Semantic dimension mismatch" in error

        # Test invalid cognitive dimension (by modifying a valid vector)
        invalid_cognitive_vector_shape = Vector(semantic, np.random.rand(16))
        invalid_cognitive_vector_shape.dimensions = np.random.rand(10)
        is_valid, error = manager.validate_vector(invalid_cognitive_vector_shape)
        assert is_valid is False
        assert "Cognitive dimension mismatch" in error

        # Test unnormalized semantic
        unnormalized_semantic = np.random.rand(384) * 2
        unnormalized_vector = Vector(semantic=unnormalized_semantic, dimensions=cognitive)
        is_valid, error = manager.validate_vector(unnormalized_vector)
        assert is_valid is False
        assert "Semantic vector not normalized" in error

        # Test invalid cognitive dimension range
        invalid_cognitive_range = np.random.rand(16) * 2
        invalid_cognitive_vector_range = Vector(semantic=semantic, dimensions=invalid_cognitive_range)
        is_valid, error = manager.validate_vector(invalid_cognitive_vector_range)
        assert is_valid is False
        assert "Cognitive dimensions outside [0, 1] range" in error

    def test_normalize_vector(self, manager):
        """Test vector normalization/clipping."""
        semantic = np.random.rand(384) * 5  # Unnormalized
        cognitive = np.random.rand(16) * 2 - 0.5  # Outside [0,1]
        vector = Vector(semantic, cognitive)

        normalized_vector = manager.normalize_vector(vector)

        assert np.allclose(np.linalg.norm(normalized_vector.semantic), 1.0)
        assert np.all((normalized_vector.dimensions >= 0) & (normalized_vector.dimensions <= 1))

    def test_get_vector_stats(self, manager):
        """Test vector statistics."""
        semantic = np.random.rand(384)
        semantic = semantic / np.linalg.norm(semantic)
        cognitive = np.random.rand(16)
        vector = manager.compose_vector(semantic, cognitive)

        stats = manager.get_vector_stats(vector)
        assert stats.is_valid is True
        assert stats.is_normalized
        assert np.isclose(stats.semantic_norm, 1.0)

    def test_batch_compose(self, manager):
        """Test batch composition."""
        semantics = np.random.rand(5, 384)
        semantics = semantics / np.linalg.norm(semantics, axis=1, keepdims=True)
        cognitives = np.random.rand(5, 16)

        vectors = manager.batch_compose(semantics, cognitives)
        assert len(vectors) == 5
        assert all(v.full_vector.shape == (400,) for v in vectors)

    def test_compute_similarity(self, manager):
        """Test similarity computation."""
        semantic1 = np.random.rand(384)
        semantic1 = semantic1 / np.linalg.norm(semantic1)
        cognitive1 = np.random.rand(16)
        vector1 = manager.compose_vector(semantic1, cognitive1)

        # Identical vector
        vector2 = manager.compose_vector(semantic1, cognitive1)
        sim = manager.compute_similarity(vector1, vector2)
        assert np.isclose(sim, 1.0)

        # Different vector
        semantic3 = np.random.uniform(-1, 1, 384)
        semantic3 = semantic3 / np.linalg.norm(semantic3)
        cognitive3 = np.random.uniform(0, 1, 16)
        vector3 = manager.compose_vector(semantic3, cognitive3)
        sim = manager.compute_similarity(vector1, vector3)
        assert sim < 0.2  # Should be low similarity

    def test_get_dimension_value(self, manager):
        """Test getting specific dimension values."""
        semantic = np.random.rand(384)
        cognitive = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        vector = manager.compose_vector(semantic, cognitive)

        assert manager.get_dimension_value(vector, "urgency") == 0.1
        assert manager.get_dimension_value(vector, "polarity") == 0.5
        assert manager.get_dimension_value(vector, "value") == 0.6

        with pytest.raises(ValueError):
            manager.get_dimension_value(vector, "non_existent_dim")

    def test_set_dimension_value(self, manager):
        """Test setting specific dimension values."""
        semantic = np.random.rand(384)
        cognitive = np.random.rand(16)
        vector = manager.compose_vector(semantic, cognitive)

        updated_vector = manager.set_dimension_value(vector, "urgency", 0.9)
        assert manager.get_dimension_value(updated_vector, "urgency") == 0.9

        # Test clipping
        clipped_vector = manager.set_dimension_value(vector, "polarity", 1.5)
        assert manager.get_dimension_value(clipped_vector, "polarity") == 1.0


class TestTemporalDimensionExtractor:
    """Test suite for TemporalDimensionExtractor."""

    @pytest.fixture
    def extractor(self):
        return TemporalDimensionExtractor()

    def test_extract_urgency(self, extractor):
        assert extractor.extract("This is urgent.")[0] == 1.0
        assert extractor.extract("We need to do this asap.")[0] == 1.0
        assert extractor.extract("No urgency here.")[0] == 0.0

    def test_extract_deadline_proximity(self, extractor):
        assert extractor.extract("Due by next Friday.")[1] > 0.0
        assert extractor.extract("Deadline is tomorrow.")[1] == 0.0
        assert extractor.extract("No deadline mentioned.")[1] == 0.0

    def test_extract_sequence_position(self, extractor):
        context = {"current_memory_index": 5, "total_memories": 10}
        assert extractor.extract("text", context)[2] == 0.5
        context_end = {"current_memory_index": 9, "total_memories": 10}
        assert extractor.extract("text", context_end)[2] == 0.9
        assert extractor.extract("text")[2] == 0.5  # Default if no context

    def test_extract_duration_relevance(self, extractor):
        assert extractor.extract("Meeting will be 30 minutes.")[3] > 0.0
        assert extractor.extract("This took many hours.")[3] > 0.0
        assert extractor.extract("No duration.")[3] == 0.0


class TestEmotionalDimensionExtractor:
    """Test suite for EmotionalDimensionExtractor."""

    @pytest.fixture
    def extractor(self):
        return EmotionalDimensionExtractor()

    def test_extract_positive_sentiment(self, extractor):
        dims = extractor.extract("This is a fantastic idea!")
        assert dims[0] > 0.5  # Polarity should be positive
        assert dims[1] > 0.0  # Intensity should be present

    def test_extract_negative_sentiment(self, extractor):
        dims = extractor.extract("This is a terrible problem.")
        assert dims[0] < 0.5  # Polarity should be negative
        assert dims[1] > 0.0  # Intensity should be present

    def test_extract_neutral_sentiment(self, extractor):
        dims = extractor.extract("The sky is blue.")
        assert np.isclose(dims[0], 0.5, atol=0.1)  # Polarity should be neutral
        assert dims[1] > 0.0  # Neutral also has intensity

    def test_extract_empty_text(self, extractor):
        dims = extractor.extract("")
        assert np.allclose(dims, [0.5, 0.0, 0.5])


class TestDimensionAnalyzer:
    """Test suite for DimensionAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return DimensionAnalyzer()

    def test_analyze_all_dimensions(self, analyzer):
        text = "This is an urgent and positive message about the project deadline."
        context = {"current_memory_index": 5, "total_memories": 10}
        dims = analyzer.analyze(text, context)

        assert dims.shape == (16,)
        # Check temporal (urgency, deadline, sequence)
        assert dims[0] == 1.0
        assert dims[1] == 0.0
        assert dims[2] == 0.5
        # Check emotional (polarity)
        assert dims[4] > 0.5
        # Check placeholders (social, causal, strategic)
        assert np.allclose(dims[7:16], 0.5)

    def test_analyze_empty_text(self, analyzer):
        dims = analyzer.analyze("")
        assert dims.shape == (16,)
        # Temporal: urgency, deadline, duration should be 0.0, sequence 0.5
        assert np.allclose(dims[[0, 1, 3]], 0.0)
        assert np.isclose(dims[2], 0.5)
        # Emotional: polarity 0.5, intensity 0.0, confidence 0.5
        assert np.isclose(dims[4], 0.5)
        assert np.isclose(dims[5], 0.0)
        assert np.isclose(dims[6], 0.5)
        # Placeholders
        assert np.allclose(dims[7:16], 0.5)

