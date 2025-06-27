"""
Unit tests for the vector manager module.
"""

import pytest
import numpy as np
import json

from src.embedding.vector_manager import VectorManager, VectorStats, get_vector_manager
from src.models.entities import Vector


class TestVectorManager:
    """Test the VectorManager class."""

    @pytest.fixture
    def manager(self):
        """Create a vector manager instance."""
        return VectorManager()

    @pytest.fixture
    def valid_semantic(self):
        """Create a valid normalized semantic embedding."""
        semantic = np.random.randn(384)
        return semantic / np.linalg.norm(semantic)

    @pytest.fixture
    def valid_dimensions(self):
        """Create valid cognitive dimensions."""
        return np.array(
            [
                # Temporal (4D)
                0.8,
                0.7,
                0.5,
                0.6,
                # Emotional (3D)
                0.9,
                0.8,
                0.7,
                # Social (3D)
                0.6,
                0.7,
                0.8,
                # Causal (3D)
                0.5,
                0.6,
                0.7,
                # Strategic (3D)
                0.8,
                0.9,
                0.7,
            ]
        )

    def test_compose_vector(self, manager, valid_semantic, valid_dimensions):
        """Test vector composition."""
        vector = manager.compose_vector(valid_semantic, valid_dimensions)

        assert isinstance(vector, Vector)
        assert vector.semantic.shape == (384,)
        assert vector.dimensions.shape == (16,)
        assert vector.full_vector.shape == (400,)

        # Check normalization
        assert np.isclose(np.linalg.norm(vector.semantic), 1.0)

        # Check dimensions are in range
        assert np.all(vector.dimensions >= 0)
        assert np.all(vector.dimensions <= 1)

    def test_compose_vector_invalid_dimensions(self, manager, valid_semantic):
        """Test error handling for invalid dimensions."""
        # Wrong semantic dimension
        with pytest.raises(ValueError, match="Semantic embedding must be 384D"):
            manager.compose_vector(np.zeros(100), np.zeros(16))

        # Wrong cognitive dimension
        with pytest.raises(ValueError, match="Cognitive dimensions must be 16D"):
            manager.compose_vector(valid_semantic, np.zeros(10))

    def test_compose_vector_normalization(self, manager, valid_dimensions):
        """Test that semantic vectors are normalized during composition."""
        # Unnormalized semantic
        semantic = np.ones(384) * 2.0  # Not normalized
        vector = manager.compose_vector(semantic, valid_dimensions)

        # Should be normalized
        assert np.isclose(np.linalg.norm(vector.semantic), 1.0)

    def test_compose_vector_dimension_clipping(self, manager, valid_semantic):
        """Test that cognitive dimensions are clipped to [0, 1]."""
        # Out of range dimensions
        dimensions = np.array(
            [
                -0.5,
                1.5,
                0.3,
                0.7,  # Mix of invalid and valid
                0.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ]
        )

        vector = manager.compose_vector(valid_semantic, dimensions)

        # Should be clipped
        assert vector.dimensions[0] == 0.0  # -0.5 -> 0.0
        assert vector.dimensions[1] == 1.0  # 1.5 -> 1.0
        assert np.all(vector.dimensions >= 0)
        assert np.all(vector.dimensions <= 1)

    def test_decompose_vector(self, manager, valid_semantic, valid_dimensions):
        """Test vector decomposition."""
        vector = manager.compose_vector(valid_semantic, valid_dimensions)
        semantic, dimensions = manager.decompose_vector(vector)

        # Should return copies
        assert semantic is not vector.semantic
        assert dimensions is not vector.dimensions

        # Should be equal
        np.testing.assert_array_equal(semantic, vector.semantic)
        np.testing.assert_array_equal(dimensions, vector.dimensions)

    def test_validate_vector(self, manager, valid_semantic, valid_dimensions):
        """Test vector validation."""
        # Valid vector
        vector = manager.compose_vector(valid_semantic, valid_dimensions)
        is_valid, error = manager.validate_vector(vector)
        assert is_valid
        assert error is None

        # Invalid semantic dimension
        bad_vector = Vector(semantic=np.zeros(100), dimensions=valid_dimensions)
        is_valid, error = manager.validate_vector(bad_vector)
        assert not is_valid
        assert "Semantic dimension mismatch" in error

        # Invalid cognitive dimension
        bad_vector = Vector(semantic=valid_semantic, dimensions=np.zeros(10))
        is_valid, error = manager.validate_vector(bad_vector)
        assert not is_valid
        assert "Cognitive dimension mismatch" in error

        # Non-normalized semantic
        bad_vector = Vector(semantic=np.ones(384) * 2.0, dimensions=valid_dimensions)
        is_valid, error = manager.validate_vector(bad_vector)
        assert not is_valid
        assert "not normalized" in error

        # Out of range dimensions
        bad_dimensions = valid_dimensions.copy()
        bad_dimensions[0] = 1.5
        bad_vector = Vector(semantic=valid_semantic, dimensions=bad_dimensions)
        is_valid, error = manager.validate_vector(bad_vector)
        assert not is_valid
        assert "outside [0, 1] range" in error

        # NaN in semantic
        bad_semantic = valid_semantic.copy()
        bad_semantic[0] = np.nan
        bad_vector = Vector(semantic=bad_semantic, dimensions=valid_dimensions)
        is_valid, error = manager.validate_vector(bad_vector)
        assert not is_valid
        assert "NaN or inf" in error

    def test_normalize_vector(self, manager):
        """Test vector normalization."""
        # Unnormalized vector
        vector = Vector(
            semantic=np.ones(384) * 2.0,
            dimensions=np.array(
                [-0.5, 1.5, 0.3, 0.7, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            ),
        )

        normalized = manager.normalize_vector(vector)

        # Semantic should be normalized
        assert np.isclose(np.linalg.norm(normalized.semantic), 1.0)

        # Dimensions should be clipped
        assert np.all(normalized.dimensions >= 0)
        assert np.all(normalized.dimensions <= 1)
        assert normalized.dimensions[0] == 0.0
        assert normalized.dimensions[1] == 1.0

        # Should be valid
        is_valid, _ = manager.validate_vector(normalized)
        assert is_valid

    def test_get_vector_stats(self, manager, valid_semantic, valid_dimensions):
        """Test vector statistics calculation."""
        vector = manager.compose_vector(valid_semantic, valid_dimensions)
        stats = manager.get_vector_stats(vector)

        assert isinstance(stats, VectorStats)
        assert np.isclose(stats.semantic_norm, 1.0)
        assert 0 <= stats.dimensions_mean <= 1
        assert stats.dimensions_min >= 0
        assert stats.dimensions_max <= 1
        assert stats.is_normalized
        assert stats.is_valid

    def test_batch_compose(self, manager):
        """Test batch vector composition."""
        n_vectors = 5
        semantic_embeddings = np.random.randn(n_vectors, 384)
        # Normalize each
        for i in range(n_vectors):
            semantic_embeddings[i] /= np.linalg.norm(semantic_embeddings[i])

        cognitive_dimensions = np.random.rand(n_vectors, 16)

        vectors = manager.batch_compose(semantic_embeddings, cognitive_dimensions)

        assert len(vectors) == n_vectors
        for vector in vectors:
            assert isinstance(vector, Vector)
            is_valid, _ = manager.validate_vector(vector)
            assert is_valid

    def test_batch_compose_mismatch(self, manager):
        """Test error handling for mismatched batch sizes."""
        semantic_embeddings = np.random.randn(5, 384)
        cognitive_dimensions = np.random.rand(3, 16)

        with pytest.raises(ValueError, match="must match"):
            manager.batch_compose(semantic_embeddings, cognitive_dimensions)

    def test_vectors_to_array(self, manager, valid_semantic, valid_dimensions):
        """Test converting vectors to numpy array."""
        vectors = []
        for i in range(3):
            vector = manager.compose_vector(
                valid_semantic + np.random.randn(384) * 0.01, valid_dimensions * (0.8 + i * 0.1)
            )
            vectors.append(vector)

        array = manager.vectors_to_array(vectors)

        assert array.shape == (3, 400)
        assert isinstance(array, np.ndarray)

        # Test empty list
        empty_array = manager.vectors_to_array([])
        assert empty_array.shape == (0, 400)

    def test_array_to_vectors(self, manager):
        """Test converting numpy array to vectors."""
        array = np.random.randn(3, 400)
        # Normalize semantic parts
        for i in range(3):
            array[i, :384] /= np.linalg.norm(array[i, :384])
            array[i, 384:] = np.clip(array[i, 384:], 0, 1)

        vectors = manager.array_to_vectors(array)

        assert len(vectors) == 3
        for i, vector in enumerate(vectors):
            assert isinstance(vector, Vector)
            np.testing.assert_array_almost_equal(vector.full_vector, array[i])

    def test_array_to_vectors_invalid(self, manager):
        """Test error handling for invalid array dimensions."""
        with pytest.raises(ValueError, match="must have 400 columns"):
            manager.array_to_vectors(np.zeros((5, 300)))

    def test_get_dimension_names(self, manager):
        """Test dimension name retrieval."""
        names = manager.get_dimension_names()

        assert isinstance(names, dict)
        assert "temporal" in names
        assert "emotional" in names
        assert "social" in names
        assert "causal" in names
        assert "strategic" in names

        # Check counts
        assert len(names["temporal"]) == 4
        assert len(names["emotional"]) == 3
        assert len(names["social"]) == 3
        assert len(names["causal"]) == 3
        assert len(names["strategic"]) == 3

        # Total should be 16
        total = sum(len(v) for v in names.values())
        assert total == 16

    def test_get_dimension_value(self, manager, valid_semantic, valid_dimensions):
        """Test getting specific dimension values."""
        vector = manager.compose_vector(valid_semantic, valid_dimensions)

        # Test each dimension type
        assert manager.get_dimension_value(vector, "urgency") == valid_dimensions[0]
        assert manager.get_dimension_value(vector, "polarity") == valid_dimensions[4]
        assert manager.get_dimension_value(vector, "authority") == valid_dimensions[7]
        assert manager.get_dimension_value(vector, "dependencies") == valid_dimensions[10]
        assert manager.get_dimension_value(vector, "alignment") == valid_dimensions[13]

        # Test invalid dimension
        with pytest.raises(ValueError, match="Unknown dimension"):
            manager.get_dimension_value(vector, "invalid_dimension")

    def test_set_dimension_value(self, manager, valid_semantic, valid_dimensions):
        """Test setting specific dimension values."""
        vector = manager.compose_vector(valid_semantic, valid_dimensions)

        # Set urgency to new value
        new_vector = manager.set_dimension_value(vector, "urgency", 0.95)

        # Original should be unchanged
        assert manager.get_dimension_value(vector, "urgency") == valid_dimensions[0]

        # New vector should have updated value
        assert manager.get_dimension_value(new_vector, "urgency") == 0.95

        # Other dimensions should be unchanged
        for i in range(1, 16):
            assert new_vector.dimensions[i] == vector.dimensions[i]

        # Test clipping
        clipped_vector = manager.set_dimension_value(vector, "polarity", 1.5)
        assert manager.get_dimension_value(clipped_vector, "polarity") == 1.0

        clipped_vector = manager.set_dimension_value(vector, "impact", -0.5)
        assert manager.get_dimension_value(clipped_vector, "impact") == 0.0

    def test_compute_similarity(self, manager, valid_semantic, valid_dimensions):
        """Test similarity computation."""
        vector1 = manager.compose_vector(valid_semantic, valid_dimensions)

        # Identical vector
        sim = manager.compute_similarity(vector1, vector1)
        assert np.isclose(sim, 1.0)

        # Slightly different semantic
        semantic2 = valid_semantic + np.random.randn(384) * 0.1
        semantic2 /= np.linalg.norm(semantic2)
        vector2 = manager.compose_vector(semantic2, valid_dimensions)

        sim = manager.compute_similarity(vector1, vector2)
        assert 0.8 < sim < 1.0  # Should be high but not perfect

        # Different dimensions
        dimensions3 = valid_dimensions * 0.5
        vector3 = manager.compose_vector(valid_semantic, dimensions3)

        sim = manager.compute_similarity(vector1, vector3)
        assert 0.5 < sim < 0.9  # Should be moderate

        # Test weight adjustment
        sim_semantic = manager.compute_similarity(
            vector1, vector2, semantic_weight=1.0, cognitive_weight=0.0
        )
        sim_cognitive = manager.compute_similarity(
            vector1, vector3, semantic_weight=0.0, cognitive_weight=1.0
        )

        # Semantic-only should be higher for vector2
        # Cognitive-only should be lower for vector3
        assert sim_semantic > sim_cognitive

    def test_to_json(self, manager, valid_semantic, valid_dimensions):
        """Test JSON serialization."""
        vector = manager.compose_vector(valid_semantic, valid_dimensions)
        json_str = manager.to_json(vector)

        # Should be valid JSON
        data = json.loads(json_str)
        assert "semantic" in data
        assert "dimensions" in data
        assert len(data["semantic"]) == 384
        assert len(data["dimensions"]) == 16

    def test_from_json(self, manager, valid_semantic, valid_dimensions):
        """Test JSON deserialization."""
        vector = manager.compose_vector(valid_semantic, valid_dimensions)
        json_str = manager.to_json(vector)

        # Deserialize
        restored = manager.from_json(json_str)

        # Should be equal
        np.testing.assert_array_almost_equal(restored.semantic, vector.semantic)
        np.testing.assert_array_almost_equal(restored.dimensions, vector.dimensions)

    def test_singleton_pattern(self):
        """Test the get_vector_manager singleton pattern."""
        manager1 = get_vector_manager()
        manager2 = get_vector_manager()

        # Should be the same instance
        assert manager1 is manager2

    def test_dimension_indices(self, manager):
        """Test dimension index constants."""
        assert manager.TEMPORAL_START == 0
        assert manager.TEMPORAL_END == 4
        assert manager.EMOTIONAL_START == 4
        assert manager.EMOTIONAL_END == 7
        assert manager.SOCIAL_START == 7
        assert manager.SOCIAL_END == 10
        assert manager.CAUSAL_START == 10
        assert manager.CAUSAL_END == 13
        assert manager.STRATEGIC_START == 13
        assert manager.STRATEGIC_END == 16

        # Total should cover all 16 dimensions
        assert manager.STRATEGIC_END - manager.TEMPORAL_START == 16

    def test_zero_vector_handling(self, manager, valid_dimensions):
        """Test handling of zero semantic vectors."""
        zero_semantic = np.zeros(384)
        vector = manager.compose_vector(zero_semantic, valid_dimensions)

        # Should create a valid vector with first component set to 1
        assert vector.semantic[0] == 1.0
        assert np.isclose(np.linalg.norm(vector.semantic), 1.0)

        # Normalize should also handle this
        zero_vector = Vector(semantic=np.zeros(384), dimensions=valid_dimensions)
        normalized = manager.normalize_vector(zero_vector)
        assert normalized.semantic[0] == 1.0
