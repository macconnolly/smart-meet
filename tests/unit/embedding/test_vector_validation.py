"""
Unit tests for vector validation utilities.
"""

import pytest
import numpy as np

from src.models.entities import Vector
from src.embedding.vector_manager import VectorManager
from src.embedding.vector_validation import (
    VectorValidator,
    ValidationResult,
    get_vector_validator
)


class TestVectorValidator:
    """Test vector validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create vector validator instance."""
        return VectorValidator()
    
    @pytest.fixture
    def vector_manager(self):
        """Create vector manager instance."""
        return VectorManager()
    
    @pytest.fixture
    def valid_vector(self, vector_manager):
        """Create a valid vector."""
        semantic = np.random.randn(384)
        semantic = semantic / np.linalg.norm(semantic)  # Normalize
        cognitive = np.random.rand(16) * 0.8 + 0.1  # Range [0.1, 0.9]
        return vector_manager.compose_vector(semantic, cognitive)
    
    def test_validate_valid_vector(self, validator, valid_vector):
        """Test validation of a valid vector."""
        result = validator.validate_vector(valid_vector)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.stats is not None
    
    def test_validate_unnormalized_semantic(self, validator):
        """Test validation catches unnormalized semantic vector."""
        semantic = np.random.randn(384) * 2  # Not normalized
        cognitive = np.random.rand(16)
        vector = Vector(semantic=semantic, dimensions=cognitive)
        
        result = validator.validate_vector(vector, strict=True)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "not normalized" in result.errors[0]
    
    def test_validate_out_of_range_dimensions(self, validator, vector_manager):
        """Test validation catches out-of-range cognitive dimensions."""
        semantic = np.random.randn(384)
        semantic = semantic / np.linalg.norm(semantic)
        cognitive = np.array([1.5, -0.5] + [0.5] * 14)  # Out of range
        
        # Vector manager should clip, but direct construction won't
        vector = Vector(semantic=semantic, dimensions=cognitive)
        
        result = validator.validate_vector(vector)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "outside [0, 1] range" in result.errors[0]
    
    def test_validate_uniform_dimensions(self, validator, vector_manager):
        """Test validation warns about uniform dimensions."""
        semantic = np.random.randn(384)
        semantic = semantic / np.linalg.norm(semantic)
        cognitive = np.ones(16) * 0.5  # All same value
        vector = vector_manager.compose_vector(semantic, cognitive)
        
        result = validator.validate_vector(vector)
        
        assert result.is_valid  # Still valid, just suspicious
        assert len(result.warnings) > 0
        assert "Low dimension variance" in result.warnings[0]
    
    def test_validate_extreme_values(self, validator, vector_manager):
        """Test validation warns about too many extreme values."""
        semantic = np.random.randn(384)
        semantic = semantic / np.linalg.norm(semantic)
        # Many 0s and 1s
        cognitive = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] + [0.5] * 6)
        vector = vector_manager.compose_vector(semantic, cognitive)
        
        result = validator.validate_vector(vector)
        
        assert result.is_valid
        assert len(result.warnings) > 0
        assert "extreme dimension values" in result.warnings[0]
    
    def test_validate_batch(self, validator, vector_manager):
        """Test batch validation."""
        vectors = []
        
        # Valid vector
        semantic1 = np.random.randn(384)
        semantic1 = semantic1 / np.linalg.norm(semantic1)
        cognitive1 = np.random.rand(16)
        vectors.append(vector_manager.compose_vector(semantic1, cognitive1))
        
        # Invalid vector
        semantic2 = np.random.randn(384) * 2  # Not normalized
        cognitive2 = np.random.rand(16)
        vectors.append(Vector(semantic=semantic2, dimensions=cognitive2))
        
        # Valid vector
        semantic3 = np.random.randn(384)
        semantic3 = semantic3 / np.linalg.norm(semantic3)
        cognitive3 = np.random.rand(16) * 0.5 + 0.25
        vectors.append(vector_manager.compose_vector(semantic3, cognitive3))
        
        results = validator.validate_batch(vectors)
        
        assert len(results) == 3
        assert results[0].is_valid
        assert not results[1].is_valid
        assert results[2].is_valid
    
    def test_check_vector_similarity(self, validator, vector_manager):
        """Test vector similarity checking."""
        # Create two similar vectors
        semantic = np.random.randn(384)
        semantic = semantic / np.linalg.norm(semantic)
        cognitive = np.random.rand(16)
        
        vector1 = vector_manager.compose_vector(semantic, cognitive)
        
        # Slightly modify for second vector
        semantic2 = semantic + np.random.randn(384) * 0.01
        semantic2 = semantic2 / np.linalg.norm(semantic2)
        cognitive2 = cognitive + np.random.randn(16) * 0.01
        cognitive2 = np.clip(cognitive2, 0, 1)
        vector2 = vector_manager.compose_vector(semantic2, cognitive2)
        
        is_similar, similarity = validator.check_vector_similarity(
            vector1, vector2, threshold=0.95
        )
        
        assert isinstance(is_similar, bool)
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
        assert similarity > 0.9  # Should be very similar
    
    def test_find_anomalies(self, validator, vector_manager):
        """Test anomaly detection in batch."""
        vectors = []
        
        # Create normal vectors
        for _ in range(5):
            semantic = np.random.randn(384)
            semantic = semantic / np.linalg.norm(semantic)
            cognitive = np.random.rand(16) * 0.4 + 0.3  # Range [0.3, 0.7]
            vectors.append(vector_manager.compose_vector(semantic, cognitive))
        
        # Create anomalous vector
        semantic = np.random.randn(384)
        semantic = semantic / np.linalg.norm(semantic)
        cognitive = np.ones(16) * 0.99  # All very high
        vectors.append(vector_manager.compose_vector(semantic, cognitive))
        
        anomalies = validator.find_anomalies(vectors, z_threshold=2.0)
        
        assert len(anomalies) > 0
        assert anomalies[0][0] == 5  # Last vector is anomalous
    
    def test_generate_validation_report(self, validator, vector_manager):
        """Test validation report generation."""
        vectors = []
        
        # Mix of valid and invalid vectors
        for i in range(5):
            semantic = np.random.randn(384)
            if i < 3:
                semantic = semantic / np.linalg.norm(semantic)
            cognitive = np.random.rand(16)
            if i == 0:
                vector = vector_manager.compose_vector(semantic, cognitive)
            else:
                vector = Vector(semantic=semantic, dimensions=cognitive)
            vectors.append(vector)
        
        report = validator.generate_validation_report(vectors)
        
        assert "total_vectors" in report
        assert "valid_count" in report
        assert "invalid_count" in report
        assert "validation_rate" in report
        assert "errors" in report
        assert "warnings" in report
        assert "dimension_statistics" in report
        
        assert report["total_vectors"] == 5
        assert report["valid_count"] >= 1
        assert report["invalid_count"] >= 1
    
    def test_dimension_statistics_in_report(self, validator, vector_manager):
        """Test dimension statistics in validation report."""
        vectors = []
        
        # Create vectors with known patterns
        for i in range(3):
            semantic = np.random.randn(384)
            semantic = semantic / np.linalg.norm(semantic)
            cognitive = np.zeros(16)
            cognitive[0] = i * 0.3  # urgency: 0, 0.3, 0.6
            vectors.append(vector_manager.compose_vector(semantic, cognitive))
        
        report = validator.generate_validation_report(vectors)
        
        dim_stats = report["dimension_statistics"]
        assert "urgency" in dim_stats
        
        urgency_stats = dim_stats["urgency"]
        assert urgency_stats["mean"] == pytest.approx(0.3, rel=1e-5)
        assert urgency_stats["min"] == 0.0
        assert urgency_stats["max"] == 0.6
    
    def test_singleton_instance(self):
        """Test singleton pattern."""
        validator1 = get_vector_validator()
        validator2 = get_vector_validator()
        
        assert validator1 is validator2