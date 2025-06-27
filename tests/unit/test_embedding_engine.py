"""
Unit tests for the embedding engine module.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, MagicMock, patch

from src.embedding.engine import EmbeddingEngine, EmbeddingEngineConfig, get_embedding_engine
from src.models.entities import Vector, Memory, MemoryType, ContentType
from datetime import datetime


class TestEmbeddingEngineConfig:
    """Test the EmbeddingEngineConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingEngineConfig()

        assert config.encoder_config is None
        assert config.enable_caching is True
        assert config.batch_size == 32
        assert config.warmup_on_init is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EmbeddingEngineConfig(enable_caching=False, batch_size=64, warmup_on_init=False)

        assert config.enable_caching is False
        assert config.batch_size == 64
        assert config.warmup_on_init is False


class TestEmbeddingEngine:
    """Test the EmbeddingEngine class."""

    @pytest.fixture
    def mock_encoder(self):
        """Create a mock ONNX encoder."""
        encoder = Mock()
        encoder.get_embedding_dimension.return_value = 384
        encoder.encode.return_value = np.random.randn(384)
        encoder.encode_batch.return_value = np.random.randn(3, 384)
        encoder.set_cache_enabled = Mock()
        encoder.warmup = Mock()
        encoder.clear_cache = Mock()
        encoder.get_performance_stats.return_value = {
            "avg_encoding_time_ms": 50.0,
            "cache_hit_rate": 0.8,
        }
        return encoder

    @pytest.fixture
    def mock_vector_manager(self):
        """Create a mock vector manager."""
        manager = Mock()

        def compose_vector(semantic, dimensions):
            return Vector(semantic=semantic, dimensions=dimensions)

        manager.compose_vector.side_effect = compose_vector
        manager.batch_compose.return_value = [
            Vector(semantic=np.random.randn(384), dimensions=np.random.rand(16)) for _ in range(3)
        ]
        manager.validate_vector.return_value = (True, None)
        manager.normalize_vector.side_effect = lambda v: v
        manager.compute_similarity.return_value = 0.85

        return manager

    @pytest.fixture
    def mock_dimension_analyzer(self):
        """Create a mock dimension analyzer."""
        analyzer = Mock()
        analyzer.analyze.return_value = np.random.rand(16)
        return analyzer

    @pytest.fixture
    def engine(self, mock_encoder, mock_vector_manager, mock_dimension_analyzer):
        """Create an engine with mocked components."""
        with patch("src.embedding.engine.get_encoder", return_value=mock_encoder):
            with patch("src.embedding.engine.get_vector_manager", return_value=mock_vector_manager):
                with patch(
                    "src.embedding.engine.DimensionAnalyzer", return_value=mock_dimension_analyzer
                ):
                    config = EmbeddingEngineConfig(warmup_on_init=False)
                    engine = EmbeddingEngine(config)
                    return engine

    def test_initialization(self, mock_encoder, mock_vector_manager, mock_dimension_analyzer):
        """Test engine initialization."""
        with patch("src.embedding.engine.get_encoder", return_value=mock_encoder):
            with patch("src.embedding.engine.get_vector_manager", return_value=mock_vector_manager):
                with patch(
                    "src.embedding.engine.DimensionAnalyzer", return_value=mock_dimension_analyzer
                ):
                    config = EmbeddingEngineConfig(enable_caching=False, warmup_on_init=True)
                    engine = EmbeddingEngine(config)

                    # Check components initialized
                    assert engine.encoder is mock_encoder
                    assert engine.vector_manager is mock_vector_manager
                    assert engine.dimension_analyzer is mock_dimension_analyzer

                    # Check cache setting
                    mock_encoder.set_cache_enabled.assert_called_once_with(False)

                    # Check warmup called
                    mock_encoder.warmup.assert_called()

    def test_warmup(self, engine):
        """Test engine warmup."""
        engine.encoder.warmup.reset_mock()
        engine.dimension_analyzer.analyze.reset_mock()

        engine.warmup(iterations=3)

        # Encoder warmup called
        engine.encoder.warmup.assert_called_once_with(3)

        # Dimension analyzer warmed up
        # 3 test texts × 3 iterations = 9 calls
        assert engine.dimension_analyzer.analyze.call_count == 9

    @pytest.mark.asyncio
    async def test_create_vector(self, engine):
        """Test async vector creation."""
        text = "This is a test sentence."
        context = {"speaker": "John"}

        vector = await engine.create_vector(text, context)

        assert isinstance(vector, Vector)
        assert vector.semantic.shape == (384,)
        assert vector.dimensions.shape == (16,)

        # Check components called correctly
        engine.encoder.encode.assert_called_with(text)
        engine.dimension_analyzer.analyze.assert_called_with(text, context)
        engine.vector_manager.compose_vector.assert_called()
        engine.vector_manager.validate_vector.assert_called()

    @pytest.mark.asyncio
    async def test_create_vector_invalid(self, engine):
        """Test vector creation with validation failure."""
        # Make validation fail
        engine.vector_manager.validate_vector.return_value = (False, "Test error")

        text = "Test sentence"
        vector = await engine.create_vector(text)

        # Should normalize
        engine.vector_manager.normalize_vector.assert_called()

    @pytest.mark.asyncio
    async def test_create_vectors_batch(self, engine):
        """Test batch vector creation."""
        texts = ["Text 1", "Text 2", "Text 3"]
        contexts = [{"id": 1}, {"id": 2}, {"id": 3}]

        vectors = await engine.create_vectors_batch(texts, contexts)

        assert len(vectors) == 3
        for vector in vectors:
            assert isinstance(vector, Vector)

        # Check each text processed
        assert engine.encoder.encode.call_count >= 3
        assert engine.dimension_analyzer.analyze.call_count >= 3

    @pytest.mark.asyncio
    async def test_create_vectors_batch_no_contexts(self, engine):
        """Test batch creation without contexts."""
        texts = ["Text 1", "Text 2"]

        vectors = await engine.create_vectors_batch(texts)

        assert len(vectors) == 2

        # Should be called with None context
        calls = engine.dimension_analyzer.analyze.call_args_list
        for call in calls:
            assert call[0][1] is None or len(call[0]) == 1

    @pytest.mark.asyncio
    async def test_create_vectors_batch_empty(self, engine):
        """Test batch creation with empty list."""
        vectors = await engine.create_vectors_batch([])
        assert vectors == []

    @pytest.mark.asyncio
    async def test_create_vectors_batch_mismatch(self, engine):
        """Test error handling for mismatched batch sizes."""
        texts = ["Text 1", "Text 2"]
        contexts = [{"id": 1}]  # Only one context

        with pytest.raises(ValueError, match="must match"):
            await engine.create_vectors_batch(texts, contexts)

    def test_create_vector_sync(self, engine):
        """Test synchronous vector creation."""
        text = "Sync test"
        context = {"sync": True}

        vector = engine.create_vector_sync(text, context)

        assert isinstance(vector, Vector)
        engine.encoder.encode.assert_called_with(text)
        engine.dimension_analyzer.analyze.assert_called_with(text, context)

    def test_create_vectors_batch_sync(self, engine):
        """Test synchronous batch creation."""
        texts = ["Text 1", "Text 2", "Text 3"]

        # Reset mock to return proper batch
        engine.encoder.encode_batch.return_value = np.random.randn(3, 384)

        vectors = engine.create_vectors_batch_sync(texts)

        assert len(vectors) == 3
        engine.encoder.encode_batch.assert_called_with(texts)
        assert engine.dimension_analyzer.analyze.call_count == 3

    def test_update_memory_vector(self, engine):
        """Test updating a memory's vector."""
        memory = Memory(
            id="test-1",
            meeting_id="meeting-1",
            content="Test memory content",
            speaker="Alice",
            timestamp=datetime.now(),
            memory_type=MemoryType.EPISODIC,
            content_type=ContentType.DISCUSSION,
            level=2,
        )

        context = {"meeting": "meeting-1"}
        updated = engine.update_memory_vector(memory, context)

        assert updated is memory  # Same object
        assert updated.vector is not None
        assert isinstance(updated.vector, Vector)

        engine.encoder.encode.assert_called_with(memory.content)
        engine.dimension_analyzer.analyze.assert_called_with(memory.content, context)

    def test_compute_similarity(self, engine):
        """Test similarity computation."""
        vector1 = Vector(semantic=np.random.randn(384), dimensions=np.random.rand(16))
        vector2 = Vector(semantic=np.random.randn(384), dimensions=np.random.rand(16))

        similarity = engine.compute_similarity(vector1, vector2)

        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1

        engine.vector_manager.compute_similarity.assert_called_with(vector1, vector2, 0.7, 0.3)

        # Test custom weights
        engine.compute_similarity(vector1, vector2, semantic_weight=0.9, cognitive_weight=0.1)

        engine.vector_manager.compute_similarity.assert_called_with(vector1, vector2, 0.9, 0.1)

    def test_get_stats(self, engine):
        """Test statistics retrieval."""
        stats = engine.get_stats()

        assert isinstance(stats, dict)
        assert "encoder" in stats
        assert stats["embedding_dimension"] == 384
        assert stats["cognitive_dimensions"] == 16
        assert stats["total_dimensions"] == 400
        assert stats["cache_enabled"] == engine.config.enable_caching

        engine.encoder.get_performance_stats.assert_called()

    def test_clear_cache(self, engine):
        """Test cache clearing."""
        engine.clear_cache()
        engine.encoder.clear_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_vector_creation(self, engine):
        """Test concurrent vector creation."""
        # Create multiple vectors concurrently
        texts = [f"Text {i}" for i in range(10)]

        tasks = [engine.create_vector(text) for text in texts]
        vectors = await asyncio.gather(*tasks)

        assert len(vectors) == 10
        for vector in vectors:
            assert isinstance(vector, Vector)

    def test_singleton_pattern(self, mock_encoder, mock_vector_manager, mock_dimension_analyzer):
        """Test the get_embedding_engine singleton pattern."""
        with patch("src.embedding.engine.get_encoder", return_value=mock_encoder):
            with patch("src.embedding.engine.get_vector_manager", return_value=mock_vector_manager):
                with patch(
                    "src.embedding.engine.DimensionAnalyzer", return_value=mock_dimension_analyzer
                ):
                    # Reset singleton
                    import src.embedding.engine

                    src.embedding.engine._engine_instance = None

                    engine1 = get_embedding_engine()
                    engine2 = get_embedding_engine()

                    # Should be the same instance
                    assert engine1 is engine2

                    # Config should be ignored on second call
                    custom_config = EmbeddingEngineConfig(batch_size=128)
                    engine3 = get_embedding_engine(custom_config)

                    assert engine3 is engine1
                    assert engine3.config.batch_size == 32  # Original config


@pytest.mark.integration
class TestEmbeddingEngineIntegration:
    """Integration tests requiring actual components."""

    @pytest.mark.skip(reason="Requires ONNX model files")
    @pytest.mark.asyncio
    async def test_real_vector_creation(self):
        """Test with real encoder and analyzer."""
        # This would test with actual ONNX model
        engine = EmbeddingEngine()

        text = "This is a real test with actual encoding."
        vector = await engine.create_vector(text)

        assert isinstance(vector, Vector)
        assert vector.semantic.shape == (384,)
        assert vector.dimensions.shape == (16,)
        assert np.isfinite(vector.full_vector).all()
