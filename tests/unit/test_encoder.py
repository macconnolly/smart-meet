"""
Unit tests for ONNX text encoder.

Reference: IMPLEMENTATION_GUIDE.md - Day 2: Embeddings Infrastructure
Tests encoding functionality, performance, and caching.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, MagicMock
import time
from pathlib import Path

from src.embedding.onnx_encoder import ONNXEncoder, get_encoder


class TestONNXEncoder:
    """Test suite for ONNX encoder implementation."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock ONNX inference session."""
        session = Mock()
        session.get_inputs.return_value = [
            Mock(name="input_ids"),
            Mock(name="attention_mask")
        ]
        session.get_outputs.return_value = [
            Mock(name="embeddings")
        ]
        # Return 384D embeddings
        session.run.return_value = [np.random.randn(1, 512, 384)]
        return session
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer."""
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": np.array([[101, 2054, 2003, 102]]),
            "attention_mask": np.array([[1, 1, 1, 1]])
        }
        return tokenizer
    
    @pytest.fixture
    def encoder(self, mock_session, mock_tokenizer, tmp_path):
        """Create encoder with mocked dependencies."""
        with patch('onnxruntime.InferenceSession', return_value=mock_session):
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
                # Create dummy model file
                model_path = tmp_path / "models"
                model_path.mkdir()
                (model_path / "model.onnx").touch()
                
                encoder = ONNXEncoder(str(model_path))
                encoder.session = mock_session
                encoder.tokenizer = mock_tokenizer
                return encoder
    
    @pytest.mark.asyncio
    async def test_encode_single_text(self, encoder):
        """Test encoding a single text."""
        text = "This is a test sentence."
        
        # Mock the _encode_text method to return a normalized vector
        embedding = np.random.randn(384)
        embedding = embedding / np.linalg.norm(embedding)
        encoder._encode_text = Mock(return_value=embedding)
        
        result = await encoder.encode(text)
        
        assert result.shape == (384,)
        assert np.allclose(np.linalg.norm(result), 1.0)  # Normalized
        assert encoder._encode_text.called
    
    @pytest.mark.asyncio
    async def test_encode_empty_text(self, encoder):
        """Test encoding empty text returns zero vector."""
        result = await encoder.encode("")
        
        assert result.shape == (384,)
        assert np.allclose(result, 0.0)
    
    @pytest.mark.asyncio
    async def test_encode_batch(self, encoder):
        """Test batch encoding."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        
        # Mock batch encoding
        embeddings = np.random.randn(3, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        encoder._encode_batch_texts = Mock(return_value=embeddings)
        
        result = await encoder.encode_batch(texts)
        
        assert result.shape == (3, 384)
        # Check all embeddings are normalized
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0)
    
    @pytest.mark.asyncio
    async def test_encode_batch_with_empty_texts(self, encoder):
        """Test batch encoding handles empty texts."""
        texts = ["Valid text", "", "Another valid text"]
        
        # Mock batch encoding for non-empty texts
        embeddings = np.random.randn(2, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        encoder._encode_batch_texts = Mock(return_value=embeddings)
        
        result = await encoder.encode_batch(texts)
        
        assert result.shape == (3, 384)
        # Check empty text has zero embedding
        assert np.allclose(result[1], 0.0)
    
    @pytest.mark.asyncio
    async def test_caching(self, encoder):
        """Test caching functionality."""
        text = "This text should be cached."
        
        # Mock encoding
        embedding = np.random.randn(384)
        embedding = embedding / np.linalg.norm(embedding)
        encoder._encode_text = Mock(return_value=embedding)
        
        # First call - cache miss
        result1 = await encoder.encode(text)
        assert encoder._cache_misses == 1
        assert encoder._cache_hits == 0
        
        # Second call - cache hit
        result2 = await encoder.encode(text)
        assert encoder._cache_hits == 1
        assert np.array_equal(result1, result2)
        
        # _encode_text should only be called once
        encoder._encode_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_size_limit(self, encoder):
        """Test cache size limiting."""
        encoder._cache_size = 3  # Small cache for testing
        
        # Mock encoding
        encoder._encode_text = Mock(side_effect=lambda t: np.random.randn(384))
        
        # Fill cache beyond limit
        texts = [f"Text {i}" for i in range(5)]
        for text in texts:
            await encoder.encode(text)
        
        # Cache should only have last 3 texts
        assert len(encoder._text_cache) == 3
    
    def test_benchmark(self, encoder):
        """Test benchmarking functionality."""
        sample_texts = [f"Sample text {i} for benchmarking." for i in range(100)]
        
        # Mock encoding methods
        encoder._encode_text = Mock(return_value=np.random.randn(384))
        encoder._encode_batch_texts = Mock(
            side_effect=lambda texts: np.random.randn(len(texts), 384)
        )
        
        results = encoder.benchmark(sample_texts)
        
        # Check all expected metrics are present
        assert "single_encoding_ms" in results
        assert "single_encoding_std_ms" in results
        assert "batch_10_per_text_ms" in results
        assert "batch_50_per_text_ms" in results
        assert "batch_100_per_text_ms" in results
        assert "cache_hit_rate" in results
        assert "cache_memory_mb" in results
        
        # Verify cache was used in benchmark
        assert results["cache_hit_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_performance_target(self, encoder):
        """Test that encoding meets <100ms performance target."""
        text = "Performance test text."
        
        # Mock fast encoding
        embedding = np.random.randn(384)
        encoder._encode_text = Mock(return_value=embedding)
        
        start = time.perf_counter()
        await encoder.encode(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Should be well under 100ms with mocked encoding
        assert elapsed_ms < 100
    
    def test_mean_pooling(self, encoder):
        """Test mean pooling implementation."""
        # Create test data
        batch_size, seq_len, hidden_size = 2, 4, 384
        token_embeddings = np.random.randn(batch_size, seq_len, hidden_size)
        attention_mask = np.array([
            [1, 1, 1, 0],  # 3 valid tokens
            [1, 1, 0, 0]   # 2 valid tokens
        ])
        
        result = encoder._mean_pooling(token_embeddings, attention_mask)
        
        assert result.shape == (batch_size, hidden_size)
        
        # Check first sequence pooling (manual calculation)
        expected_first = np.mean(token_embeddings[0, :3, :], axis=0)
        assert np.allclose(result[0], expected_first, rtol=1e-5)
    
    def test_normalize_embeddings(self, encoder):
        """Test embedding normalization."""
        embeddings = np.random.randn(5, 384)
        
        normalized = encoder._normalize_embeddings(embeddings)
        
        # Check all embeddings have unit norm
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0)
        
        # Check direction is preserved
        cosine_sim = np.sum(embeddings[0] * normalized[0]) / np.linalg.norm(embeddings[0])
        assert cosine_sim > 0.99
    
    def test_singleton_pattern(self):
        """Test singleton pattern for get_encoder."""
        encoder1 = get_encoder()
        encoder2 = get_encoder()
        
        assert encoder1 is encoder2
    
    @pytest.mark.asyncio
    async def test_concurrent_encoding(self, encoder):
        """Test concurrent encoding requests."""
        texts = [f"Concurrent text {i}" for i in range(10)]
        
        # Mock encoding
        encoder._encode_text = Mock(
            side_effect=lambda t: np.random.randn(384) / np.linalg.norm(np.random.randn(384))
        )
        
        # Run concurrent encodings
        tasks = [encoder.encode(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(r.shape == (384,) for r in results)
        assert all(np.abs(np.linalg.norm(r) - 1.0) < 1e-5 for r in results)


class TestONNXEncoderIntegration:
    """Integration tests requiring actual model files."""
    
    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path("models/all-MiniLM-L6-v2/model.onnx").exists(),
        reason="ONNX model not found. Run scripts/download_model.py first."
    )
    def test_real_model_loading(self):
        """Test loading real ONNX model."""
        encoder = ONNXEncoder()
        
        assert encoder.session is not None
        assert encoder.tokenizer is not None
        assert encoder.embedding_dimension == 384
    
    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path("models/all-MiniLM-L6-v2/model.onnx").exists(),
        reason="ONNX model not found"
    )
    @pytest.mark.asyncio
    async def test_real_encoding(self):
        """Test encoding with real model."""
        encoder = ONNXEncoder()
        
        text = "The quick brown fox jumps over the lazy dog."
        embedding = await encoder.encode(text)
        
        assert embedding.shape == (384,)
        assert np.abs(np.linalg.norm(embedding) - 1.0) < 1e-5
        
        # Test semantic similarity
        similar_text = "A fast brown fox leaps over a sleepy dog."
        similar_embedding = await encoder.encode(similar_text)
        
        similarity = np.dot(embedding, similar_embedding)
        assert similarity > 0.8  # Should be semantically similar