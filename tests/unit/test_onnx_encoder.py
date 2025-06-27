"""
Unit tests for the ONNX encoder module.
"""

import pytest
import numpy as np
from pathlib import Path
import time
import tempfile
import json

from src.embedding.onnx_encoder import ONNXEncoder, EncoderConfig, get_encoder


class TestEncoderConfig:
    """Test the EncoderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EncoderConfig()

        assert config.model_path == "models/embeddings/model.onnx"
        assert config.tokenizer_path == "models/embeddings/tokenizer"
        assert config.config_path == "models/embeddings/config.json"
        assert config.max_length == 128
        assert config.batch_size == 32
        assert config.cache_size == 10000
        assert config.providers == ["CPUExecutionProvider"]

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EncoderConfig(
            model_path="custom/model.onnx",
            max_length=256,
            batch_size=64,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        assert config.model_path == "custom/model.onnx"
        assert config.max_length == 256
        assert config.batch_size == 64
        assert config.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


class TestONNXEncoder:
    """Test the ONNXEncoder class."""

    @pytest.fixture
    def encoder(self):
        """Create an encoder instance."""
        return ONNXEncoder()

    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder is not None
        assert encoder._session is not None
        assert encoder._tokenizer is not None
        assert encoder._model_config is not None
        assert encoder.get_embedding_dimension() == 384

    def test_single_sentence_encoding(self, encoder):
        """Test encoding a single sentence."""
        sentence = "This is a test sentence for embedding."
        embedding = encoder.encode(sentence)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert np.isfinite(embedding).all()

        # Test normalization
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, rtol=1e-5)

    def test_batch_encoding(self, encoder):
        """Test encoding multiple sentences."""
        sentences = [
            "First test sentence.",
            "Second test sentence with more words.",
            "Third sentence for testing batch processing.",
        ]

        embeddings = encoder.encode(sentences)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert np.isfinite(embeddings).all()

        # Test that each embedding is normalized
        for i in range(3):
            norm = np.linalg.norm(embeddings[i])
            assert np.isclose(norm, 1.0, rtol=1e-5)

    def test_encoding_without_normalization(self, encoder):
        """Test encoding without normalization."""
        sentence = "Test sentence without normalization."
        embedding = encoder.encode(sentence, normalize=False)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

        # Norm should not necessarily be 1.0
        norm = np.linalg.norm(embedding)
        assert norm > 0  # But should be non-zero

    def test_caching_functionality(self, encoder):
        """Test that caching works correctly."""
        sentence = "This sentence will be cached."

        # First encoding (cache miss)
        start_time = time.time()
        embedding1 = encoder.encode(sentence)
        first_time = time.time() - start_time

        # Second encoding (cache hit)
        start_time = time.time()
        embedding2 = encoder.encode(sentence)
        second_time = time.time() - start_time

        # Embeddings should be identical
        np.testing.assert_array_equal(embedding1, embedding2)

        # Second encoding should be faster (cache hit)
        # Note: This might not always be true for very fast operations
        # so we just check that cache stats are updated
        stats = encoder.get_performance_stats()
        assert stats["cache_hit_rate"] > 0

    def test_disable_caching(self, encoder):
        """Test disabling cache functionality."""
        encoder.set_cache_enabled(False)

        sentence = "Test sentence for disabled cache."
        embedding1 = encoder.encode(sentence)
        embedding2 = encoder.encode(sentence)

        # Embeddings should still be equal (deterministic)
        np.testing.assert_allclose(embedding1, embedding2, rtol=1e-5)

        # Cache should be empty
        stats = encoder.get_performance_stats()
        assert stats["cache_size"] == 0

        # Re-enable caching
        encoder.set_cache_enabled(True)

    def test_clear_cache(self, encoder):
        """Test clearing the cache."""
        # Add some items to cache
        sentences = ["Cache item 1", "Cache item 2", "Cache item 3"]
        for sentence in sentences:
            encoder.encode(sentence)

        stats = encoder.get_performance_stats()
        assert stats["cache_size"] > 0

        # Clear cache
        encoder.clear_cache()

        stats = encoder.get_performance_stats()
        assert stats["cache_size"] == 0
        assert stats["cache_hit_rate"] == 0.0

    def test_encode_batch_method(self, encoder):
        """Test the encode_batch method for large batches."""
        # Create a large batch that will be split
        sentences = [f"Test sentence number {i}" for i in range(100)]

        embeddings = encoder.encode_batch(sentences)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (100, 384)
        assert np.isfinite(embeddings).all()

    def test_performance_stats(self, encoder):
        """Test performance statistics tracking."""
        # Clear any previous stats
        encoder._encoding_times.clear()
        encoder._cache_hits = 0
        encoder._cache_misses = 0

        # Encode some sentences
        sentences = [
            "Test sentence 1",
            "Test sentence 2",
            "Test sentence 1",
        ]  # Repeat for cache hit
        for sentence in sentences:
            encoder.encode(sentence)

        stats = encoder.get_performance_stats()

        assert "avg_encoding_time_ms" in stats
        assert "min_encoding_time_ms" in stats
        assert "max_encoding_time_ms" in stats
        assert "cache_hit_rate" in stats
        assert "total_encodings" in stats
        assert "cache_size" in stats

        assert stats["total_encodings"] == 3
        assert stats["cache_hit_rate"] > 0  # Should have at least one cache hit

    def test_warmup(self, encoder):
        """Test model warmup functionality."""
        # Clear stats before warmup
        encoder._encoding_times.clear()

        encoder.warmup(num_iterations=5)

        # Stats should be cleared after warmup
        assert len(encoder._encoding_times) == 0

    def test_empty_input(self, encoder):
        """Test handling of empty input."""
        # Empty string
        embedding = encoder.encode("")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

        # Empty list
        embeddings = encoder.encode([])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0, 384)

    def test_long_input_truncation(self, encoder):
        """Test that long inputs are properly truncated."""
        # Create a very long sentence
        long_sentence = " ".join(["word"] * 1000)

        embedding = encoder.encode(long_sentence)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert np.isfinite(embedding).all()

    def test_special_characters(self, encoder):
        """Test encoding sentences with special characters."""
        sentences = [
            "Test with émojis 😊 and symbols!",
            "Numbers 123 and punctuation...",
            "New\nlines\nand\ttabs",
            "Unicode: 你好世界",
        ]

        embeddings = encoder.encode(sentences)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (4, 384)
        assert np.isfinite(embeddings).all()

    def test_singleton_pattern(self):
        """Test the get_encoder singleton pattern."""
        encoder1 = get_encoder()
        encoder2 = get_encoder()

        # Should be the same instance
        assert encoder1 is encoder2

        # Test with custom config (should still return existing instance)
        custom_config = EncoderConfig(max_length=256)
        encoder3 = get_encoder(custom_config)

        assert encoder3 is encoder1

    @pytest.mark.parametrize(
        "num_sentences,expected_shape",
        [
            (1, (384,)),
            (5, (5, 384)),
            (32, (32, 384)),  # Exactly one batch
            (50, (50, 384)),  # More than one batch
        ],
    )
    def test_various_batch_sizes(self, encoder, num_sentences, expected_shape):
        """Test encoding various batch sizes."""
        sentences = [f"Test sentence {i}" for i in range(num_sentences)]

        if num_sentences == 1:
            embeddings = encoder.encode(sentences[0])
        else:
            embeddings = encoder.encode(sentences)

        assert embeddings.shape == expected_shape
        assert np.isfinite(embeddings).all()

    def test_deterministic_encoding(self, encoder):
        """Test that encoding is deterministic."""
        sentence = "This should produce the same embedding every time."

        embeddings = []
        for _ in range(3):
            # Clear cache to ensure fresh encoding
            encoder.clear_cache()
            embedding = encoder.encode(sentence, use_cache=False)
            embeddings.append(embedding)

        # All embeddings should be identical
        for i in range(1, len(embeddings)):
            np.testing.assert_allclose(
                embeddings[0],
                embeddings[i],
                rtol=1e-5,
                err_msg=f"Embedding {i} differs from embedding 0",
            )

    def test_performance_requirements(self, encoder):
        """Test that encoder meets performance requirements."""
        # Warm up the model
        encoder.warmup(num_iterations=5)

        # Test single sentence encoding
        sentence = "This is a typical sentence for performance testing."

        times = []
        for _ in range(10):
            start = time.time()
            encoder.encode(sentence, use_cache=False)
            times.append((time.time() - start) * 1000)  # Convert to ms

        avg_time = np.mean(times)

        # Should be < 100ms as per requirements
        assert avg_time < 100, f"Average encoding time {avg_time:.2f}ms exceeds 100ms requirement"

        print(f"Average encoding time: {avg_time:.2f}ms")


@pytest.mark.integration
class TestONNXEncoderIntegration:
    """Integration tests for ONNX encoder."""

    def test_model_files_exist(self):
        """Test that model files exist in expected locations."""
        config = EncoderConfig()

        assert Path(config.model_path).exists(), f"Model file not found at {config.model_path}"
        assert Path(
            config.tokenizer_path
        ).exists(), f"Tokenizer directory not found at {config.tokenizer_path}"
        assert Path(config.config_path).exists(), f"Config file not found at {config.config_path}"

        # Check specific tokenizer files
        tokenizer_files = ["tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]
        for file in tokenizer_files:
            file_path = Path(config.tokenizer_path) / file
            assert file_path.exists(), f"Tokenizer file not found: {file_path}"

    def test_consistency_across_instances(self):
        """Test that different encoder instances produce same results."""
        sentence = "Test sentence for consistency."

        # Create two separate instances
        encoder1 = ONNXEncoder()
        encoder2 = ONNXEncoder()

        embedding1 = encoder1.encode(sentence, use_cache=False)
        embedding2 = encoder2.encode(sentence, use_cache=False)

        np.testing.assert_allclose(embedding1, embedding2, rtol=1e-5)
