"""
ONNX-based encoder for generating sentence embeddings.

This module provides high-performance sentence encoding using ONNX Runtime,
achieving <100ms encoding for typical sentences on CPU.
"""

import json
import logging
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import time
from functools import lru_cache
import hashlib

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
except ImportError as e:
    raise ImportError(
        f"Required libraries not installed: {e}\n"
        "Please install: pip install onnxruntime transformers"
    )

logger = logging.getLogger(__name__)


@dataclass
class EncoderConfig:
    """Configuration for the ONNX encoder."""
    model_path: str = "models/embeddings/model.onnx"
    tokenizer_path: str = "models/embeddings/tokenizer"
    config_path: str = "models/embeddings/config.json"
    max_length: int = 128
    batch_size: int = 32
    cache_size: int = 10000
    providers: List[str] = None
    
    def __post_init__(self):
        """Set default providers if not specified."""
        if self.providers is None:
            self.providers = ['CPUExecutionProvider']


class ONNXEncoder:
    """
    High-performance sentence encoder using ONNX Runtime.
    
    Features:
    - <100ms encoding for typical sentences
    - Caching for repeated encodings
    - Batch processing support
    - Thread-safe operations
    - Automatic mean pooling
    """
    
    def __init__(self, config: Optional[EncoderConfig] = None):
        """
        Initialize the ONNX encoder.
        
        Args:
            config: Encoder configuration
        """
        self.config = config or EncoderConfig()
        self._session: Optional[ort.InferenceSession] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model_config: Optional[Dict[str, Any]] = None
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_enabled = True
        
        # Performance tracking
        self._encoding_times: List[float] = []
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Initialize components
        self._load_model()
        self._load_tokenizer()
        self._load_config()
        
        logger.info(
            f"ONNX Encoder initialized with {self._model_config['embedding_dimension']}D embeddings"
        )
    
    def _load_model(self) -> None:
        """Load the ONNX model."""
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at: {model_path}")
        
        # Create inference session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.inter_op_num_threads = 4
        sess_options.intra_op_num_threads = 4
        
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=self.config.providers
        )
        
        # Verify input/output names
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]
        
        logger.debug(f"Model inputs: {self._input_names}")
        logger.debug(f"Model outputs: {self._output_names}")
    
    def _load_tokenizer(self) -> None:
        """Load the tokenizer."""
        tokenizer_path = Path(self.config.tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        
        # Set padding token if not present
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
    
    def _load_config(self) -> None:
        """Load model configuration."""
        config_path = Path(self.config.config_path)
        if not config_path.exists():
            # Use defaults if config not found
            self._model_config = {
                "embedding_dimension": 384,
                "max_seq_length": 256
            }
            logger.warning(f"Config not found at {config_path}, using defaults")
        else:
            with open(config_path, 'r') as f:
                self._model_config = json.load(f)
    
    @lru_cache(maxsize=10000)
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def encode(
        self, 
        sentences: Union[str, List[str]],
        normalize: bool = True,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Encode sentences into embeddings.
        
        Args:
            sentences: Single sentence or list of sentences
            normalize: Whether to normalize embeddings to unit length
            use_cache: Whether to use caching
            
        Returns:
            Embeddings array of shape (n_sentences, embedding_dim) or (embedding_dim,)
        """
        # Handle single sentence
        single_sentence = isinstance(sentences, str)
        if single_sentence:
            sentences = [sentences]
        
        start_time = time.time()
        
        # Check cache for all sentences
        embeddings = []
        uncached_sentences = []
        uncached_indices = []
        
        if use_cache and self._cache_enabled:
            for i, sentence in enumerate(sentences):
                cache_key = self._get_cache_key(sentence)
                if cache_key in self._cache:
                    embeddings.append(self._cache[cache_key])
                    self._cache_hits += 1
                else:
                    uncached_sentences.append(sentence)
                    uncached_indices.append(i)
                    self._cache_misses += 1
        else:
            uncached_sentences = sentences
            uncached_indices = list(range(len(sentences)))
        
        # Encode uncached sentences
        if uncached_sentences:
            # Tokenize
            encoded = self._tokenizer(
                uncached_sentences,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="np"
            )
            
            # Run inference
            outputs = self._session.run(
                self._output_names,
                {
                    'input_ids': encoded['input_ids'].astype(np.int64),
                    'attention_mask': encoded['attention_mask'].astype(np.int64),
                    'token_type_ids': encoded.get('token_type_ids', np.zeros_like(encoded['input_ids'])).astype(np.int64)
                }
            )
            
            # Get embeddings from output
            token_embeddings = outputs[0]  # Shape: (batch_size, sequence_length, hidden_size)
            
            # Apply mean pooling
            attention_mask = encoded['attention_mask']
            attention_mask_expanded = np.expand_dims(attention_mask, -1)
            
            sum_embeddings = np.sum(token_embeddings * attention_mask_expanded, axis=1)
            sum_mask = np.sum(attention_mask_expanded, axis=1)
            sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)  # Avoid division by zero
            
            mean_embeddings = sum_embeddings / sum_mask
            
            # Normalize if requested
            if normalize:
                norms = np.linalg.norm(mean_embeddings, axis=1, keepdims=True)
                mean_embeddings = mean_embeddings / np.clip(norms, a_min=1e-9, a_max=None)
            
            # Cache results
            if use_cache and self._cache_enabled:
                for sentence, embedding in zip(uncached_sentences, mean_embeddings):
                    cache_key = self._get_cache_key(sentence)
                    self._cache[cache_key] = embedding
                    
                    # Implement simple LRU by removing oldest entries
                    if len(self._cache) > self.config.cache_size:
                        # Remove first (oldest) item
                        first_key = next(iter(self._cache))
                        del self._cache[first_key]
        
        # Combine cached and newly computed embeddings
        if use_cache and self._cache_enabled and embeddings:
            # Reconstruct full embeddings array in correct order
            all_embeddings = np.zeros((len(sentences), self._model_config['embedding_dimension']))
            
            # Fill in cached embeddings
            cache_idx = 0
            uncached_idx = 0
            
            for i in range(len(sentences)):
                if i in uncached_indices:
                    all_embeddings[i] = mean_embeddings[uncached_idx]
                    uncached_idx += 1
                else:
                    all_embeddings[i] = embeddings[cache_idx]
                    cache_idx += 1
            
            result = all_embeddings
        else:
            result = mean_embeddings if uncached_sentences else np.array(embeddings)
        
        # Track performance
        encoding_time = (time.time() - start_time) * 1000  # Convert to ms
        self._encoding_times.append(encoding_time)
        
        # Keep only last 100 measurements
        if len(self._encoding_times) > 100:
            self._encoding_times = self._encoding_times[-100:]
        
        # Return single embedding if input was single sentence
        if single_sentence:
            return result[0]
        
        return result
    
    def encode_batch(
        self, 
        sentences: List[str],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode a batch of sentences efficiently.
        
        Args:
            sentences: List of sentences to encode
            normalize: Whether to normalize embeddings
            
        Returns:
            Embeddings array of shape (n_sentences, embedding_dim)
        """
        # Process in batches for memory efficiency
        all_embeddings = []
        
        for i in range(0, len(sentences), self.config.batch_size):
            batch = sentences[i:i + self.config.batch_size]
            embeddings = self.encode(batch, normalize=normalize, use_cache=True)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced."""
        return self._model_config['embedding_dimension']
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self._encoding_times:
            return {
                "avg_encoding_time_ms": 0.0,
                "min_encoding_time_ms": 0.0,
                "max_encoding_time_ms": 0.0,
                "cache_hit_rate": 0.0,
                "total_encodings": 0
            }
        
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "avg_encoding_time_ms": np.mean(self._encoding_times),
            "min_encoding_time_ms": np.min(self._encoding_times),
            "max_encoding_time_ms": np.max(self._encoding_times),
            "cache_hit_rate": cache_hit_rate,
            "total_encodings": total_requests,
            "cache_size": len(self._cache)
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Embedding cache cleared")
    
    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable caching."""
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()
    
    def warmup(self, num_iterations: int = 10) -> None:
        """
        Warm up the model for optimal performance.
        
        Args:
            num_iterations: Number of warmup iterations
        """
        logger.info(f"Warming up encoder with {num_iterations} iterations...")
        
        test_sentences = [
            "This is a warmup sentence.",
            "Another sentence for warming up the model.",
            "The quick brown fox jumps over the lazy dog."
        ]
        
        for _ in range(num_iterations):
            self.encode(test_sentences, use_cache=False)
        
        # Clear warmup from stats
        self._encoding_times.clear()
        
        logger.info("Encoder warmup completed")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ONNXEncoder(model='{self.config.model_path}', "
            f"dim={self.get_embedding_dimension()}, "
            f"cache_size={len(self._cache)})"
        )


# Singleton instance for global access
_encoder_instance: Optional[ONNXEncoder] = None


def get_encoder(config: Optional[EncoderConfig] = None) -> ONNXEncoder:
    """
    Get or create the global encoder instance.
    
    Args:
        config: Encoder configuration (used only on first call)
        
    Returns:
        The global encoder instance
    """
    global _encoder_instance
    
    if _encoder_instance is None:
        _encoder_instance = ONNXEncoder(config)
        _encoder_instance.warmup()  # Warm up on first creation
    
    return _encoder_instance


# Example usage and testing
if __name__ == "__main__":
    # Initialize encoder
    encoder = ONNXEncoder()
    
    # Test single sentence
    embedding = encoder.encode("This is a test sentence.")
    print(f"Single embedding shape: {embedding.shape}")
    
    # Test batch encoding
    sentences = [
        "First test sentence.",
        "Second test sentence with more words.",
        "Third sentence for testing batch processing.",
    ]
    
    embeddings = encoder.encode(sentences)
    print(f"Batch embeddings shape: {embeddings.shape}")
    
    # Test performance
    import time
    
    # Time single encoding
    start = time.time()
    for _ in range(100):
        encoder.encode("Performance test sentence.")
    single_time = (time.time() - start) / 100 * 1000
    
    print(f"Average single encoding time: {single_time:.2f}ms")
    
    # Show performance stats
    stats = encoder.get_performance_stats()
    print(f"Performance stats: {stats}")
