"""
Embedding engine for generating 400D cognitive vectors.

This module coordinates the ONNX encoder and dimension extractors
to create complete cognitive vectors for the memory system.
"""

import logging
from typing import List, Dict, Optional, Any
import numpy as np
from dataclasses import dataclass
import asyncio

from .onnx_encoder import EncoderConfig, get_encoder
from .vector_manager import get_vector_manager
from src.models.entities import Vector, Memory
from src.extraction.dimensions.dimension_analyzer import DimensionAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingEngineConfig:
    """Configuration for the embedding engine."""

    encoder_config: Optional[EncoderConfig] = None
    enable_caching: bool = True
    batch_size: int = 32
    warmup_on_init: bool = True


class EmbeddingEngine:
    """
    Main engine for generating 400D cognitive vectors.

    This engine coordinates:
    - ONNX encoder for 384D semantic embeddings
    - Dimension analyzer for 16D cognitive features
    - Vector manager for composition and validation
    """

    def __init__(self, config: Optional[EmbeddingEngineConfig] = None):
        """
        Initialize the embedding engine.

        Args:
            config: Engine configuration
        """
        self.config = config or EmbeddingEngineConfig()

        # Initialize components
        self.encoder = get_encoder(self.config.encoder_config)
        self.vector_manager = get_vector_manager()
        self.dimension_analyzer = DimensionAnalyzer()

        # Enable/disable caching
        self.encoder.set_cache_enabled(self.config.enable_caching)

        # Warmup if requested
        if self.config.warmup_on_init:
            self.warmup()

        logger.info(
            f"EmbeddingEngine initialized "
            f"(dim={self.encoder.get_embedding_dimension() + 16}D, "
            f"cache={'enabled' if self.config.enable_caching else 'disabled'})"
        )

    def warmup(self, iterations: int = 5) -> None:
        """
        Warm up the engine for optimal performance.

        Args:
            iterations: Number of warmup iterations
        """
        logger.info(f"Warming up embedding engine with {iterations} iterations...")

        # Warmup encoder
        self.encoder.warmup(iterations)

        # Warmup dimension analyzer
        test_texts = [
            "This is a warmup sentence for the analyzer.",
            "Testing emotional content with excitement!",
            "Meeting scheduled for tomorrow at 2 PM.",
        ]

        for _ in range(iterations):
            for text in test_texts:
                self.dimension_analyzer.analyze(text)

        logger.info("Embedding engine warmup completed")

    async def create_vector(self, text: str, context: Optional[Dict[str, Any]] = None) -> Vector:
        """
        Create a complete 400D cognitive vector from text.

        Args:
            text: Input text
            context: Optional context information

        Returns:
            Complete Vector object
        """
        # Generate semantic embedding (384D)
        semantic_task = asyncio.create_task(self._encode_semantic_async(text))

        # Extract cognitive dimensions (16D)
        dimensions_task = asyncio.create_task(self._extract_dimensions_async(text, context))

        # Wait for both components
        semantic_embedding, cognitive_dimensions = await asyncio.gather(
            semantic_task, dimensions_task
        )

        # Compose into vector
        vector = self.vector_manager.compose_vector(semantic_embedding, cognitive_dimensions)

        # Validate
        is_valid, error = self.vector_manager.validate_vector(vector)
        if not is_valid:
            logger.warning(f"Invalid vector created: {error}")
            # Normalize to fix issues
            vector = self.vector_manager.normalize_vector(vector)

        return vector

    async def create_vectors_batch(
        self, texts: List[str], contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[Vector]:
        """
        Create multiple vectors in batch for efficiency.

        Args:
            texts: List of input texts
            contexts: Optional list of contexts

        Returns:
            List of Vector objects
        """
        if not texts:
            return []

        # Ensure contexts match texts
        if contexts is None:
            contexts = [None] * len(texts)
        elif len(contexts) != len(texts):
            raise ValueError("Number of contexts must match number of texts")

        # Process in batches
        vectors = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_contexts = contexts[i : i + batch_size]

            # Create tasks for this batch
            tasks = [
                self.create_vector(text, context)
                for text, context in zip(batch_texts, batch_contexts)
            ]

            # Process batch concurrently
            batch_vectors = await asyncio.gather(*tasks)
            vectors.extend(batch_vectors)

        return vectors

    async def _encode_semantic_async(self, text: str) -> np.ndarray:
        """
        Asynchronously encode text to semantic embedding.

        Args:
            text: Input text

        Returns:
            384D semantic embedding
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encoder.encode, text)

    async def _extract_dimensions_async(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Asynchronously extract cognitive dimensions.

        Args:
            text: Input text
            context: Optional context

        Returns:
            16D cognitive dimensions
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.dimension_analyzer.analyze, text, context)

    def create_vector_sync(self, text: str, context: Optional[Dict[str, Any]] = None) -> Vector:
        """
        Synchronous version of create_vector for compatibility.

        Args:
            text: Input text
            context: Optional context

        Returns:
            Complete Vector object
        """
        # Generate components
        semantic_embedding = self.encoder.encode(text)
        cognitive_dimensions = self.dimension_analyzer.analyze(text, context)

        # Compose and validate
        vector = self.vector_manager.compose_vector(semantic_embedding, cognitive_dimensions)

        # Ensure validity
        is_valid, error = self.vector_manager.validate_vector(vector)
        if not is_valid:
            vector = self.vector_manager.normalize_vector(vector)

        return vector

    def create_vectors_batch_sync(
        self, texts: List[str], contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[Vector]:
        """
        Synchronous batch vector creation.

        Args:
            texts: List of input texts
            contexts: Optional list of contexts

        Returns:
            List of Vector objects
        """
        if not texts:
            return []

        # Batch encode semantics
        semantic_embeddings = self.encoder.encode_batch(texts)

        # Extract dimensions for each text
        if contexts is None:
            contexts = [None] * len(texts)

        cognitive_dimensions = []
        for text, context in zip(texts, contexts):
            dims = self.dimension_analyzer.analyze(text, context)
            cognitive_dimensions.append(dims)

        # Compose vectors
        cognitive_array = np.vstack(cognitive_dimensions)
        vectors = self.vector_manager.batch_compose(semantic_embeddings, cognitive_array)

        return vectors

    def update_memory_vector(
        self, memory: Memory, context: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """
        Update a memory's vector based on its content.

        Args:
            memory: Memory object to update
            context: Optional context

        Returns:
            Updated memory with new vector
        """
        # Create new vector
        vector = self.create_vector_sync(memory.content, context)

        # Update memory
        memory.vector = vector

        return memory

    def compute_similarity(
        self,
        vector1: Vector,
        vector2: Vector,
        semantic_weight: float = 0.7,
        cognitive_weight: float = 0.3,
    ) -> float:
        """
        Compute similarity between two vectors.

        Args:
            vector1: First vector
            vector2: Second vector
            semantic_weight: Weight for semantic similarity
            cognitive_weight: Weight for cognitive similarity

        Returns:
            Similarity score in [0, 1]
        """
        return self.vector_manager.compute_similarity(
            vector1, vector2, semantic_weight, cognitive_weight
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary of statistics
        """
        encoder_stats = self.encoder.get_performance_stats()

        return {
            "encoder": encoder_stats,
            "embedding_dimension": self.encoder.get_embedding_dimension(),
            "cognitive_dimensions": 16,
            "total_dimensions": 400,
            "cache_enabled": self.config.enable_caching,
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.encoder.clear_cache()
        logger.info("Embedding engine cache cleared")


# Singleton instance
_engine_instance: Optional[EmbeddingEngine] = None


def get_embedding_engine(config: Optional[EmbeddingEngineConfig] = None) -> EmbeddingEngine:
    """
    Get or create the global embedding engine instance.

    Args:
        config: Engine configuration (used only on first call)

    Returns:
        The global engine instance
    """
    global _engine_instance

    if _engine_instance is None:
        _engine_instance = EmbeddingEngine(config)

    return _engine_instance


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_engine():
        # Create engine
        engine = EmbeddingEngine()

        # Test single vector creation
        text = "This is an important decision about project timeline."
        vector = await engine.create_vector(text)
        print(f"Created vector: {vector.full_vector.shape}")

        # Test batch creation
        texts = [
            "We need to discuss the budget allocation.",
            "The deadline has been moved to next Friday.",
            "Team morale is improving significantly.",
        ]
        vectors = await engine.create_vectors_batch(texts)
        print(f"Created {len(vectors)} vectors")

        # Test similarity
        sim = engine.compute_similarity(vectors[0], vectors[1])
        print(f"Similarity between first two: {sim:.3f}")

        # Show stats
        stats = engine.get_stats()
        print(f"Engine stats: {stats}")

    # Run test
    asyncio.run(test_engine())
