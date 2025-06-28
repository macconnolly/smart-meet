"""
Vector manager for composing and decomposing 400D cognitive vectors.

This module handles the composition of 384D semantic embeddings with
16D cognitive dimensions, creating the full 400D vectors used throughout
the system.
"""

import logging
from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
import json

from src.models.entities import Vector
from src.extraction.dimensions.dimension_analyzer import CognitiveDimensions

logger = logging.getLogger(__name__)


@dataclass
class VectorStats:
    """Statistics about a vector or set of vectors."""

    semantic_norm: float
    dimensions_mean: float
    dimensions_std: float
    dimensions_min: float
    dimensions_max: float
    is_normalized: bool
    is_valid: bool


class VectorManager:
    """
    Manages the composition and decomposition of 400D cognitive vectors.

    The 400D vector consists of:
    - 384D: Semantic embedding from ONNX encoder (normalized)
    - 16D: Cognitive dimensions (values in [0, 1])

    Cognitive dimensions (16D):
    - Temporal (4D): urgency, deadline_proximity, sequence_position, duration_relevance
    - Emotional (3D): polarity, intensity, confidence
    - Social (3D): authority, influence, team_dynamics
    - Causal (3D): dependencies, impact, risk_factors
    - Strategic (3D): alignment, innovation, value
    """

    SEMANTIC_DIM = 384
    COGNITIVE_DIM = 16
    TOTAL_DIM = 400

    # Dimension indices for easy access
    TEMPORAL_START = 0
    TEMPORAL_END = 4
    EMOTIONAL_START = 4
    EMOTIONAL_END = 7
    SOCIAL_START = 7
    SOCIAL_END = 10
    CAUSAL_START = 10
    CAUSAL_END = 13
    STRATEGIC_START = 13
    STRATEGIC_END = 16

    def __init__(self):
        """Initialize the vector manager."""
        self._stats_cache: Dict[str, VectorStats] = {}

    def compose_vector(
        self, semantic_embedding: np.ndarray, cognitive_dimensions: CognitiveDimensions
    ) -> np.ndarray:
        """
        Composes a 400D vector from a 384D semantic embedding and 16D cognitive dimensions.

        Args:
            semantic_embedding: 384D semantic embedding
            cognitive_dimensions: CognitiveDimensions object

        Returns:
            Vector object

        Raises:
            ValueError: If input dimensions are incorrect
        """
        if semantic_embedding.shape != (384,):
            raise ValueError(f"Semantic embedding must be 384D, got {semantic_embedding.shape}")

        cognitive_array = cognitive_dimensions.to_array()
        if cognitive_array.shape != (16,):
            raise ValueError(f"Cognitive dimensions must be 16D, got {cognitive_array.shape}")

        return np.concatenate([semantic_embedding, cognitive_array])

    def decompose_vector(self, vector: Vector) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose a vector into semantic and cognitive components.

        Args:
            vector: Vector object to decompose

        Returns:
            Tuple of (semantic_embedding, cognitive_dimensions)
        """
        return vector.semantic.copy(), vector.dimensions.copy()

    def validate_vector(self, vector: Vector) -> Tuple[bool, Optional[str]]:
        """
        Validate a vector for correctness.

        Args:
            vector: Vector to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check dimensions
        if vector.semantic.shape != (self.SEMANTIC_DIM,):
            return False, f"Semantic dimension mismatch: {vector.semantic.shape}"

        if vector.dimensions.shape != (self.COGNITIVE_DIM,):
            return False, f"Cognitive dimension mismatch: {vector.dimensions.shape}"

        # Check semantic normalization (with tolerance)
        semantic_norm = np.linalg.norm(vector.semantic)
        if abs(semantic_norm - 1.0) > 1e-6 and semantic_norm > 0:
            return False, f"Semantic vector not normalized: norm={semantic_norm}"

        # Check cognitive dimensions range
        if not np.all((vector.dimensions >= 0) & (vector.dimensions <= 1)):
            return False, "Cognitive dimensions outside [0, 1] range"

        # Check for NaN or inf
        if np.any(np.isnan(vector.semantic)) or np.any(np.isinf(vector.semantic)):
            return False, "Semantic vector contains NaN or inf"

        if np.any(np.isnan(vector.dimensions)) or np.any(np.isinf(vector.dimensions)):
            return False, "Cognitive dimensions contain NaN or inf"

        return True, None

    def normalize_vector(self, vector: Vector) -> Vector:
        """
        Normalize a vector to ensure it meets requirements.

        Args:
            vector: Vector to normalize

        Returns:
            Normalized Vector object
        """
        # Normalize semantic component
        semantic_norm = np.linalg.norm(vector.semantic)
        if semantic_norm > 0:
            normalized_semantic = vector.semantic / semantic_norm
        else:
            # Handle zero vector
            normalized_semantic = np.zeros(self.SEMANTIC_DIM)
            normalized_semantic[0] = 1.0  # Set first component to 1

        # Clip cognitive dimensions to [0, 1]
        normalized_dimensions = np.clip(vector.dimensions, 0, 1)

        return Vector(semantic=normalized_semantic, dimensions=normalized_dimensions)

    def get_vector_stats(self, vector: Vector) -> VectorStats:
        """
        Get statistics about a vector.

        Args:
            vector: Vector to analyze

        Returns:
            VectorStats object
        """
        semantic_norm = np.linalg.norm(vector.semantic)
        is_normalized = abs(semantic_norm - 1.0) < 1e-6 or semantic_norm == 0

        dims = vector.dimensions
        is_valid, _ = self.validate_vector(vector)

        return VectorStats(
            semantic_norm=semantic_norm,
            dimensions_mean=float(np.mean(dims)),
            dimensions_std=float(np.std(dims)),
            dimensions_min=float(np.min(dims)),
            dimensions_max=float(np.max(dims)),
            is_normalized=is_normalized,
            is_valid=is_valid,
        )

    def batch_compose(
        self, semantic_embeddings: np.ndarray, cognitive_dimensions: List[CognitiveDimensions]
    ) -> List[Vector]:
        """
        Compose multiple vectors in batch.

        Args:
            semantic_embeddings: Array of shape (n, 384)
            cognitive_dimensions: Array of shape (n, 16)

        Returns:
            List of Vector objects
        """
        if semantic_embeddings.shape[0] != len(cognitive_dimensions):
            raise ValueError("Number of semantic embeddings and cognitive dimensions must match")

        vectors = []
        for i in range(semantic_embeddings.shape[0]):
            vector = self.compose_vector(semantic_embeddings[i], cognitive_dimensions[i])
            vectors.append(vector)

        return vectors

    def vectors_to_array(self, vectors: List[Vector]) -> np.ndarray:
        """
        Convert list of vectors to numpy array.

        Args:
            vectors: List of Vector objects

        Returns:
            Array of shape (n, 400)
        """
        if not vectors:
            return np.array([]).reshape(0, self.TOTAL_DIM)

        return np.vstack([v.full_vector for v in vectors])

    def array_to_vectors(self, array: np.ndarray) -> List[Vector]:
        """
        Convert numpy array to list of vectors.

        Args:
            array: Array of shape (n, 400)

        Returns:
            List of Vector objects
        """
        if array.shape[1] != self.TOTAL_DIM:
            raise ValueError(f"Array must have {self.TOTAL_DIM} columns")

        vectors = []
        for i in range(array.shape[0]):
            vector = Vector(
                semantic=array[i, : self.SEMANTIC_DIM], dimensions=array[i, self.SEMANTIC_DIM :]
            )
            vectors.append(vector)

        return vectors

    def get_dimension_names(self) -> Dict[str, List[str]]:
        """
        Get human-readable names for all dimensions.

        Returns:
            Dictionary mapping dimension groups to dimension names
        """
        return {
            "temporal": [
                "urgency",
                "deadline_proximity",
                "sequence_position",
                "duration_relevance",
            ],
            "emotional": ["polarity", "intensity", "confidence"],
            "social": ["authority", "influence", "team_dynamics"],
            "causal": ["dependencies", "impact", "risk_factors"],
            "strategic": ["alignment", "innovation", "value"],
        }

    def get_dimension_value(self, vector: Vector, dimension_name: str) -> float:
        """
        Get a specific cognitive dimension value by name.

        Args:
            vector: Vector object
            dimension_name: Name of the dimension

        Returns:
            Dimension value

        Raises:
            ValueError: If dimension name is invalid
        """
        dimension_map = {
            # Temporal
            "urgency": 0,
            "deadline_proximity": 1,
            "sequence_position": 2,
            "duration_relevance": 3,
            # Emotional
            "polarity": 4,
            "intensity": 5,
            "confidence": 6,
            # Social
            "authority": 7,
            "influence": 8,
            "team_dynamics": 9,
            # Causal
            "dependencies": 10,
            "impact": 11,
            "risk_factors": 12,
            # Strategic
            "alignment": 13,
            "innovation": 14,
            "value": 15,
        }

        if dimension_name not in dimension_map:
            raise ValueError(f"Unknown dimension: {dimension_name}")

        return float(vector.dimensions[dimension_map[dimension_name]])

    def set_dimension_value(self, vector: Vector, dimension_name: str, value: float) -> Vector:
        """
        Set a specific cognitive dimension value.

        Args:
            vector: Vector object
            dimension_name: Name of the dimension
            value: New value (will be clipped to [0, 1])

        Returns:
            New Vector object with updated dimension
        """
        dimension_map = {
            # Temporal
            "urgency": 0,
            "deadline_proximity": 1,
            "sequence_position": 2,
            "duration_relevance": 3,
            # Emotional
            "polarity": 4,
            "intensity": 5,
            "confidence": 6,
            # Social
            "authority": 7,
            "influence": 8,
            "team_dynamics": 9,
            # Causal
            "dependencies": 10,
            "impact": 11,
            "risk_factors": 12,
            # Strategic
            "alignment": 13,
            "innovation": 14,
            "value": 15,
        }

        if dimension_name not in dimension_map:
            raise ValueError(f"Unknown dimension: {dimension_name}")

        # Create copy and update
        new_dimensions = vector.dimensions.copy()
        new_dimensions[dimension_map[dimension_name]] = np.clip(value, 0, 1)

        return Vector(semantic=vector.semantic.copy(), dimensions=new_dimensions)

    def compute_similarity(
        self,
        vector1: Vector,
        vector2: Vector,
        semantic_weight: float = 0.7,
        cognitive_weight: float = 0.3,
    ) -> float:
        """
        Compute weighted similarity between two vectors.

        Args:
            vector1: First vector
            vector2: Second vector
            semantic_weight: Weight for semantic similarity
            cognitive_weight: Weight for cognitive similarity

        Returns:
            Similarity score in [0, 1]
        """
        # Normalize weights
        total_weight = semantic_weight + cognitive_weight
        semantic_weight /= total_weight
        cognitive_weight /= total_weight

        # Compute semantic similarity (cosine)
        semantic_sim = np.dot(vector1.semantic, vector2.semantic)

        # Compute cognitive similarity (1 - normalized L2 distance)
        cognitive_dist = np.linalg.norm(vector1.dimensions - vector2.dimensions)
        cognitive_sim = 1 - (cognitive_dist / np.sqrt(self.COGNITIVE_DIM))

        # Weighted combination
        total_sim = semantic_weight * semantic_sim + cognitive_weight * cognitive_sim

        return float(np.clip(total_sim, 0, 1))

    def to_json(self, vector: Vector) -> str:
        """
        Serialize vector to JSON string.

        Args:
            vector: Vector to serialize

        Returns:
            JSON string representation
        """
        data = {"semantic": vector.semantic.tolist(), "dimensions": vector.dimensions.tolist()}
        return json.dumps(data)

    def from_json(self, json_str: str) -> Vector:
        """
        Deserialize vector from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            Vector object
        """
        data = json.loads(json_str)
        return Vector(semantic=np.array(data["semantic"]), dimensions=np.array(data["dimensions"]))


# Singleton instance
_vector_manager_instance: Optional[VectorManager] = None


def get_vector_manager() -> VectorManager:
    """Get or create the global vector manager instance."""
    global _vector_manager_instance

    if _vector_manager_instance is None:
        _vector_manager_instance = VectorManager()

    return _vector_manager_instance


# Example usage and testing
if __name__ == "__main__":
    # Create manager
    manager = VectorManager()

    # Test vector composition
    semantic = np.random.randn(384)
    semantic = semantic / np.linalg.norm(semantic)  # Normalize

    cognitive = np.array(
        [
            # Temporal
            0.8,
            0.7,
            0.5,
            0.6,
            # Emotional
            0.9,
            0.8,
            0.7,
            # Social
            0.6,
            0.7,
            0.8,
            # Causal
            0.5,
            0.6,
            0.7,
            # Strategic
            0.8,
            0.9,
            0.7,
        ]
    )

    # Compose vector
    vector = manager.compose_vector(semantic, cognitive)
    print(f"Vector shape: {vector.full_vector.shape}")

    # Validate vector
    is_valid, error = manager.validate_vector(vector)
    print(f"Valid: {is_valid}, Error: {error}")

    # Get stats
    stats = manager.get_vector_stats(vector)
    print(f"Stats: {stats}")

    # Test dimension access
    urgency = manager.get_dimension_value(vector, "urgency")
    print(f"Urgency: {urgency}")

    # Test similarity
    vector2 = manager.compose_vector(semantic + np.random.randn(384) * 0.1, cognitive * 0.9)
    similarity = manager.compute_similarity(vector, vector2)
    print(f"Similarity: {similarity}")
