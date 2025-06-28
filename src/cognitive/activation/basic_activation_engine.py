"""
Basic activation engine implementation using BFS traversal.

This module implements the foundational activation spreading mechanism
that activates memories through breadth-first search traversal of the
memory connection graph.

Reference: IMPLEMENTATION_GUIDE.md - Phase 2: Activation Spreading
"""

import time
from collections import deque
from typing import List, Dict, Optional, Set

import numpy as np
from loguru import logger

from ...models.entities import Memory, MemoryConnection
from ...storage.sqlite.repositories import MemoryRepository, MemoryConnectionRepository
from ...storage.qdrant.vector_store import QdrantVectorStore, SearchFilter
from qdrant_client.http import models as rest
from ...extraction.dimensions.dimension_analyzer import CognitiveDimensions


def _build_cognitive_filter(
    dimensions: CognitiveDimensions,
    project_id: Optional[str],
    threshold: float = 0.7
) -> rest.Filter:
    """Builds a Qdrant filter based on significant cognitive dimensions."""
    must_conditions = []
    if project_id:
        must_conditions.append(
            rest.FieldCondition(
                key="project_id",
                match=rest.MatchValue(value=project_id),
            )
        )

    # Prioritize memories with similar high-scoring cognitive dimensions
    # Only add conditions if the dimension value is above a certain threshold
    if dimensions.temporal.urgency > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_temporal_urgency", range=rest.Range(gte=threshold))
        )
    if dimensions.temporal.deadline_proximity > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_temporal_deadline_proximity", range=rest.Range(gte=threshold))
        )
    if dimensions.temporal.sequence_position > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_temporal_sequence_position", range=rest.Range(gte=threshold))
        )
    if dimensions.temporal.duration_relevance > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_temporal_duration_relevance", range=rest.Range(gte=threshold))
        )

    if dimensions.emotional.polarity > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_emotional_polarity", range=rest.Range(gte=threshold))
        )
    if dimensions.emotional.intensity > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_emotional_intensity", range=rest.Range(gte=threshold))
        )
    if dimensions.emotional.confidence > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_emotional_confidence", range=rest.Range(gte=threshold))
        )

    if dimensions.social.authority > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_social_authority", range=rest.Range(gte=threshold))
        )
    if dimensions.social.influence > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_social_influence", range=rest.Range(gte=threshold))
        )
    if dimensions.social.team_dynamics > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_social_team_dynamics", range=rest.Range(gte=threshold))
        )

    if dimensions.causal.dependencies > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_causal_dependencies", range=rest.Range(gte=threshold))
        )
    if dimensions.causal.impact > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_causal_impact", range=rest.Range(gte=threshold))
        )
    if dimensions.causal.risk_factors > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_causal_risk_factors", range=rest.Range(gte=threshold))
        )

    if dimensions.strategic.alignment > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_strategic_alignment", range=rest.Range(gte=threshold))
        )
    if dimensions.strategic.innovation > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_strategic_innovation", range=rest.Range(gte=threshold))
        )
    if dimensions.strategic.value > threshold:
        must_conditions.append(
            rest.FieldCondition(key="dim_strategic_value", range=rest.Range(gte=threshold))
        )

    if must_conditions:
        return rest.Filter(must=must_conditions)
    return None


class ActivationResult:
    """Result of activation spreading process."""

    def __init__(self):
        self.core_memories: List[Memory] = []
        self.peripheral_memories: List[Memory] = []
        self.activation_strengths: Dict[str, float] = {}
        self.activation_paths: Dict[str, List[str]] = {} # Stores the path of activation for each memory
        self.activation_explanations: Dict[str, str] = {} # Stores explanations for activated memories
        self.activation_time_ms: float = 0.0
        self.total_activated: int = 0

    @property
    def total_activated(self) -> int:
        return len(self.core_memories) + len(self.peripheral_memories)

    def add_activated_memory(
        self,
        memory: Memory,
        strength: float,
        path: List[str],
        explanation: str,
        core_threshold: float,
        peripheral_threshold: float
    ):
        self.activation_strengths[memory.id] = strength
        self.activation_paths[memory.id] = path
        self.activation_explanations[memory.id] = explanation

        if strength >= core_threshold:
            self.core_memories.append(memory)
        elif strength >= peripheral_threshold:
            self.peripheral_memories.append(memory)


class BasicActivationEngine:
    """
    Basic activation engine using BFS traversal for memory activation.

    Implements context-driven activation spreading that starts with high-similarity
    memories at L0 (concepts) and spreads activation through the connection graph
    using breadth-first search with threshold-based filtering.

    This is the foundation that ConsultingActivationEngine will extend.
    """

    def __init__(
        self,
        memory_repo: MemoryRepository,
        connection_repo: MemoryConnectionRepository,
        vector_store: QdrantVectorStore,
        core_threshold: float = 0.7,
        peripheral_threshold: float = 0.5,
        decay_factor: float = 0.8
    ):
        """
        Initialize basic activation engine.

        Args:
            memory_repo: Repository for memory access
            connection_repo: Repository for connection access
            vector_store: Qdrant vector store for similarity search
            core_threshold: Threshold for core memory activation
            peripheral_threshold: Threshold for peripheral memory activation
            decay_factor: Factor for activation strength decay during spreading
        """
        self.memory_repo = memory_repo
        self.connection_repo = connection_repo
        self.vector_store = vector_store
        self.core_threshold = core_threshold
        self.peripheral_threshold = peripheral_threshold
        self.decay_factor = decay_factor

    async def activate_memories(
        self,
        context_vector: np.ndarray,
        query_cognitive_dimensions: CognitiveDimensions,
        threshold: float,
        max_activations: int = 50,
        project_id: Optional[str] = None
    ) -> ActivationResult:
        """
        Activate memories based on context with spreading activation.

        Implementation follows the algorithm specification:
        1. Find high-similarity L0 concepts as starting points (Phase 1)
        2. Use BFS to spread activation through connection graph (Phase 2)
        3. Apply threshold-based filtering to limit computational overhead
        4. Track activation strength for result ranking
        5. Generate explanations for activated memories

        Args:
            context: Context vector for similarity computation
            threshold: Minimum activation threshold
            max_activations: Maximum number of memories to activate
            project_id: Optional project filter

        Returns:
            ActivationResult with core and peripheral memories
        """
        start_time = time.time()
        result = ActivationResult()

        try:
            # Phase 1: Find high-similarity L0 concepts as starting points
            starting_memories = await self._find_starting_memories(
                context_vector, query_cognitive_dimensions, threshold, project_id
            )

            if not starting_memories:
                logger.debug("No starting memories found for activation")
                result.activation_time_ms = (time.time() - start_time) * 1000
                return result

            # Phase 2: BFS traversal through connection graph
            result = await self._bfs_activation(
                context_vector, starting_memories, threshold, max_activations, project_id
            )

            # Calculate timing
            result.activation_time_ms = (time.time() - start_time) * 1000

            logger.debug(
                "Memory activation completed",
                core_count=len(result.core_memories),
                peripheral_count=len(result.peripheral_memories),
                total_activated=result.total_activated,
                time_ms=result.activation_time_ms,
            )

            return result

        except Exception as e:
            logger.error("Memory activation failed", error=str(e))
            result.activation_time_ms = (time.time() - start_time) * 1000
            return result

    async def _find_starting_memories(
        self,
        context_vector: np.ndarray,
        query_cognitive_dimensions: CognitiveDimensions,
        threshold: float,
        project_id: Optional[str] = None
    ) -> List[Memory]:
        """
        Find L0 memories with high similarity to context as starting points,
        considering both semantic and cognitive dimensions.
        """
        qdrant_filter = _build_cognitive_filter(
            dimensions=query_cognitive_dimensions,
            project_id=project_id,
            threshold=0.7 # Use a higher threshold for filtering starting memories
        )

        search_filter = SearchFilter(project_id=project_id) # Basic project filter
        if qdrant_filter:
            # Combine filters if cognitive filter is present
            if search_filter.to_qdrant_filter():
                search_filter.to_qdrant_filter().must.append(qdrant_filter)
            else:
                search_filter = qdrant_filter

        search_results = await self.vector_store.search(
            query_vector=context_vector,
            level=0, # Always search L0 for starting concepts
            limit=10,
            filters=search_filter,
            score_threshold=threshold,
        )

        memory_ids = [res.payload.get("memory_id") for res in search_results]
        memories = [await self.memory_repo.get_by_id(mid) for mid in memory_ids if mid]
        memories = [m for m in memories if m] # Filter out None

        logger.debug(f"Found {len(memories)} starting memories for activation")
        return memories

    async def _bfs_activation(
        self,
        context: np.ndarray,
        starting_memories: List[Memory],
        threshold: float,
        max_activations: int,
        project_id: Optional[str] = None
    ) -> ActivationResult:
        """
        Perform BFS traversal to activate connected memories.

        Args:
            context: Context vector for similarity computation
            starting_memories: Starting memories for BFS
            threshold: Minimum activation threshold
            max_activations: Maximum number of memories to activate
            project_id: Optional project filter

        Returns:
            ActivationResult with activated memories
        """
        result = ActivationResult()

        # Initialize BFS structures
        queue = deque([(m, 1.0, 0, [m.id]) for m in starting_memories])  # (memory, strength, depth, path)
        activated_ids: Set[str] = set()

        # Process starting memories
        for memory in starting_memories:
            activated_ids.add(memory.id)
            explanation = self._generate_activation_explanation(memory, 1.0, [])
            result.add_activated_memory(memory, 1.0, [memory.id], explanation, self.core_threshold, self.peripheral_threshold)

        # BFS traversal through connection graph
        while queue and len(activated_ids) < max_activations:
            current_memory, current_strength, depth, path = queue.popleft()

            # Get connected memories
            connections = await self.connection_repo.get_connections_for_memory(
                current_memory.id,
                min_strength=self.peripheral_threshold
            )

            for connection in connections:
                target_id = connection.target_id if connection.source_id == current_memory.id else connection.source_id

                if target_id not in activated_ids:
                    # Get the connected memory
                    connected_memory = await self.memory_repo.get_by_id(target_id)
                    if not connected_memory:
                        continue

                    # Apply project filter if specified
                    if project_id and connected_memory.project_id != project_id:
                        continue

                    # Calculate activation strength with decay
                    strength = current_strength * self.decay_factor * connection.connection_strength

                    # Apply threshold filtering
                    if strength >= threshold:
                        activated_ids.add(target_id)
                        new_path = path + [target_id]
                        explanation = self._generate_activation_explanation(connected_memory, strength, new_path)
                        result.add_activated_memory(connected_memory, strength, new_path, explanation, self.core_threshold, self.peripheral_threshold)

                        # Add to queue for further traversal
                        queue.append((connected_memory, strength, depth + 1, new_path))

                        # Check activation limit
                        if len(activated_ids) >= max_activations:
                            break

        # Sort memories by activation strength
        result.core_memories.sort(
            key=lambda m: result.activation_strengths.get(m.id, 0.0),
            reverse=True
        )
        result.peripheral_memories.sort(
            key=lambda m: result.activation_strengths.get(m.id, 0.0),
            reverse=True
        )

        return result

    def _generate_activation_explanation(
        self,
        memory: Memory,
        strength: float,
        path: List[str]
    ) -> str:
        """
        Generates a human-readable explanation for why a memory was activated.
        """
        explanation_parts = []

        if not path or len(path) == 1: # Starting memory
            explanation_parts.append(f"This memory was directly activated as a starting point.")
        else:
            explanation_parts.append(f"This memory was activated through its connection to memory '{path[-2]}'.")

        explanation_parts.append(f"It has an activation strength of {strength:.2f}.")

        if memory.memory_type:
            explanation_parts.append(f"It is a '{memory.memory_type}' type memory.")
        if memory.speaker:
            explanation_parts.append(f"It was contributed by '{memory.speaker}'.")
        if memory.timestamp_ms:
            explanation_parts.append(f"It occurred around {time.ctime(memory.timestamp_ms / 1000)}.")
        if memory.project_id:
            explanation_parts.append(f"It is associated with project '{memory.project_id}'.")

        # Add details from enhanced cognitive dimensions
        if memory.dimensions_json:
            try:
                import json
                dims = json.loads(memory.dimensions_json)
                if dims.get("temporal", {}).get("urgency", 0) > 0.7:
                    explanation_parts.append(f"It is a highly urgent memory (urgency: {dims['temporal']['urgency']:.2f}).")
                if dims.get("causal", {}).get("impact", 0) > 0.7:
                    explanation_parts.append(f"It highlights a significant impact (impact: {dims['causal']['impact']:.2f}).")
                if dims.get("social", {}).get("authority", 0) > 0.7:
                    explanation_parts.append(f"It comes from an authoritative source (authority: {dims['social']['authority']:.2f}).")
                if dims.get("strategic", {}).get("innovation", 0) > 0.7:
                    explanation_parts.append(f"It represents a high level of innovation (innovation: {dims['strategic']['innovation']:.2f}).")
                if dims.get("causal", {}).get("risk_factors", 0) > 0.7:
                    explanation_parts.append(f"It carries significant risk (risk: {dims['causal']['risk_factors']:.2f}).")
                if dims.get("causal", {}).get("dependencies", 0) > 0.7:
                    explanation_parts.append(f"It highlights important dependencies (dependencies: {dims['causal']['dependencies']:.2f}).")
            except json.JSONDecodeError:
                logger.warning(f"Could not decode dimensions_json for memory {memory.id} in explanation generation.")

        return " ".join(explanation_parts)

    def get_activation_config(self) -> dict:
        """
        Get current activation configuration.

        Returns:
            Dictionary with activation thresholds
        """
        return {
            "core_threshold": self.core_threshold,
            "peripheral_threshold": self.peripheral_threshold,
            "decay_factor": self.decay_factor
        }

    def update_thresholds(
        self,
        core_threshold: float,
        peripheral_threshold: float
    ) -> None:
        """
        Update activation thresholds.

        Args:
            core_threshold: New core threshold
            peripheral_threshold: New peripheral threshold
        """
        self.core_threshold = max(0.0, min(1.0, core_threshold))
        self.peripheral_threshold = max(0.0, min(1.0, peripheral_threshold))

        logger.debug(
            "Activation thresholds updated",
            core_threshold=self.core_threshold,
            peripheral_threshold=self.peripheral_threshold,
        )

    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            # Ensure arrays have compatible dtypes
            if vec1.dtype != vec2.dtype:
                vec2 = vec2.astype(vec1.dtype)

            # Compute cosine similarity
            dot_product = np.dot(vec1.flatten(), vec2.flatten())
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # Clamp to [0, 1] range and handle numerical issues
            similarity = np.clip(similarity, 0.0, 1.0)

            return float(similarity)

        except Exception as e:
            logger.warning("Cosine similarity computation failed", error=str(e))
            return 0.0
