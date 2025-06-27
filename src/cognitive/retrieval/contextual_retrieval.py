"""
Contextual retrieval coordinator implementation.

This module provides the high-level coordination of activation, similarity search,
and bridge discovery to create a unified retrieval system that categorizes
memories and aggregates results.

Reference: IMPLEMENTATION_GUIDE.md - Phase 2: Integrated Retrieval
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from loguru import logger

from ...models.entities import Memory
from .basic_activation import BasicActivationEngine, ActivationResult
from .bridge_discovery import SimpleBridgeDiscovery, BridgeMemory
from .similarity_search import SimilaritySearch, SearchResult


@dataclass
class ContextualRetrievalResult:
    """
    Result from contextual retrieval containing categorized memories.

    Organizes retrieval results into core memories (highly relevant),
    peripheral memories (moderately relevant), and bridge memories
    (serendipitous connections).
    """
    
    core_memories: List[Memory]
    peripheral_memories: List[Memory]
    bridge_memories: List[BridgeMemory]
    activation_result: Optional[ActivationResult] = None
    similarity_results: Optional[List[SearchResult]] = None
    retrieval_time_ms: float = 0.0
    context_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize computed properties."""
        self.context_metadata = self.context_metadata or {}
        self.total_memories = len(self.core_memories) + len(self.peripheral_memories)
        self.total_bridges = len(self.bridge_memories)

    def get_all_memories(self) -> List[Memory]:
        """Get all memories (core + peripheral)."""
        return self.core_memories + self.peripheral_memories

    def get_memories_by_level(self, level: int) -> List[Memory]:
        """Get all memories at a specific hierarchy level."""
        all_memories = self.get_all_memories()
        return [m for m in all_memories if m.level == level]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "core_memories": [m.to_dict() for m in self.core_memories],
            "peripheral_memories": [m.to_dict() for m in self.peripheral_memories],
            "bridge_memories": [
                {
                    "memory": b.memory.to_dict(),
                    "novelty_score": b.novelty_score,
                    "connection_potential": b.connection_potential,
                    "bridge_score": b.bridge_score,
                    "explanation": b.explanation,
                }
                for b in self.bridge_memories
            ],
            "total_memories": self.total_memories,
            "total_bridges": self.total_bridges,
            "retrieval_time_ms": self.retrieval_time_ms,
            "context_metadata": self.context_metadata,
        }


class ContextualRetrieval:
    """
    High-level retrieval coordinator that integrates activation, similarity search,
    and bridge discovery into a unified contextual retrieval system.

    Provides the main interface for memory retrieval with automatic result
    categorization and ranking.
    """

    def __init__(
        self,
        memory_repo: Any,  # MemoryRepository
        connection_repo: Any,  # MemoryConnectionRepository
        vector_store: Any,  # QdrantVectorStore
        activation_engine: Optional[BasicActivationEngine] = None,
        similarity_search: Optional[SimilaritySearch] = None,
        bridge_discovery: Optional[SimpleBridgeDiscovery] = None,
    ):
        """
        Initialize contextual retrieval coordinator.

        Args:
            memory_repo: Repository for memory access
            connection_repo: Repository for connection access
            vector_store: Vector storage for similarity search
            activation_engine: Optional activation engine (created if None)
            similarity_search: Optional similarity search (created if None)
            bridge_discovery: Optional bridge discovery (created if None)
        """
        self.memory_repo = memory_repo
        self.connection_repo = connection_repo
        self.vector_store = vector_store

        # Initialize retrieval components
        self.activation_engine = activation_engine or BasicActivationEngine(
            memory_repo, connection_repo, vector_store
        )
        self.similarity_search = similarity_search or SimilaritySearch(
            memory_repo, vector_store
        )
        self.bridge_discovery = bridge_discovery or SimpleBridgeDiscovery(
            memory_repo, vector_store
        )

    async def retrieve_memories(
        self,
        query_context: np.ndarray,
        max_core: int = 10,
        max_peripheral: int = 15,
        max_bridges: int = 5,
        activation_threshold: float = 0.6,
        similarity_threshold: float = 0.3,
        use_activation: bool = True,
        use_similarity: bool = True,
        use_bridges: bool = True,
        project_id: Optional[str] = None,
        stakeholder_filter: Optional[List[str]] = None,
        meeting_type_filter: Optional[List[str]] = None,
    ) -> ContextualRetrievalResult:
        """
        Retrieve memories using integrated activation, similarity, and bridge discovery.

        Args:
            query_context: Query context vector (400D)
            max_core: Maximum core memories to return
            max_peripheral: Maximum peripheral memories to return
            max_bridges: Maximum bridge memories to return
            activation_threshold: Threshold for activation spreading
            similarity_threshold: Threshold for similarity search
            use_activation: Whether to use activation spreading
            use_similarity: Whether to use similarity search
            use_bridges: Whether to use bridge discovery
            project_id: Optional project filter
            stakeholder_filter: Optional stakeholder filter
            meeting_type_filter: Optional meeting type filter

        Returns:
            ContextualRetrievalResult with categorized memories
        """
        start_time = time.time()

        try:
            # Phase 1: Activation spreading (if enabled)
            activation_result = None
            activated_memories = []

            if use_activation:
                activation_result = await self.activation_engine.activate_memories(
                    query_context, 
                    activation_threshold, 
                    max_core + max_peripheral,
                    project_id
                )
                activated_memories = (
                    activation_result.core_memories + 
                    activation_result.peripheral_memories
                )

                logger.debug(
                    "Activation completed",
                    core_count=len(activation_result.core_memories),
                    peripheral_count=len(activation_result.peripheral_memories),
                )

            # Phase 2: Similarity search (if enabled)
            similarity_results = []
            similarity_memories = []

            if use_similarity:
                similarity_results = await self.similarity_search.search_memories(
                    query_context,
                    k=max_core + max_peripheral,
                    min_similarity=similarity_threshold,
                    project_id=project_id,
                    filters=self._build_filters(stakeholder_filter, meeting_type_filter)
                )
                similarity_memories = [r.memory for r in similarity_results]

                logger.debug(
                    f"Similarity search found {len(similarity_results)} memories"
                )

            # Phase 3: Merge and categorize memories
            core_memories, peripheral_memories = self._merge_and_categorize_memories(
                activated_memories,
                similarity_memories,
                query_context,
                max_core,
                max_peripheral,
            )

            # Phase 4: Bridge discovery (if enabled)
            bridge_memories = []

            if use_bridges:
                all_retrieved = core_memories + peripheral_memories
                if all_retrieved:  # Only search for bridges if we have retrieved memories
                    bridge_memories = await self.bridge_discovery.discover_bridges(
                        query_context, 
                        all_retrieved, 
                        max_bridges
                    )

                    logger.debug(
                        f"Bridge discovery found {len(bridge_memories)} bridges"
                    )

            # Create result
            retrieval_time_ms = (time.time() - start_time) * 1000

            result = ContextualRetrievalResult(
                core_memories=core_memories,
                peripheral_memories=peripheral_memories,
                bridge_memories=bridge_memories,
                activation_result=activation_result,
                similarity_results=similarity_results,
                retrieval_time_ms=retrieval_time_ms,
                context_metadata={
                    "activation_threshold": activation_threshold,
                    "similarity_threshold": similarity_threshold,
                    "used_activation": use_activation,
                    "used_similarity": use_similarity,
                    "used_bridges": use_bridges,
                    "project_id": project_id,
                    "stakeholder_filter": stakeholder_filter,
                    "meeting_type_filter": meeting_type_filter,
                },
            )

            logger.info(
                "Contextual retrieval completed",
                core_memories=len(core_memories),
                peripheral_memories=len(peripheral_memories),
                bridge_memories=len(bridge_memories),
                retrieval_time_ms=retrieval_time_ms,
            )

            return result

        except Exception as e:
            logger.error("Contextual retrieval failed", error=str(e))
            return ContextualRetrievalResult(
                [], [], [], retrieval_time_ms=(time.time() - start_time) * 1000
            )

    def _merge_and_categorize_memories(
        self,
        activated_memories: List[Memory],
        similarity_memories: List[Memory],
        query_context: np.ndarray,
        max_core: int,
        max_peripheral: int,
    ) -> Tuple[List[Memory], List[Memory]]:
        """
        Merge memories from activation and similarity search, then categorize.

        Args:
            activated_memories: Memories from activation spreading
            similarity_memories: Memories from similarity search
            query_context: Original query context for scoring
            max_core: Maximum core memories
            max_peripheral: Maximum peripheral memories

        Returns:
            Tuple of (core_memories, peripheral_memories)
        """
        # Combine and deduplicate memories
        memory_map = {}
        memory_scores = {}

        # Add activated memories with their activation strengths
        for memory in activated_memories:
            if memory.id not in memory_map:
                memory_map[memory.id] = memory
                # Calculate score based on vector similarity
                if hasattr(memory, 'vector_embedding') and memory.vector_embedding is not None:
                    similarity = self._compute_cosine_similarity(
                        query_context, memory.vector_embedding
                    )
                    memory_scores[memory.id] = similarity
                else:
                    memory_scores[memory.id] = 0.5  # Default score

        # Add similarity memories, updating scores if already present
        for memory in similarity_memories:
            if memory.id in memory_map:
                # Take the higher score
                if hasattr(memory, 'vector_embedding') and memory.vector_embedding is not None:
                    similarity = self._compute_cosine_similarity(
                        query_context, memory.vector_embedding
                    )
                    memory_scores[memory.id] = max(memory_scores[memory.id], similarity)
            else:
                memory_map[memory.id] = memory
                if hasattr(memory, 'vector_embedding') and memory.vector_embedding is not None:
                    similarity = self._compute_cosine_similarity(
                        query_context, memory.vector_embedding
                    )
                    memory_scores[memory.id] = similarity
                else:
                    memory_scores[memory.id] = 0.5

        # Sort memories by score
        sorted_memories = sorted(
            memory_map.values(),
            key=lambda m: memory_scores.get(m.id, 0.0),
            reverse=True,
        )

        # Categorize as core or peripheral based on score thresholds
        core_memories: List[Memory] = []
        peripheral_memories: List[Memory] = []

        for memory in sorted_memories:
            score = memory_scores.get(memory.id, 0.0)

            if score >= 0.7 and len(core_memories) < max_core:
                core_memories.append(memory)
            elif score >= 0.4 and len(peripheral_memories) < max_peripheral:
                peripheral_memories.append(memory)
            elif len(core_memories) < max_core:
                # Fill core if we have space and no more high-scoring memories
                core_memories.append(memory)
            elif len(peripheral_memories) < max_peripheral:
                # Fill peripheral if we have space
                peripheral_memories.append(memory)
            else:
                # Both categories are full
                break

        return core_memories, peripheral_memories

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

            # Flatten vectors for dot product
            vec1_flat = vec1.flatten()
            vec2_flat = vec2.flatten()

            # Compute cosine similarity
            dot_product = np.dot(vec1_flat, vec2_flat)
            norm1 = np.linalg.norm(vec1_flat)
            norm2 = np.linalg.norm(vec2_flat)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # Clamp to [0, 1] range and handle numerical issues
            similarity = np.clip(similarity, 0.0, 1.0)

            return float(similarity)

        except Exception as e:
            logger.warning("Cosine similarity computation failed", error=str(e))
            return 0.0

    def _build_filters(
        self,
        stakeholder_filter: Optional[List[str]],
        meeting_type_filter: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        """Build filters for similarity search."""
        if not stakeholder_filter and not meeting_type_filter:
            return None
            
        filters = {}
        if stakeholder_filter:
            filters["stakeholders"] = stakeholder_filter
        if meeting_type_filter:
            filters["meeting_type"] = meeting_type_filter
            
        return filters

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get retrieval system statistics and configuration.

        Returns:
            Dictionary with system stats and configuration
        """
        stats: Dict[str, Any] = {
            "has_activation_engine": self.activation_engine is not None,
            "has_similarity_search": self.similarity_search is not None,
            "has_bridge_discovery": self.bridge_discovery is not None,
        }

        # Add component configurations
        if self.activation_engine:
            stats["activation_config"] = self.activation_engine.get_activation_config()

        if self.similarity_search:
            stats["similarity_config"] = self.similarity_search.get_search_config()

        if self.bridge_discovery:
            stats["bridge_config"] = {
                "algorithm": "simple_bridge_discovery",
                "novelty_weight": 0.5,
                "connection_weight": 0.5
            }

        return stats
