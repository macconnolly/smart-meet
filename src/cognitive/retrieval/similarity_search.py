"""
Similarity-based memory search implementation.

This module implements cosine similarity-based retrieval across all
hierarchy levels with recency bias and configurable ranking strategies.

Reference: IMPLEMENTATION_GUIDE.md - Phase 2: Similarity Search
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import numpy as np
from loguru import logger

from ...models.entities import Memory
from ...storage.qdrant.vector_store import QdrantVectorStore


@dataclass
class SearchResult:
    """Result from similarity search."""
    memory: Memory
    similarity_score: float
    distance: float
    metadata: Dict[str, Any]
    
    # Additional computed properties
    combined_score: Optional[float] = None
    recency_score: Optional[float] = None


class SimilaritySearch:
    """
    Similarity-based memory search using cosine similarity.

    Implements k-nearest neighbor search across hierarchy levels (L0, L1, L2)
    with recency bias for recent memory preference and configurable result
    ranking and filtering.
    """

    def __init__(
        self,
        memory_repo: Any,  # MemoryRepository
        vector_store: QdrantVectorStore,
        recency_weight: float = 0.2,
        similarity_weight: float = 0.8,
        recency_decay_hours: float = 168.0,  # 1 week
        modification_date_weight: float = 0.3,
        modification_recency_decay_days: float = 30.0,
        similarity_closeness_threshold: float = 0.1,
    ):
        """
        Initialize similarity search.

        Args:
            memory_repo: Repository for memory metadata access
            vector_store: Vector storage for similarity computation
            recency_weight: Weight for recency bias (0.0 to 1.0)
            similarity_weight: Weight for similarity score (0.0 to 1.0)
            recency_decay_hours: Hours for exponential recency decay
            modification_date_weight: Weight for modification date ranking
            modification_recency_decay_days: Days for modification recency decay
            similarity_closeness_threshold: Threshold for grouping similar scores
        """
        self.memory_repo = memory_repo
        self.vector_store = vector_store
        self.recency_decay_hours = recency_decay_hours
        self.modification_date_weight = modification_date_weight
        self.modification_recency_decay_days = modification_recency_decay_days
        self.similarity_closeness_threshold = similarity_closeness_threshold

        # Validate and normalize weights
        total_weight = recency_weight + similarity_weight
        if total_weight > 0:
            if abs(total_weight - 1.0) > 0.001:
                logger.debug(
                    "Normalizing similarity search weights to sum to 1.0",
                    original_recency=recency_weight,
                    original_similarity=similarity_weight,
                    total=total_weight,
                )
                self.recency_weight = recency_weight / total_weight
                self.similarity_weight = similarity_weight / total_weight
            else:
                self.recency_weight = recency_weight
                self.similarity_weight = similarity_weight
        else:
            logger.warning("Invalid zero weights, using defaults")
            self.recency_weight = 0.2
            self.similarity_weight = 0.8

    async def search_memories(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        levels: Optional[List[int]] = None,
        min_similarity: float = 0.1,
        include_recency_bias: bool = True,
        project_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar memories across specified hierarchy levels.

        Args:
            query_vector: Query vector for similarity computation (400D)
            k: Number of top results to return
            levels: Hierarchy levels to search (None = all levels)
            min_similarity: Minimum similarity threshold
            include_recency_bias: Whether to apply recency bias
            project_id: Optional project filter
            filters: Additional metadata filters

        Returns:
            List of SearchResult objects ranked by combined score
        """
        start_time = time.time()

        try:
            if levels is None:
                levels = [0, 1, 2]  # Search all hierarchy levels

            all_results = []

            # Build combined filters
            combined_filters = filters or {}
            if project_id:
                combined_filters["project_id"] = project_id

            # Search each hierarchy level
            for level in levels:
                level_results = await self._search_level(
                    query_vector, 
                    level, 
                    k, 
                    min_similarity, 
                    include_recency_bias,
                    combined_filters
                )
                all_results.extend(level_results)

            # Apply date-based secondary ranking if enabled
            all_results = self._apply_date_based_ranking(all_results)

            # Sort by combined score and return top-k
            all_results.sort(
                key=lambda r: r.combined_score or r.similarity_score,
                reverse=True,
            )
            top_results = all_results[:k]

            search_time_ms = (time.time() - start_time) * 1000

            logger.debug(
                "Similarity search completed",
                levels_searched=levels,
                total_candidates=len(all_results),
                returned_results=len(top_results),
                search_time_ms=search_time_ms,
            )

            return top_results

        except Exception as e:
            logger.error("Similarity search failed", error=str(e))
            return []

    async def search_by_level(
        self,
        query_vector: np.ndarray,
        level: int,
        k: int = 10,
        min_similarity: float = 0.1,
        include_recency_bias: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search memories at a specific hierarchy level.

        Args:
            query_vector: Query vector for similarity computation
            level: Hierarchy level to search (0, 1, or 2)
            k: Number of top results to return
            min_similarity: Minimum similarity threshold
            include_recency_bias: Whether to apply recency bias
            filters: Additional metadata filters

        Returns:
            List of SearchResult objects from the specified level
        """
        try:
            results = await self._search_level(
                query_vector, 
                level,
                k,
                min_similarity, 
                include_recency_bias,
                filters
            )

            # Sort and return top-k
            results.sort(
                key=lambda r: r.combined_score or r.similarity_score,
                reverse=True,
            )
            return results[:k]

        except Exception as e:
            logger.error("Level-specific search failed", level=level, error=str(e))
            return []

    async def find_most_similar(
        self,
        query_vector: np.ndarray,
        candidate_memories: List[Memory],
        include_recency_bias: bool = True,
    ) -> Optional[SearchResult]:
        """
        Find the most similar memory from a list of candidates.

        Args:
            query_vector: Query vector for similarity computation
            candidate_memories: List of candidate memories
            include_recency_bias: Whether to apply recency bias

        Returns:
            SearchResult with the most similar memory, or None if no candidates
        """
        if not candidate_memories:
            return None

        results = []
        for memory in candidate_memories:
            if hasattr(memory, 'vector_embedding') and memory.vector_embedding is not None:
                similarity = self._compute_cosine_similarity(
                    query_vector, memory.vector_embedding
                )
                
                if similarity >= 0.0:  # Accept all similarities
                    # Calculate combined score with optional recency bias
                    if include_recency_bias:
                        recency_score = self._calculate_recency_score(memory)
                        combined_score = self._calculate_combined_score(
                            similarity, recency_score
                        )
                    else:
                        combined_score = similarity
                        recency_score = 0.0

                    result = SearchResult(
                        memory=memory,
                        similarity_score=similarity,
                        distance=1.0 - similarity,
                        metadata={
                            "hierarchy_level": memory.level,
                        },
                        combined_score=combined_score,
                        recency_score=recency_score
                    )
                    results.append(result)

        if results:
            return max(results, key=lambda r: r.combined_score or r.similarity_score)

        return None

    async def _search_level(
        self,
        query_vector: np.ndarray,
        level: int,
        k: int,
        min_similarity: float,
        include_recency_bias: bool,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search memories at a specific level with similarity computation.

        Args:
            query_vector: Query vector for similarity computation
            level: Hierarchy level to search
            k: Number of results to retrieve
            min_similarity: Minimum similarity threshold
            include_recency_bias: Whether to apply recency bias
            filters: Additional metadata filters

        Returns:
            List of SearchResult objects above minimum similarity
        """
        # Search in vector store
        vector_results = await self.vector_store.search(
            query_vector=query_vector.tolist(),
            collection_name=f"L{level}_cognitive_{'concepts' if level == 0 else 'contexts' if level == 1 else 'episodes'}",
            limit=k * 2,  # Get more to filter by similarity
            score_threshold=min_similarity,
            filter_conditions=filters
        )

        results = []
        for vr in vector_results:
            # Get full memory from repository
            memory = await self.memory_repo.get_by_id(vr.id)
            if memory:
                similarity = vr.score

                # Calculate combined score with optional recency bias
                if include_recency_bias:
                    recency_score = self._calculate_recency_score(memory)
                    combined_score = self._calculate_combined_score(
                        similarity, recency_score
                    )
                else:
                    combined_score = similarity
                    recency_score = 0.0

                # Create search result
                result = SearchResult(
                    memory=memory,
                    similarity_score=similarity,
                    distance=1.0 - similarity,
                    metadata={
                        "pure_similarity": similarity,
                        "recency_score": recency_score,
                        "combined_score": combined_score,
                        "hierarchy_level": memory.level,
                    },
                    combined_score=combined_score,
                    recency_score=recency_score
                )

                results.append(result)

        return results

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

    def _calculate_recency_score(self, memory: Memory) -> float:
        """
        Calculate recency score with exponential decay.

        Args:
            memory: Memory to calculate recency score for

        Returns:
            Recency score (0.0 to 1.0, higher = more recent)
        """
        try:
            # Use last_accessed if available, otherwise use timestamp
            reference_time = memory.last_accessed or memory.timestamp

            # Calculate time difference
            time_diff = datetime.now() - reference_time
            hours_elapsed = time_diff.total_seconds() / 3600

            # Exponential decay: score = exp(-hours_elapsed / decay_constant)
            decay_constant = self.recency_decay_hours
            recency_score = np.exp(-hours_elapsed / decay_constant)

            # Clamp to [0, 1] range
            return float(np.clip(recency_score, 0.0, 1.0))

        except Exception as e:
            logger.warning(
                "Recency score calculation failed", 
                memory_id=memory.id, 
                error=str(e)
            )
            return 0.5  # Default neutral recency score

    def _calculate_combined_score(self, similarity: float, recency: float) -> float:
        """
        Calculate combined score from similarity and recency scores.

        Args:
            similarity: Similarity score (0.0 to 1.0)
            recency: Recency score (0.0 to 1.0)

        Returns:
            Combined weighted score (0.0 to 1.0)
        """
        return self.similarity_weight * similarity + self.recency_weight * recency

    def get_search_config(self) -> Dict[str, Any]:
        """
        Get current search configuration.

        Returns:
            Dictionary with search parameters
        """
        return {
            "recency_weight": self.recency_weight,
            "similarity_weight": self.similarity_weight,
            "recency_decay_hours": self.recency_decay_hours,
            "modification_date_weight": self.modification_date_weight,
            "modification_recency_decay_days": self.modification_recency_decay_days,
            "similarity_closeness_threshold": self.similarity_closeness_threshold,
            "algorithm": "cosine_similarity_with_recency_bias",
        }

    def update_weights(self, recency_weight: float, similarity_weight: float) -> None:
        """
        Update search weights with validation.

        Args:
            recency_weight: New recency weight (0.0 to 1.0)
            similarity_weight: New similarity weight (0.0 to 1.0)
        """
        # Validate and normalize weights
        total_weight = recency_weight + similarity_weight
        if total_weight > 0:
            self.recency_weight = recency_weight / total_weight
            self.similarity_weight = similarity_weight / total_weight
        else:
            logger.warning("Invalid weights provided, keeping current configuration")
            return

        logger.debug(
            "Search weights updated",
            recency_weight=self.recency_weight,
            similarity_weight=self.similarity_weight,
        )

    def set_recency_decay(self, decay_hours: float) -> None:
        """
        Update recency decay parameter.

        Args:
            decay_hours: New decay time in hours
        """
        self.recency_decay_hours = max(1.0, decay_hours)  # Minimum 1 hour

        logger.debug(
            "Recency decay updated",
            decay_hours=self.recency_decay_hours,
        )

    def _apply_date_based_ranking(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Apply date-based secondary ranking to closely-scored memories.

        Groups results by similarity score clusters and applies modification
        date recency as a secondary ranking factor for close scores.

        Args:
            results: List of search results to re-rank

        Returns:
            Re-ranked list of search results
        """
        if not results:
            return results

        # Group results into similarity clusters
        clusters = self._group_by_similarity_threshold(
            results, self.similarity_closeness_threshold
        )

        final_results = []
        for cluster in clusters:
            if len(cluster) > 1:
                # Apply secondary ranking by modification date for clusters with multiple results
                for result in cluster:
                    mod_recency = self._calculate_modification_recency_score(
                        result.memory
                    )

                    # Blend existing combined score with modification recency
                    original_score = result.combined_score or result.similarity_score
                    result.combined_score = self._blend_with_modification_score(
                        original_score, mod_recency, self.modification_date_weight
                    )

                # Re-sort cluster by new combined score
                cluster.sort(key=lambda r: r.combined_score or r.similarity_score, reverse=True)

            final_results.extend(cluster)

        return final_results

    def _group_by_similarity_threshold(
        self, results: List[SearchResult], threshold: float
    ) -> List[List[SearchResult]]:
        """
        Group results into clusters based on similarity score closeness.

        Args:
            results: List of search results to group
            threshold: Similarity threshold for grouping

        Returns:
            List of result clusters
        """
        if not results:
            return []

        # Sort by similarity score first
        sorted_results = sorted(results, key=lambda r: r.similarity_score, reverse=True)

        clusters = []
        current_cluster = [sorted_results[0]]

        for i in range(1, len(sorted_results)):
            current_result = sorted_results[i]
            last_in_cluster = current_cluster[-1]

            # Check if within threshold of the cluster
            score_diff = abs(
                last_in_cluster.similarity_score - current_result.similarity_score
            )

            if score_diff <= threshold:
                current_cluster.append(current_result)
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [current_result]

        # Add the last cluster
        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def _calculate_modification_recency_score(self, memory: Memory) -> float:
        """
        Calculate modification recency score based on when content was last modified.

        Args:
            memory: Memory to calculate modification recency for

        Returns:
            Modification recency score (0.0 to 1.0, higher = more recent)
        """
        try:
            # Get the modification date from the memory
            modification_date = self._get_memory_modification_date(memory)

            if modification_date is None:
                return 0.0  # No modification date available

            # Calculate days since modification
            time_diff = datetime.now() - modification_date
            days_elapsed = time_diff.total_seconds() / 86400

            # Exponential decay based on configured decay period
            decay_days = self.modification_recency_decay_days
            recency_score = np.exp(-days_elapsed / decay_days)

            # Clamp to [0, 1] range
            return float(np.clip(recency_score, 0.0, 1.0))

        except Exception as e:
            logger.warning(
                "Modification recency score calculation failed",
                memory_id=memory.id,
                error=str(e),
            )
            return 0.0  # Default neutral score

    def _get_memory_modification_date(self, memory: Memory) -> Optional[datetime]:
        """
        Extract modification date from memory metadata.

        Args:
            memory: Memory to extract modification date from

        Returns:
            Modification datetime or None if not available
        """
        # Check metadata for modification date
        if memory.metadata:
            # Try various metadata fields
            for field in ["modified_date", "updated_at", "last_modified"]:
                if field in memory.metadata:
                    try:
                        return datetime.fromisoformat(memory.metadata[field])
                    except (ValueError, TypeError):
                        pass

        # Fallback to memory timestamp
        return memory.timestamp

    def _blend_with_modification_score(
        self,
        original_score: float,
        modification_score: float,
        modification_weight: float,
    ) -> float:
        """
        Blend original similarity score with modification recency score.

        Args:
            original_score: Original combined score (similarity + access recency)
            modification_score: Modification recency score
            modification_weight: Weight for modification score blending

        Returns:
            Blended score incorporating modification recency
        """
        # Use weighted average approach to prevent ceiling effects
        mod_weight = min(modification_weight, 0.5)  # Cap at 50% influence

        # Calculate weighted average
        blended_score = (
            1.0 - mod_weight
        ) * original_score + mod_weight * modification_score

        # For very close similarity scores, add a small boost for modification recency
        if modification_score > 0.5:  # Only boost if modification is reasonably recent
            boost = mod_weight * 0.1 * modification_score  # Small boost
            blended_score += boost

        # Ensure result stays within reasonable bounds
        return min(1.0, blended_score)
