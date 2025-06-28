"""
Bridge discovery implementation for finding serendipitous connections.

This module implements bridge discovery to find memories that create
unexpected but valuable connections between different concepts.

Reference: IMPLEMENTATION_GUIDE.md - Phase 3: Bridge Discovery
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import numpy as np
from loguru import logger

from ...models.entities import Memory
from ...storage.qdrant.vector_store import QdrantVectorStore


@dataclass
class BridgeMemory:
    """A memory that serves as a bridge between concepts."""
    
    memory: Memory
    novelty_score: float
    connection_potential: float
    surprise_score: float
    bridge_score: float
    explanation: str
    connected_concepts: List[str]


class SimpleBridgeDiscovery:
    """
    Simple bridge discovery implementation using distance inversion
    and novelty detection to find serendipitous connections.
    """

    def __init__(
        self,
        memory_repo: Any,  # MemoryRepository
        vector_store: QdrantVectorStore,
        novelty_weight: float = 0.5,
        connection_weight: float = 0.5,
        min_bridge_score: float = 0.6,
    ):
        """
        Initialize bridge discovery engine.

        Args:
            memory_repo: Repository for memory access
            vector_store: Vector storage for distance calculations
            novelty_weight: Weight for novelty in bridge scoring
            connection_weight: Weight for connection potential
            min_bridge_score: Minimum score to qualify as a bridge
        """
        self.memory_repo = memory_repo
        self.vector_store = vector_store
        self.novelty_weight = novelty_weight
        self.connection_weight = connection_weight
        self.min_bridge_score = min_bridge_score

    async def discover_bridges(
        self,
        query_context: np.ndarray,
        retrieved_memories: List[Memory],
        max_bridges: int = 5,
        search_expansion: int = 50,
    ) -> List[BridgeMemory]:
        """
        Discover bridge memories that create unexpected connections.

        Args:
            query_context: Query context vector
            retrieved_memories: Already retrieved memories
            max_bridges: Maximum number of bridges to return
            search_expansion: Number of candidates to explore

        Returns:
            List of BridgeMemory objects with explanations
        """
        try:
            if not retrieved_memories:
                return []

            # Get concepts from retrieved memories
            concept_vectors = self._extract_concept_vectors(retrieved_memories)
            
            # Find potential bridge candidates
            candidates = await self._find_bridge_candidates(
                query_context,
                concept_vectors,
                retrieved_memories,
                search_expansion
            )

            # Score and rank bridges
            bridges = []
            for candidate in candidates:
                bridge = await self._evaluate_bridge(
                    candidate,
                    query_context,
                    retrieved_memories,
                    concept_vectors
                )
                
                if bridge and bridge.bridge_score >= self.min_bridge_score:
                    bridges.append(bridge)

            # Sort by bridge score and return top bridges
            bridges.sort(key=lambda b: b.bridge_score, reverse=True)
            
            logger.debug(
                f"Bridge discovery found {len(bridges)} bridges from {len(candidates)} candidates"
            )
            
            return bridges[:max_bridges]

        except Exception as e:
            logger.error("Bridge discovery failed", error=str(e))
            return []

    async def _find_bridge_candidates(
        self,
        query_context: np.ndarray,
        concept_vectors: List[np.ndarray],
        retrieved_memories: List[Memory],
        search_expansion: int,
    ) -> List[Memory]:
        """
        Find potential bridge candidates using distance inversion.

        Args:
            query_context: Query context vector
            concept_vectors: Vectors representing main concepts
            retrieved_memories: Already retrieved memories
            search_expansion: Number of candidates to find

        Returns:
            List of potential bridge memories
        """
        # Create set of already retrieved IDs
        retrieved_ids = {m.id for m in retrieved_memories}
        
        # Calculate centroid of concept vectors
        if concept_vectors:
            concept_centroid = np.mean(concept_vectors, axis=0)
        else:
            concept_centroid = query_context

        # Search for memories that are:
        # 1. Moderately distant from query (not too close, not too far)
        # 2. Connected to multiple concepts
        # 3. Not already retrieved
        
        candidates = []
        
        # Search each level for candidates
        for level in [0, 1, 2]:
            collection_name = f"L{level}_cognitive_{'concepts' if level == 0 else 'contexts' if level == 1 else 'episodes'}"
            
            # Search with inverted query (find moderately distant memories)
            inverted_query = self._create_inverted_query(query_context, concept_centroid)
            
            results = await self.vector_store.search(
                query_vector=inverted_query.tolist(),
                collection_name=collection_name,
                limit=search_expansion // 3,  # Distribute across levels
                score_threshold=0.4,  # Moderate distance
            )
            
            for result in results:
                if result.id not in retrieved_ids:
                    memory = await self.memory_repo.get_by_id(result.id)
                    if memory:
                        candidates.append(memory)

        return candidates

    def _create_inverted_query(
        self, 
        query_vector: np.ndarray, 
        concept_centroid: np.ndarray
    ) -> np.ndarray:
        """
        Create an inverted query vector for finding bridges.

        Args:
            query_vector: Original query vector
            concept_centroid: Centroid of concept vectors

        Returns:
            Inverted query vector
        """
        # Combine query and concept centroid
        combined = 0.7 * query_vector + 0.3 * concept_centroid
        
        # Add noise for exploration
        noise = np.random.normal(0, 0.1, combined.shape)
        inverted = combined + noise
        
        # Normalize
        inverted = inverted / (np.linalg.norm(inverted) + 1e-8)
        
        return inverted

    async def _evaluate_bridge(
        self,
        candidate: Memory,
        query_context: np.ndarray,
        retrieved_memories: List[Memory],
        concept_vectors: List[np.ndarray],
    ) -> Optional[BridgeMemory]:
        """
        Evaluate a candidate memory as a potential bridge.

        Args:
            candidate: Candidate bridge memory
            query_context: Query context vector
            retrieved_memories: Already retrieved memories
            concept_vectors: Vectors representing main concepts

        Returns:
            BridgeMemory if qualifies, None otherwise
        """
        try:
            if not hasattr(candidate, 'vector_embedding') or candidate.vector_embedding is None:
                return None

            # Calculate novelty score
            novelty_score = self._calculate_novelty_score(
                candidate.vector_embedding,
                concept_vectors
            )

            # Calculate connection potential
            connection_potential = await self._calculate_connection_potential(
                candidate,
                retrieved_memories
            )

            # Calculate surprise score
            surprise_score = self._calculate_surprise_score(
                candidate,
                query_context,
                retrieved_memories
            )

            # Calculate bridge score
            bridge_score = (
                self.novelty_weight * novelty_score +
                self.connection_weight * connection_potential +
                surprise_score # Add surprise score to bridge score
            )

            # Generate explanation
            explanation = self._generate_bridge_explanation(
                candidate,
                novelty_score,
                connection_potential,
                surprise_score,
                retrieved_memories
            )

            # Find connected concepts
            connected_concepts = self._identify_connected_concepts(
                candidate,
                retrieved_memories
            )

            return BridgeMemory(
                memory=candidate,
                novelty_score=novelty_score,
                connection_potential=connection_potential,
                surprise_score=surprise_score,
                bridge_score=bridge_score,
                explanation=explanation,
                connected_concepts=connected_concepts
            )

        except Exception as e:
            logger.warning(
                f"Failed to evaluate bridge candidate {candidate.id}: {e}"
            )
            return None

    def _calculate_novelty_score(
        self,
        candidate_vector: np.ndarray,
        concept_vectors: List[np.ndarray],
    ) -> float:
        """
        Calculate how novel/different a memory is from existing concepts.

        Args:
            candidate_vector: Vector of candidate bridge
            concept_vectors: Vectors of main concepts

        Returns:
            Novelty score (0.0 to 1.0, higher = more novel)
        """
        if not concept_vectors:
            return 0.5

        # Calculate distances to all concept vectors
        distances = []
        for concept_vec in concept_vectors:
            similarity = self._compute_cosine_similarity(
                candidate_vector, concept_vec
            )
            distance = 1.0 - similarity
            distances.append(distance)

        # Novelty is based on average distance (not too close, not too far)
        avg_distance = np.mean(distances)
        
        # Ideal distance is around 0.5 (moderately different)
        # Too close (0.0) or too far (1.0) are less interesting
        novelty = 1.0 - abs(avg_distance - 0.5) * 2
        
        return float(np.clip(novelty, 0.0, 1.0))

    async def _calculate_connection_potential(
        self,
        candidate: Memory,
        retrieved_memories: List[Memory],
    ) -> float:
        """
        Calculate potential for creating connections between concepts.

        Args:
            candidate: Candidate bridge memory
            retrieved_memories: Already retrieved memories

        Returns:
            Connection potential score (0.0 to 1.0)
        """
        if not retrieved_memories:
            return 0.0

        # Check various connection indicators
        connection_scores = []

        # 1. Shared entities (stakeholders, deliverables)
        shared_entities = self._count_shared_entities(candidate, retrieved_memories)
        entity_score = min(1.0, shared_entities / 3.0)  # Normalize
        connection_scores.append(entity_score)

        # 2. Temporal proximity
        temporal_score = self._calculate_temporal_proximity(
            candidate, retrieved_memories
        )
        connection_scores.append(temporal_score)

        # 3. Content overlap (without being too similar)
        content_score = self._calculate_content_overlap(
            candidate, retrieved_memories
        )
        connection_scores.append(content_score)

        # 4. Cross-meeting/cross-project potential
        cross_score = self._calculate_cross_reference_potential(
            candidate, retrieved_memories
        )
        connection_scores.append(cross_score)

        # Average all connection scores
        return float(np.mean(connection_scores))

    def _count_shared_entities(
        self,
        candidate: Memory,
        retrieved_memories: List[Memory],
    ) -> int:
        """Count shared entities between candidate and retrieved memories."""
        count = 0
        
        # Check stakeholders
        if candidate.speaker:
            for memory in retrieved_memories:
                if memory.speaker == candidate.speaker:
                    count += 1
                    break

        # Check metadata for shared references
        if candidate.metadata:
            candidate_refs = set(candidate.metadata.get("references", []))
            for memory in retrieved_memories:
                if memory.metadata:
                    memory_refs = set(memory.metadata.get("references", []))
                    if candidate_refs & memory_refs:
                        count += 1

        return count

    def _calculate_temporal_proximity(
        self,
        candidate: Memory,
        retrieved_memories: List[Memory],
    ) -> float:
        """Calculate temporal proximity score."""
        if not retrieved_memories:
            return 0.0

        # Find closest temporal distance
        min_distance = float('inf')
        for memory in retrieved_memories:
            distance = abs((candidate.timestamp - memory.timestamp).total_seconds())
            min_distance = min(min_distance, distance)

        # Convert to score (1 day = 0.5 score, exponential decay)
        hours = min_distance / 3600
        score = np.exp(-hours / 24.0)
        
        return float(np.clip(score, 0.0, 1.0))

    def _calculate_content_overlap(
        self,
        candidate: Memory,
        retrieved_memories: List[Memory],
    ) -> float:
        """Calculate content overlap without being too similar."""
        if not retrieved_memories:
            return 0.0

        # Simple word overlap calculation
        candidate_words = set(candidate.content.lower().split())
        
        overlap_scores = []
        for memory in retrieved_memories:
            memory_words = set(memory.content.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(candidate_words & memory_words)
            union = len(candidate_words | memory_words)
            
            if union > 0:
                similarity = intersection / union
                # We want moderate overlap (not too much, not too little)
                score = 1.0 - abs(similarity - 0.3) * 3.33  # Peak at 0.3
                overlap_scores.append(max(0.0, score))

        return float(np.mean(overlap_scores)) if overlap_scores else 0.0

    def _calculate_cross_reference_potential(
        self,
        candidate: Memory,
        retrieved_memories: List[Memory],
    ) -> float:
        """Calculate potential for cross-meeting/project references."""
        score = 0.0
        
        # Check if candidate is from different meeting
        different_meetings = sum(
            1 for m in retrieved_memories 
            if m.meeting_id != candidate.meeting_id
        )
        if different_meetings > 0:
            score += 0.5

        # Check if candidate is from different project
        if candidate.project_id:
            different_projects = sum(
                1 for m in retrieved_memories 
                if m.project_id and m.project_id != candidate.project_id
            )
            if different_projects > 0:
                score += 0.5

        return min(1.0, score)

    def _generate_bridge_explanation(
        self,
        candidate: Memory,
        novelty_score: float,
        connection_potential: float,
        surprise_score: float,
        retrieved_memories: List[Memory],
    ) -> str:
        """Generate human-readable explanation for why this is a bridge."""
        explanations = []

        # Novelty explanation
        if novelty_score > 0.7:
            explanations.append("introduces a new perspective")
        elif novelty_score > 0.5:
            explanations.append("offers a different angle")

        # Connection explanation
        if connection_potential > 0.7:
            explanations.append("strongly connects multiple concepts")
        elif connection_potential > 0.5:
            explanations.append("links related ideas")

        # Add explanations based on enhanced cognitive dimensions
        if candidate.cognitive_dimensions:
            candidate_dims = candidate.cognitive_dimensions.to_dict()
            if candidate_dims.get("authority", 0) > 0.7:
                explanations.append(f"comes from an authoritative source ({candidate_dims["authority"]:.2f})")
            if candidate_dims.get("impact", 0) > 0.7:
                explanations.append(f"highlights a significant impact ({candidate_dims["impact"]:.2f})")
            if candidate_dims.get("innovation_level", 0) > 0.7:
                explanations.append(f"introduces a high level of innovation ({candidate_dims["innovation_level"]:.2f})")
            if candidate_dims.get("risk_factors", 0) > 0.7:
                explanations.append(f"identifies significant risk ({candidate_dims["risk_factors"]:.2f})")
            if candidate_dims.get("dependencies", 0) > 0.7:
                explanations.append(f"reveals important dependencies ({candidate_dims["dependencies"]:.2f})")
            if candidate_dims.get("urgency", 0) > 0.7:
                explanations.append(f"carries high urgency ({candidate_dims["urgency"]:.2f})")

        # Add explanation for cognitive distance/surprise
        if surprise_score > 0.7:
            explanations.append("and offers a truly unexpected perspective.")
        elif surprise_score > 0.5:
            explanations.append("and provides a surprising connection.")

        # Highlight the "Gap" Bridged
        # This is a more complex logic, for now, let's add a general statement
        if novelty_score > 0.6 and connection_potential > 0.6:
            explanations.append("It bridges seemingly disparate concepts.")

        # Suggest Implications/Actions (simplified for now)
        if candidate.content_type.value in ["action", "decision", "risk"]:
            explanations.append(f"Consider the implications of this {candidate.content_type.value}.")
            if candidate.owner:
                explanations.append(f"Follow up with {candidate.owner}.")

        # Incorporate Stakeholder Context
        if candidate.speaker:
            explanations.append(f"Contributed by {candidate.speaker}.")
            if candidate.speaker_role:
                explanations.append(f"({candidate.speaker_role}).")

        if not explanations:
            explanations.append("creates an unexpected connection")

        return "This memory " + " and ".join(explanations)

    def _identify_connected_concepts(
        self,
        candidate: Memory,
        retrieved_memories: List[Memory],
    ) -> List[str]:
        """Identify which concepts this bridge connects."""
        concepts = []
        
        # Extract key terms from retrieved memories
        for memory in retrieved_memories[:3]:  # Top 3 most relevant
            # Simple keyword extraction (can be improved)
            words = memory.content.lower().split()
            # Filter out common words
            keywords = [w for w in words if len(w) > 4][:2]
            concepts.extend(keywords)

        # Remove duplicates and limit
        unique_concepts = list(set(concepts))[:5]
        
        return unique_concepts

    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
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

            # Clamp to [0, 1] range
            return float(np.clip(similarity, 0.0, 1.0))

        except Exception as e:
            logger.warning("Cosine similarity computation failed", error=str(e))
            return 0.0

    def _extract_concept_vectors(
        self, 
        memories: List[Memory]
    ) -> List[np.ndarray]:
        """Extract concept vectors from memories."""
        vectors = []
        
        for memory in memories[:10]:  # Limit to top 10
            if hasattr(memory, 'vector_embedding') and memory.vector_embedding is not None:
                vectors.append(memory.vector_embedding)
        
        return vectors

    def get_discovery_config(self) -> Dict[str, Any]:
        """Get bridge discovery configuration."""
        return {
            "novelty_weight": self.novelty_weight,
            "connection_weight": self.connection_weight,
            "min_bridge_score": self.min_bridge_score,
            "algorithm": "simple_distance_inversion",
        }

    def _calculate_surprise_score(
        self,
        candidate: Memory,
        query_context: np.ndarray,
        retrieved_memories: List[Memory],
    ) -> float:
        """
        Calculate how surprising a memory is given the context and retrieved memories.
        A memory is surprising if it's not too similar to already retrieved memories
        but still somewhat relevant to the query context.
        """
        if not hasattr(candidate, 'vector_embedding') or candidate.vector_embedding is None:
            return 0.0

        candidate_vector = candidate.vector_embedding.full_vector # Use full 400D vector

        # Similarity to query context (higher is better)
        query_similarity = self._compute_cosine_similarity(candidate_vector, query_context)

        # Average similarity to retrieved memories (lower is better for surprise)
        if not retrieved_memories:
            avg_retrieved_similarity = 0.0
        else:
            retrieved_vectors = [m.vector_embedding.full_vector for m in retrieved_memories if m.vector_embedding is not None]
            if not retrieved_vectors:
                avg_retrieved_similarity = 0.0
            else:
                similarities_to_retrieved = [
                    self._compute_cosine_similarity(candidate_vector, rv)
                    for rv in retrieved_vectors
                ]
                avg_retrieved_similarity = np.mean(similarities_to_retrieved)

        # Cognitive surprise boost
        cognitive_surprise_boost = 0.0
        if candidate.cognitive_dimensions and query_context is not None and len(query_context) == 400:
            candidate_dims = candidate.cognitive_dimensions.to_dict()
            # Assuming query_context is 400D, extract its 16D cognitive part
            query_cognitive_dims_array = query_context[384:]
            query_cognitive_dims = CognitiveDimensions.from_array(query_cognitive_dims_array).to_dict()

            # Example: If candidate has high social dimension but query is low social
            if candidate_dims.get("authority", 0) > 0.7 and query_cognitive_dims.get("authority", 0) < 0.3:
                cognitive_surprise_boost += 0.15
            if candidate_dims.get("innovation_level", 0) > 0.7 and query_cognitive_dims.get("innovation_level", 0) < 0.3:
                cognitive_surprise_boost += 0.15
            if candidate_dims.get("risk_factors", 0) > 0.7 and query_cognitive_dims.get("risk_factors", 0) < 0.3:
                cognitive_surprise_boost += 0.15
            # Add more rules based on specific dimension combinations that indicate surprise

        # Surprise score: High query similarity, low similarity to retrieved memories + cognitive boost
        surprise = query_similarity * (1 - avg_retrieved_similarity) + cognitive_surprise_boost

        return float(np.clip(surprise, 0.0, 1.0))
