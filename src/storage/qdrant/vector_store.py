"""
Qdrant vector store interface for the 3-tier cognitive memory system.

This module provides the interface for storing and retrieving vectors
from Qdrant, managing the three-tier hierarchy of memories.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        PointStruct,
        PointIdsList,
        SearchRequest,
        SearchParams,
        Filter,
        FieldCondition,
        MatchValue,
        Range,
        DatetimeRange,
        MatchAny,
        HasIdCondition,
        UpdateStatus,
        ScrollRequest,
        ScrollResult,
    )
except ImportError:
    raise ImportError("Qdrant client not installed. Please install: pip install qdrant-client")

from ...models.entities import Memory
from ...extraction.dimensions.dimension_analyzer import CognitiveDimensions

logger = logging.getLogger(__name__)


@dataclass
class SearchFilter:
    """Filters for vector search."""

    project_id: Optional[str] = None
    meeting_id: Optional[str] = None
    memory_type: Optional[str] = None
    content_type: Optional[str] = None
    speaker: Optional[str] = None
    min_importance: Optional[float] = None
    max_importance: Optional[float] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None

    def to_qdrant_filter(self) -> Optional[Filter]:
        """Convert to Qdrant filter format."""
        conditions = []

        if self.project_id:
            conditions.append(
                FieldCondition(key="project_id", match=MatchValue(value=self.project_id))
            )

        if self.meeting_id:
            conditions.append(
                FieldCondition(key="meeting_id", match=MatchValue(value=self.meeting_id))
            )

        if self.memory_type:
            conditions.append(
                FieldCondition(key="memory_type", match=MatchValue(value=self.memory_type))
            )

        if self.content_type:
            conditions.append(
                FieldCondition(key="content_type", match=MatchValue(value=self.content_type))
            )

        if self.speaker:
            conditions.append(FieldCondition(key="speaker", match=MatchValue(value=self.speaker)))

        if self.min_importance is not None or self.max_importance is not None:
            conditions.append(
                FieldCondition(
                    key="importance_score",
                    range=Range(gte=self.min_importance, lte=self.max_importance),
                )
            )

        if self.created_after or self.created_before:
            # Convert to timestamps
            gte = int(self.created_after.timestamp()) if self.created_after else None
            lte = int(self.created_before.timestamp()) if self.created_before else None

            conditions.append(FieldCondition(key="created_at", range=Range(gte=gte, lte=lte)))

        if conditions:
            return Filter(must=conditions)

        return None


@dataclass
class VectorSearchResult:
    """Result from vector search."""

    memory_id: str
    score: float
    vector: Vector
    payload: Dict[str, Any]


class QdrantVectorStore:
    """
    Interface for storing and retrieving vectors from Qdrant.

    Manages the 3-tier hierarchy:
    - L0: Cognitive concepts (semantic memories)
    - L1: Cognitive contexts (consolidated patterns)
    - L2: Cognitive episodes (raw memories)
    """

    # Collection names
    L0_COLLECTION = "L0_cognitive_concepts"
    L1_COLLECTION = "L1_cognitive_contexts"
    L2_COLLECTION = "L2_cognitive_episodes"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        timeout: int = 30,
    ):
        """
        Initialize Qdrant client.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            api_key: Optional API key
            prefer_grpc: Whether to use gRPC (faster)
            timeout: Operation timeout in seconds
        """
        self.client = QdrantClient(
            host=host, port=port, api_key=api_key, prefer_grpc=prefer_grpc, timeout=timeout
        )

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Cache for collection info
        self._collection_info_cache = {}

        # Verify collections exist
        self._verify_collections()

    def _verify_collections(self) -> None:
        """Verify required collections exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            required = [self.L0_COLLECTION, self.L1_COLLECTION, self.L2_COLLECTION]
            missing = [name for name in required if name not in collection_names]

            if missing:
                raise RuntimeError(
                    f"Missing required collections: {missing}. "
                    "Run 'python scripts/init_qdrant.py' to initialize."
                )

            logger.info("All required Qdrant collections verified")

        except Exception as e:
            logger.error(f"Failed to verify Qdrant collections: {e}")
            raise

    def _get_collection_for_level(self, level: int) -> str:
        """Get collection name for memory level."""
        if level == 0:
            return self.L0_COLLECTION
        elif level == 1:
            return self.L1_COLLECTION
        elif level == 2:
            return self.L2_COLLECTION
        else:
            raise ValueError(f"Invalid memory level: {level}")

    async def store_memory(self, memory: Memory, vector: Vector) -> str:
        """
        Store a memory with its vector.

        Args:
            memory: Memory to store
            vector: Vector representation

        Returns:
            Qdrant point ID
        """
        # Generate Qdrant ID if not present
        if not memory.qdrant_id:
            memory.qdrant_id = str(uuid.uuid4())

        # Get collection based on memory level
        collection_name = self._get_collection_for_level(memory.level)

        # Build payload
        payload = self._build_payload(memory)

        # Create point
        point = PointStruct(
            id=memory.qdrant_id, vector=vector.full_vector.tolist(), payload=payload
        )

        # Store in Qdrant (run in thread pool)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            lambda: self.client.upsert(collection_name=collection_name, points=[point], wait=True),
        )

        logger.debug(f"Stored memory {memory.id} in {collection_name}")

        return memory.qdrant_id

    async def batch_store_memories(
        self, memories: List[Memory], vectors: List[Vector]
    ) -> List[str]:
        """
        Store multiple memories efficiently.

        Args:
            memories: List of memories to store
            vectors: Corresponding vectors

        Returns:
            List of Qdrant point IDs
        """
        if len(memories) != len(vectors):
            raise ValueError("Number of memories and vectors must match")

        # Group by level
        level_groups: Dict[int, List[Tuple[Memory, Vector]]] = {0: [], 1: [], 2: []}

        for memory, vector in zip(memories, vectors):
            level_groups[memory.level].append((memory, vector))

        # Store each level batch
        all_ids = []

        for level, items in level_groups.items():
            if not items:
                continue

            collection_name = self._get_collection_for_level(level)
            points = []

            for memory, vector in items:
                if not memory.qdrant_id:
                    memory.qdrant_id = str(uuid.uuid4())

                points.append(
                    PointStruct(
                        id=memory.qdrant_id,
                        vector=vector.full_vector.tolist(),
                        payload=self._build_payload(memory),
                    )
                )
                all_ids.append(memory.qdrant_id)

            # Batch upsert
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                lambda: self.client.upsert(
                    collection_name=collection_name, points=points, wait=True
                ),
            )

            logger.info(f"Stored {len(points)} memories in {collection_name}")

        return all_ids

    async def search(
        self,
        query_vector: Vector,
        level: int,
        limit: int = 10,
        filters: Optional[SearchFilter] = None,
        score_threshold: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors in a specific level.

        Args:
            query_vector: Query vector
            level: Memory level to search
            limit: Maximum results
            filters: Optional search filters
            score_threshold: Minimum similarity score

        Returns:
            List of search results
        """
        collection_name = self._get_collection_for_level(level)

        # Build filter
        qdrant_filter = filters.to_qdrant_filter() if filters else None

        # Search parameters
        search_params = SearchParams(
            hnsw_ef=128, exact=False  # Higher ef for better recall  # Use HNSW index
        )

        # Execute search
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            lambda: self.client.search(
                collection_name=collection_name,
                query_vector=query_vector.full_vector.tolist(),
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True,
                search_params=search_params,
                score_threshold=score_threshold,
            ),
        )

        # Convert results
        search_results = []
        for result in results:
            # Reconstruct vector
            vector_array = result.vector
            vector = Vector.from_list(vector_array)

            search_results.append(
                VectorSearchResult(
                    memory_id=result.payload.get("memory_id", result.id),
                    score=result.score,
                    vector=vector,
                    payload=result.payload,
                )
            )

        return search_results

    async def search_all_levels(
        self,
        query_vector: Vector,
        limit_per_level: int = 10,
        filters: Optional[SearchFilter] = None,
        score_threshold: Optional[float] = None,
    ) -> Dict[int, List[VectorSearchResult]]:
        """
        Search across all memory levels.

        Args:
            query_vector: Query vector
            limit_per_level: Maximum results per level
            filters: Optional search filters
            score_threshold: Minimum similarity score

        Returns:
            Dictionary mapping level to search results
        """
        # Search all levels in parallel
        tasks = []
        for level in [0, 1, 2]:
            task = self.search(
                query_vector=query_vector,
                level=level,
                limit=limit_per_level,
                filters=filters,
                score_threshold=score_threshold,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        return {0: results[0], 1: results[1], 2: results[2]}

    async def get_by_id(
        self, qdrant_id: str, level: int
    ) -> Optional[Tuple[Vector, Dict[str, Any]]]:
        """
        Retrieve a vector by ID.

        Args:
            qdrant_id: Qdrant point ID
            level: Memory level

        Returns:
            Tuple of (vector, payload) or None if not found
        """
        collection_name = self._get_collection_for_level(level)

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            lambda: self.client.retrieve(
                collection_name=collection_name,
                ids=[qdrant_id],
                with_payload=True,
                with_vectors=True,
            ),
        )

        if results:
            result = results[0]
            vector = Vector.from_list(result.vector)
            return vector, result.payload

        return None

    async def update_payload(
        self, qdrant_id: str, level: int, payload_updates: Dict[str, Any]
    ) -> bool:
        """
        Update payload fields for a vector.

        Args:
            qdrant_id: Qdrant point ID
            level: Memory level
            payload_updates: Fields to update

        Returns:
            True if successful
        """
        collection_name = self._get_collection_for_level(level)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            lambda: self.client.set_payload(
                collection_name=collection_name,
                payload=payload_updates,
                points=[qdrant_id],
                wait=True,
            ),
        )

        return result.status == UpdateStatus.COMPLETED

    async def delete(self, qdrant_ids: List[str], level: int) -> bool:
        """
        Delete vectors by ID.

        Args:
            qdrant_ids: List of Qdrant point IDs
            level: Memory level

        Returns:
            True if successful
        """
        if not qdrant_ids:
            return True

        collection_name = self._get_collection_for_level(level)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            lambda: self.client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=qdrant_ids),
                wait=True,
            ),
        )

        return result.status == UpdateStatus.COMPLETED

    async def get_collection_stats(self, level: int) -> Dict[str, Any]:
        """
        Get statistics for a collection.

        Args:
            level: Memory level

        Returns:
            Collection statistics
        """
        collection_name = self._get_collection_for_level(level)

        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(
            self._executor, lambda: self.client.get_collection(collection_name)
        )

        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "segments_count": info.segments_count,
            "status": str(info.status),
            "optimizer_status": str(info.optimizer_status) if info.optimizer_status else None,
        }

    def _build_payload(self, memory: Memory) -> Dict[str, Any]:
        """Build Qdrant payload from memory."""
        payload = {
            "memory_id": memory.id,
            "project_id": memory.project_id,
            "meeting_id": memory.meeting_id,
            "memory_type": memory.memory_type.value,
            "content_type": memory.content_type.value,
            "importance_score": memory.importance_score,
            "created_at": int(memory.created_at.timestamp()),
        }

        # Add optional fields
        if memory.speaker:
            payload["speaker"] = memory.speaker

        if memory.timestamp_ms:
            payload["timestamp"] = memory.timestamp_ms / 1000.0

        # Level-specific fields
        if memory.level == 1:  # L1 contexts
            # Could add source_count for consolidated memories
            pass

        return payload

    async def close(self) -> None:
        """Close the vector store and cleanup resources."""
        self._executor.shutdown(wait=True)
        logger.info("Qdrant vector store closed")


# Singleton instance
_vector_store_instance: Optional[QdrantVectorStore] = None


def get_vector_store(
    host: str = "localhost", port: int = 6333, api_key: Optional[str] = None
) -> QdrantVectorStore:
    """Get or create the global vector store instance."""
    global _vector_store_instance

    if _vector_store_instance is None:
        _vector_store_instance = QdrantVectorStore(host=host, port=port, api_key=api_key)

    return _vector_store_instance
