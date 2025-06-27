"""
Qdrant vector storage implementation for cognitive memory system.

This module implements hierarchical memory storage using Qdrant vector database
with 3-tier collections: L0 (concepts), L1 (contexts), L2 (episodes).

Reference: IMPLEMENTATION_GUIDE.md - Phase 2: Enhanced Storage Integration
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from ...storage.qdrant.vector_store import QdrantVectorStore
from ...models.entities import Memory


@dataclass
class CollectionConfig:
    """Configuration for a Qdrant collection."""

    name: str
    vector_size: int
    distance: Distance
    on_disk_payload: bool = True
    replication_factor: int = 1
    write_consistency_factor: int = 1
    optimizers_indexing_threshold: int = 20000
    segments_number: int = 2


class QdrantCollectionManager:
    """Manages Qdrant collections for hierarchical memory storage."""

    def __init__(self, client: QdrantClient, vector_size: int):
        """Initialize collection manager."""
        self.client = client
        self.vector_size = vector_size
        self.collections = {
            0: CollectionConfig(
                name="L0_cognitive_concepts",
                vector_size=vector_size,
                distance=Distance.COSINE,
            ),
            1: CollectionConfig(
                name="L1_cognitive_contexts",
                vector_size=vector_size,
                distance=Distance.COSINE,
            ),
            2: CollectionConfig(
                name="L2_cognitive_episodes",
                vector_size=vector_size,
                distance=Distance.COSINE,
            ),
        }

    def initialize_collections(self) -> bool:
        """Initialize all hierarchical collections."""
        try:
            for level, config in self.collections.items():
                if not self._collection_exists(config.name):
                    self._create_collection(config)
                    logger.info(
                        f"Created collection for level {level}", collection=config.name
                    )
                else:
                    logger.debug(
                        f"Collection already exists for level {level}",
                        collection=config.name,
                    )
            return True
        except Exception as e:
            logger.error("Failed to initialize collections", error=str(e))
            return False

    def _collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception:
            return False

    def _create_collection(self, config: CollectionConfig) -> None:
        """Create a single collection with optimization settings."""
        self.client.create_collection(
            collection_name=config.name,
            vectors_config=VectorParams(
                size=config.vector_size,
                distance=config.distance,
                on_disk=config.on_disk_payload,
            ),
            replication_factor=config.replication_factor,
            write_consistency_factor=config.write_consistency_factor,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=config.optimizers_indexing_threshold,
                memmap_threshold=config.optimizers_indexing_threshold,
            ),
            shard_number=config.segments_number,
        )

    def get_collection_name(self, level: int) -> str:
        """Get collection name for memory level."""
        if level not in self.collections:
            raise ValueError(f"Invalid memory level: {level}")
        return self.collections[level].name

    def delete_all_collections(self) -> bool:
        """Delete all collections (used for cleanup)."""
        try:
            for config in self.collections.values():
                if self._collection_exists(config.name):
                    self.client.delete_collection(config.name)
                    logger.info("Deleted collection", collection=config.name)
            return True
        except Exception as e:
            logger.error("Failed to delete collections", error=str(e))
            return False


class VectorSearchEngine:
    """Sophisticated vector search with metadata filtering."""

    def __init__(
        self, client: QdrantClient, collection_manager: QdrantCollectionManager
    ):
        """Initialize search engine."""
        self.client = client
        self.collection_manager = collection_manager

    def search_level(
        self,
        level: int,
        query_vector: np.ndarray,
        k: int,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search within a specific memory level."""
        collection_name = self.collection_manager.get_collection_name(level)

        # Convert array to list for Qdrant
        query_list = (
            query_vector.tolist()
            if isinstance(query_vector, np.ndarray)
            else query_vector
        )

        # Build filter conditions
        filter_conditions = None
        if filters:
            must_conditions = []
            for key, value in filters.items():
                must_conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
            filter_conditions = models.Filter(must=must_conditions)

        try:
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_list,
                limit=k,
                query_filter=filter_conditions,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )

            results = []
            for point in search_result:
                result = {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload,
                    "collection": collection_name,
                    "level": level
                }
                results.append(result)

            logger.debug(
                "Vector search completed",
                level=level,
                collection=collection_name,
                results_count=len(results),
                query_filters=filters,
            )

            return results

        except Exception as e:
            logger.error(
                "Vector search failed",
                level=level,
                collection=collection_name,
                error=str(e),
            )
            return []

    def search_cross_level(
        self,
        query_vector: np.ndarray,
        k_per_level: int,
        levels: Optional[List[int]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Search across multiple memory levels."""
        if levels is None:
            levels = [0, 1, 2]  # All levels

        results = {}
        for level in levels:
            level_results = self.search_level(
                level=level, 
                query_vector=query_vector, 
                k=k_per_level, 
                filters=filters
            )
            results[level] = level_results

        return results


class HierarchicalMemoryStorage:
    """
    Enhanced Qdrant-based hierarchical memory storage system.

    Implements the VectorStorage interface with 3-tier hierarchical collections
    optimized for cognitive memory patterns and retrieval.
    """

    def __init__(
        self,
        vector_size: int = 400,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: Optional[int] = None,
        prefer_grpc: bool = True,
        timeout: Optional[int] = None,
    ):
        """
        Initialize hierarchical memory storage.

        Args:
            vector_size: Dimension of embedding vectors (400D)
            host: Qdrant server host
            port: Qdrant HTTP port
            grpc_port: Qdrant gRPC port (defaults to port + 1)
            prefer_grpc: Whether to prefer gRPC connection
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.grpc_port = grpc_port or (port + 1)
        self.timeout = timeout
        self.vector_size = vector_size

        # Initialize Qdrant client
        try:
            self.client = QdrantClient(
                host=host,
                port=port,
                grpc_port=self.grpc_port,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
            )
            logger.info(
                "Connected to Qdrant server",
                host=host,
                port=port,
                grpc_port=self.grpc_port,
                prefer_grpc=prefer_grpc,
            )
        except Exception as e:
            logger.error("Failed to connect to Qdrant server", error=str(e))
            raise

        # Initialize collection manager and search engine
        self.collection_manager = QdrantCollectionManager(self.client, vector_size)
        self.search_engine = VectorSearchEngine(self.client, self.collection_manager)

        # Initialize collections
        if not self.collection_manager.initialize_collections():
            raise RuntimeError("Failed to initialize Qdrant collections")

    def store_vector(
        self, id: str, vector: np.ndarray, metadata: Dict[str, Any]
    ) -> None:
        """
        Store a vector with associated metadata in appropriate hierarchy level.

        Args:
            id: Unique identifier for the vector
            vector: Cognitive embedding vector (400D)
            metadata: Associated metadata including hierarchy_level
        """
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)

        # Validate vector dimensions
        if vector.shape[-1] != self.vector_size:
            raise ValueError(
                f"Expected {self.vector_size}-dimensional vector, got {vector.shape[-1]}"
            )

        # Extract hierarchy level from metadata
        hierarchy_level = metadata.get("level", 2)  # Default to episodes
        if hierarchy_level not in [0, 1, 2]:
            raise ValueError(f"Invalid hierarchy level: {hierarchy_level}")

        # Get collection name for the level
        collection_name = self.collection_manager.get_collection_name(hierarchy_level)

        # Convert vector to list
        vector_list = vector.tolist() if vector.ndim == 1 else vector.flatten().tolist()

        # Create point structure
        point = PointStruct(id=id, vector=vector_list, payload=metadata)

        try:
            # Store in Qdrant
            self.client.upsert(collection_name=collection_name, points=[point])

            logger.debug(
                "Vector stored successfully",
                id=id,
                level=hierarchy_level,
                collection=collection_name,
                metadata_keys=list(metadata.keys()),
            )

        except Exception as e:
            logger.error(
                "Failed to store vector",
                id=id,
                level=hierarchy_level,
                collection=collection_name,
                error=str(e),
            )
            raise

    def search_similar(
        self, 
        query_vector: np.ndarray, 
        k: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors across all hierarchy levels.

        Args:
            query_vector: Query vector for similarity search
            k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results sorted by score
        """
        # Search across all levels
        k_per_level = max(1, k // 3)  # Distribute k across levels
        cross_level_results = self.search_engine.search_cross_level(
            query_vector=query_vector, 
            k_per_level=k_per_level, 
            filters=filters
        )

        # Combine and sort results
        all_results = []
        for results in cross_level_results.values():
            all_results.extend(results)

        # Sort by score (descending) and limit to k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:k]

    def search_by_level(
        self,
        query_vector: np.ndarray,
        level: int,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search within a specific hierarchy level."""
        return self.search_engine.search_level(
            level=level, 
            query_vector=query_vector, 
            k=k, 
            filters=filters
        )

    def delete_vector(self, id: str) -> bool:
        """
        Delete a vector by ID across all collections.

        Args:
            id: Vector ID to delete

        Returns:
            True if deleted, False otherwise
        """
        success = False

        for level in [0, 1, 2]:
            collection_name = self.collection_manager.get_collection_name(level)
            try:
                result = self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=[id]),
                )
                if result:
                    success = True
                    logger.debug("Vector deleted", id=id, collection=collection_name)
            except Exception as e:
                logger.debug(
                    "Vector not found in collection (expected)",
                    id=id,
                    collection=collection_name,
                    error=str(e),
                )

        return success

    def update_vector(
        self, id: str, vector: np.ndarray, metadata: Dict[str, Any]
    ) -> bool:
        """
        Update an existing vector and its metadata.

        Args:
            id: Vector ID to update
            vector: New vector data
            metadata: New metadata

        Returns:
            True if updated, False otherwise
        """
        try:
            # Store vector (upsert will update if exists)
            self.store_vector(id, vector, metadata)
            return True
        except Exception as e:
            logger.error("Failed to update vector", id=id, error=str(e))
            return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for all collections."""
        stats = {}

        for level in [0, 1, 2]:
            collection_name = self.collection_manager.get_collection_name(level)
            try:
                info = self.client.get_collection(collection_name)
                stats[f"level_{level}"] = {
                    "collection_name": collection_name,
                    "vectors_count": info.vectors_count,
                    "indexed_vectors_count": info.indexed_vectors_count,
                    "points_count": info.points_count,
                    "segments_count": info.segments_count,
                    "status": info.status,
                }
            except Exception as e:
                logger.error(f"Failed to get stats for level {level}", error=str(e))
                stats[f"level_{level}"] = {"error": str(e)}

        return stats

    def optimize_collections(self) -> bool:
        """Optimize all collections for better performance."""
        try:
            for level in [0, 1, 2]:
                collection_name = self.collection_manager.get_collection_name(level)
                self.client.update_collection(
                    collection_name=collection_name,
                    optimizer_config=models.OptimizersConfigDiff(
                        indexing_threshold=20000, 
                        memmap_threshold=20000
                    ),
                )
                logger.debug(
                    "Collection optimized", 
                    level=level, 
                    collection=collection_name
                )
            return True
        except Exception as e:
            logger.error("Failed to optimize collections", error=str(e))
            return False

    def close(self) -> None:
        """Close connection to Qdrant server."""
        try:
            # QdrantClient doesn't have explicit close in newer versions
            logger.info("Qdrant connection closed")
        except Exception as e:
            logger.error("Error closing Qdrant connection", error=str(e))

    def __enter__(self) -> "HierarchicalMemoryStorage":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
