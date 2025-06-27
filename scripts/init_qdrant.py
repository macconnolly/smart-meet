"""
Initialize Qdrant vector database with 3-tier collections.

This script creates the three collections for the hierarchical memory system:
- L0: Cognitive concepts (highest abstractions)
- L1: Cognitive contexts (patterns and themes)
- L2: Cognitive episodes (raw memories)

Usage:
    python scripts/init_qdrant.py [--host HOST] [--port PORT] [--reset]
"""

import argparse
import logging
import sys
from typing import Dict, Any, List
import time

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        CreateCollection,
        OptimizersConfigDiff,
        HnswConfigDiff,
        CollectionInfo,
        PayloadSchemaType,
    )
except ImportError:
    print("Qdrant client not installed. Please install: pip install qdrant-client")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Collection configurations for 3-tier system
COLLECTIONS_CONFIG = {
    "L0_cognitive_concepts": {
        "description": "Highest level abstractions and semantic memories",
        "vector_size": 400,
        "distance": Distance.COSINE,
        "hnsw_config": {
            "m": 32,  # Higher connectivity for better recall
            "ef_construct": 200,
            "full_scan_threshold": 1000,
        },
        "optimizer_config": {"memmap_threshold": 5000, "indexing_threshold": 10000},
        "payload_schema": {
            "project_id": PayloadSchemaType.KEYWORD,
            "memory_type": PayloadSchemaType.KEYWORD,
            "content_type": PayloadSchemaType.KEYWORD,
            "importance_score": PayloadSchemaType.FLOAT,
            "created_at": PayloadSchemaType.INTEGER,
        },
    },
    "L1_cognitive_contexts": {
        "description": "Mid-level patterns and consolidated memories",
        "vector_size": 400,
        "distance": Distance.COSINE,
        "hnsw_config": {
            "m": 24,  # Moderate connectivity
            "ef_construct": 150,
            "full_scan_threshold": 5000,
        },
        "optimizer_config": {"memmap_threshold": 20000, "indexing_threshold": 50000},
        "payload_schema": {
            "project_id": PayloadSchemaType.KEYWORD,
            "memory_type": PayloadSchemaType.KEYWORD,
            "content_type": PayloadSchemaType.KEYWORD,
            "importance_score": PayloadSchemaType.FLOAT,
            "created_at": PayloadSchemaType.INTEGER,
            "source_count": PayloadSchemaType.INTEGER,
        },
    },
    "L2_cognitive_episodes": {
        "description": "Raw episodic memories from meetings",
        "vector_size": 400,
        "distance": Distance.COSINE,
        "hnsw_config": {
            "m": 16,  # Lower connectivity for efficiency
            "ef_construct": 100,
            "full_scan_threshold": 10000,
        },
        "optimizer_config": {"memmap_threshold": 50000, "indexing_threshold": 100000},
        "payload_schema": {
            "project_id": PayloadSchemaType.KEYWORD,
            "meeting_id": PayloadSchemaType.KEYWORD,
            "memory_type": PayloadSchemaType.KEYWORD,
            "content_type": PayloadSchemaType.KEYWORD,
            "speaker": PayloadSchemaType.KEYWORD,
            "timestamp": PayloadSchemaType.FLOAT,
            "importance_score": PayloadSchemaType.FLOAT,
            "created_at": PayloadSchemaType.INTEGER,
        },
    },
}


class QdrantInitializer:
    """Initialize and manage Qdrant collections for the cognitive system."""

    def __init__(self, host: str = "localhost", port: int = 6333, api_key: str = None):
        """
        Initialize Qdrant client.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            api_key: Optional API key for authentication
        """
        self.client = QdrantClient(host=host, port=port, api_key=api_key, timeout=30)

        # Test connection
        try:
            self.client.get_collections()
            logger.info(f"Connected to Qdrant at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def create_collection(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Create a single collection with configuration.

        Args:
            name: Collection name
            config: Collection configuration

        Returns:
            True if created successfully
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if any(col.name == name for col in collections):
                logger.info(f"Collection '{name}' already exists")
                return False

            # Create collection
            logger.info(f"Creating collection '{name}': {config['description']}")

            # Configure HNSW
            hnsw_config = HnswConfigDiff(
                m=config["hnsw_config"]["m"],
                ef_construct=config["hnsw_config"]["ef_construct"],
                full_scan_threshold=config["hnsw_config"]["full_scan_threshold"],
            )

            # Configure optimizers
            optimizer_config = OptimizersConfigDiff(
                memmap_threshold=config["optimizer_config"]["memmap_threshold"],
                indexing_threshold=config["optimizer_config"]["indexing_threshold"],
            )

            # Create collection
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=config["vector_size"], distance=config["distance"]
                ),
                hnsw_config=hnsw_config,
                optimizers_config=optimizer_config,
            )

            logger.info(f"✓ Created collection '{name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection '{name}': {e}")
            raise

    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.

        Args:
            name: Collection name

        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete_collection(name)
            logger.info(f"Deleted collection '{name}'")
            return True
        except Exception as e:
            logger.warning(f"Could not delete collection '{name}': {e}")
            return False

    def initialize_all_collections(self, reset: bool = False) -> None:
        """
        Initialize all collections for the 3-tier system.

        Args:
            reset: Whether to delete existing collections first
        """
        logger.info("Initializing Qdrant collections for 3-tier cognitive system")

        # Reset if requested
        if reset:
            logger.warning("Resetting existing collections...")
            for name in COLLECTIONS_CONFIG.keys():
                self.delete_collection(name)
            time.sleep(1)  # Allow deletion to complete

        # Create collections
        created = 0
        for name, config in COLLECTIONS_CONFIG.items():
            if self.create_collection(name, config):
                created += 1

        logger.info(f"Created {created} new collections")

        # Verify all collections exist
        self.verify_collections()

    def verify_collections(self) -> bool:
        """
        Verify all required collections exist with correct configuration.

        Returns:
            True if all collections are properly configured
        """
        logger.info("Verifying collections...")

        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            all_valid = True

            for name, expected_config in COLLECTIONS_CONFIG.items():
                if name not in collection_names:
                    logger.error(f"✗ Missing collection: {name}")
                    all_valid = False
                    continue

                # Get collection info
                info = self.client.get_collection(name)

                # Verify vector size
                if info.config.params.vectors.size != expected_config["vector_size"]:
                    logger.error(
                        f"✗ Collection '{name}' has wrong vector size: "
                        f"{info.config.params.vectors.size} (expected {expected_config['vector_size']})"
                    )
                    all_valid = False

                # Verify distance metric
                if info.config.params.vectors.distance != expected_config["distance"]:
                    logger.error(
                        f"✗ Collection '{name}' has wrong distance metric: "
                        f"{info.config.params.vectors.distance} (expected {expected_config['distance']})"
                    )
                    all_valid = False

                # Log collection stats
                logger.info(
                    f"✓ Collection '{name}': "
                    f"vectors={info.vectors_count}, "
                    f"indexed={info.indexed_vectors_count}"
                )

            if all_valid:
                logger.info("✅ All collections verified successfully!")
            else:
                logger.error("❌ Collection verification failed!")

            return all_valid

        except Exception as e:
            logger.error(f"Failed to verify collections: {e}")
            return False

    def insert_test_vectors(self) -> None:
        """Insert test vectors to verify collections are working."""
        logger.info("Inserting test vectors...")

        import numpy as np
        import uuid
        from datetime import datetime

        for name in COLLECTIONS_CONFIG.keys():
            # Create test vector
            vector = np.random.randn(400).astype(np.float32)
            vector = vector / np.linalg.norm(vector)  # Normalize

            # Create test point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "test": True,
                    "project_id": "test-project",
                    "memory_type": "episodic",
                    "content_type": "test",
                    "importance_score": 0.5,
                    "created_at": int(datetime.now().timestamp()),
                },
            )

            # Insert point
            self.client.upsert(collection_name=name, points=[point])

            logger.info(f"✓ Inserted test vector into '{name}'")

        # Verify insertions
        for name in COLLECTIONS_CONFIG.keys():
            count = self.client.get_collection(name).vectors_count
            logger.info(f"Collection '{name}' now has {count} vectors")

    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all collections.

        Returns:
            Dictionary of collection statistics
        """
        stats = {}

        for name in COLLECTIONS_CONFIG.keys():
            try:
                info = self.client.get_collection(name)
                stats[name] = {
                    "vectors_count": info.vectors_count,
                    "indexed_vectors_count": info.indexed_vectors_count,
                    "points_count": info.points_count,
                    "segments_count": info.segments_count,
                    "status": info.status,
                    "optimizer_status": info.optimizer_status,
                }
            except Exception as e:
                logger.error(f"Failed to get stats for '{name}': {e}")
                stats[name] = {"error": str(e)}

        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize Qdrant collections for Cognitive Meeting Intelligence"
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Qdrant server host (default: localhost)"
    )
    parser.add_argument("--port", type=int, default=6333, help="Qdrant server port (default: 6333)")
    parser.add_argument("--api-key", type=str, help="Qdrant API key for authentication")
    parser.add_argument(
        "--reset", action="store_true", help="Delete existing collections before creating new ones"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing collections"
    )
    parser.add_argument(
        "--test-vectors", action="store_true", help="Insert test vectors after initialization"
    )
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")

    args = parser.parse_args()

    try:
        # Initialize client
        initializer = QdrantInitializer(host=args.host, port=args.port, api_key=args.api_key)

        if args.verify_only:
            # Just verify
            initializer.verify_collections()
        elif args.stats:
            # Show statistics
            stats = initializer.get_collection_stats()
            logger.info("Collection statistics:")
            for name, stat in stats.items():
                logger.info(f"  {name}: {stat}")
        else:
            # Initialize collections
            initializer.initialize_all_collections(reset=args.reset)

            # Insert test vectors if requested
            if args.test_vectors:
                initializer.insert_test_vectors()

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
