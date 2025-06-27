#!/usr/bin/env python3
"""
Simple Qdrant initialization script to create collections.
"""

import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfig


def create_collections():
    """Create the three-tier Qdrant collections."""
    client = QdrantClient(host="localhost", port=6333)

    print("🚀 Initializing Qdrant collections...")

    # Define collections with tier-specific optimization
    collections = [
        {
            "name": "cognitive_concepts",  # L0: High-level concepts
            "config": {
                "size": 400,  # 384D semantic + 16D cognitive
                "distance": Distance.COSINE,
                "hnsw_config": HnswConfig(
                    m=32,  # High quality connections
                    ef_construct=400,
                    full_scan_threshold=1000,
                ),
            },
        },
        {
            "name": "cognitive_contexts",  # L1: Contextual memories
            "config": {
                "size": 400,
                "distance": Distance.COSINE,
                "hnsw_config": HnswConfig(
                    m=24,  # Balanced quality
                    ef_construct=300,
                    full_scan_threshold=5000,
                ),
            },
        },
        {
            "name": "cognitive_episodes",  # L2: Episodic memories
            "config": {
                "size": 400,
                "distance": Distance.COSINE,
                "hnsw_config": HnswConfig(
                    m=16,  # Fast search
                    ef_construct=200,
                    full_scan_threshold=10000,
                ),
            },
        },
    ]

    # Create each collection
    for collection in collections:
        try:
            # Try to delete existing collection first
            try:
                client.delete_collection(collection["name"])
                print(f"  ✓ Deleted existing collection: {collection['name']}")
            except:
                pass  # Collection doesn't exist

            # Create new collection
            client.create_collection(
                collection_name=collection["name"],
                vectors_config=VectorParams(
                    size=collection["config"]["size"], distance=collection["config"]["distance"]
                ),
                hnsw_config=collection["config"]["hnsw_config"],
            )
            print(f"  ✓ Created collection: {collection['name']}")

        except Exception as e:
            print(f"  ✗ Error creating {collection['name']}: {e}")
            raise

    # Verify collections
    print("\n📊 Verifying collections...")
    collections_info = client.get_collections()
    for collection in collections_info.collections:
        info = client.get_collection(collection.name)
        print(f"  • {collection.name}:")
        print(f"    - Vectors: {info.points_count}")
        print(f"    - Vector size: {info.config.params.vectors.size}")
        print(f"    - Distance: {info.config.params.vectors.distance}")

    print("\n✅ Qdrant initialization complete!")

    # Test with sample data from our SQLite database
    print("\n🔍 Testing vector insertion...")

    # Sample vector (would come from ONNX encoder in real implementation)
    import numpy as np

    # Create sample 400D vectors for our memories
    sample_memories = [
        {
            "id": "qdrant_mem_001",
            "vector": np.random.rand(400).tolist(),
            "payload": {
                "memory_id": "mem_001",
                "content": "Digital transformation in three phases",
                "content_type": "decision",
                "level": 2,
                "importance_score": 0.95,
            },
        },
        {
            "id": "qdrant_mem_002",
            "vector": np.random.rand(400).tolist(),
            "payload": {
                "memory_id": "mem_002",
                "content": "Monolithic architecture bottleneck",
                "content_type": "issue",
                "level": 2,
                "importance_score": 0.88,
            },
        },
    ]

    # Insert into L2 (episodes) collection
    from qdrant_client.models import PointStruct

    points = [
        PointStruct(id=mem["id"], vector=mem["vector"], payload=mem["payload"])
        for mem in sample_memories
    ]

    client.upsert(collection_name="cognitive_episodes", points=points)

    print(f"  ✓ Inserted {len(points)} sample vectors into cognitive_episodes")

    # Test search
    query_vector = np.random.rand(400).tolist()
    search_results = client.search(
        collection_name="cognitive_episodes", query_vector=query_vector, limit=5
    )

    print(f"  ✓ Search test completed, found {len(search_results)} results")

    return True


if __name__ == "__main__":
    try:
        create_collections()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)
