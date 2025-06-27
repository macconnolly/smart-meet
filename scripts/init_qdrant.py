#!/usr/bin/env python3
"""
Initialize Qdrant collections for 3-tier memory system.

Reference: IMPLEMENTATION_GUIDE.md - Day 4: Storage Layer
Creates L0, L1, L2 collections with optimized HNSW settings.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.qdrant.vector_store import get_vector_store
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def init_qdrant_collections():
    """
    Initialize all Qdrant collections.
    
    TODO Day 4:
    - [ ] Connect to Qdrant
    - [ ] Create L0 collection (concepts)
    - [ ] Create L1 collection (contexts)
    - [ ] Create L2 collection (episodes)
    - [ ] Verify collections created
    - [ ] Set optimal HNSW parameters
    """
    vector_store = get_vector_store()
    
    try:
        logger.info("Initializing Qdrant collections...")
        
        # TODO Day 4: Initialize collections
        await vector_store.initialize_collections()
        
        # TODO Day 4: Verify each collection
        collections = ["cognitive_concepts", "cognitive_contexts", "cognitive_episodes"]
        for collection in collections:
            info = await vector_store.get_collection_info(collection)
            logger.info(f"{collection}: {info}")
        
        logger.info("Qdrant initialization complete!")
        
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}")
        raise


def main():
    """Main entry point."""
    # TODO Day 4: Add command line arguments
    asyncio.run(init_qdrant_collections())


if __name__ == "__main__":
    main()
