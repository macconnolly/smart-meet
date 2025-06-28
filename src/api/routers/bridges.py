import logging
import time
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import numpy as np

from ..dependencies import get_db_connection, get_vector_store_instance
from ...embedding.onnx_encoder import get_encoder
from ...embedding.vector_manager import get_vector_manager
from ...extraction.dimensions.dimension_analyzer import get_dimension_analyzer, DimensionExtractionContext
from ...storage.sqlite.repositories import get_memory_repository
from ...storage.qdrant.vector_store import SearchFilter
from ...cognitive.retrieval.bridge_discovery import SimpleBridgeDiscovery, BridgeMemory

logger = logging.getLogger(__name__)
router = APIRouter()


class BridgeDiscoveryRequest(BaseModel):
    """Request model for bridge discovery."""
    query: str = Field(..., description="Natural language query for bridge discovery")
    project_id: Optional[str] = Field(None, description="Optional project ID to filter memories")
    max_bridges: int = Field(5, ge=1, le=10, description="Maximum number of bridges to return")
    search_expansion: int = Field(50, ge=10, le=200, description="Number of candidates to explore for bridge discovery")
    novelty_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for novelty in bridge scoring")
    connection_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for connection potential in bridge scoring")
    min_bridge_score: float = Field(0.6, ge=0.0, le=1.0, description="Minimum score for a memory to qualify as a bridge")

    class Config:
        schema_extra = {
            "example": {
                "query": "How can we connect our sales strategy with product development?",
                "project_id": "proj-sales-dev",
                "max_bridges": 3
            }
        }


class DiscoveredBridge(BaseModel):
    """Details of a discovered bridge memory."""
    memory_id: str
    content: str
    speaker: Optional[str]
    meeting_id: str
    memory_type: str
    novelty_score: float
    connection_potential: float
    surprise_score: float
    bridge_score: float
    explanation: str
    connected_concepts: List[str]


class BridgeDiscoveryResult(BaseModel):
    """Response model for bridge discovery results."""
    query: str
    discovered_bridges: List[DiscoveredBridge]
    discovery_time_ms: float
    status: str
    errors: List[str] = []


@router.post("/discover-bridges", response_model=BridgeDiscoveryResult)
async def discover_bridges(
    request: BridgeDiscoveryRequest,
    db=Depends(get_db_connection),
    vector_store=Depends(get_vector_store_instance),
) -> BridgeDiscoveryResult:
    """
    Discovers serendipitous connections (bridges) between memories.

    Args:
        request: Bridge discovery parameters

    Returns:
        BridgeDiscoveryResult with discovered bridges and explanations
    """
    start_time = time.time()
    errors = []
    discovered_bridges_response = []

    try:
        encoder = get_encoder()
        vector_manager = get_vector_manager()
        memory_repo = get_memory_repository(db)

        # 1. Encode query to get context vector (400D)
        query_embedding = encoder.encode(request.query, normalize=True)
        # For bridge discovery query, cognitive dimensions are important
        dimension_analyzer = get_dimension_analyzer()
        query_dim_context = DimensionExtractionContext(content_type="query")
        query_cognitive_dimensions = await dimension_analyzer.analyze(request.query, query_dim_context)
        query_context_vector = vector_manager.compose_vector(query_embedding, query_cognitive_dimensions)

        # 2. Retrieve initial set of memories (e.g., top N semantic search results)
        # This forms the "retrieved_memories" for bridge discovery
        # Re-using the existing /search logic for initial retrieval
        # For simplicity, let's do a direct vector search here.
        initial_search_results = await vector_store.search_all_levels(
            query_vector=query_context.full_vector.tolist(),
            limit_per_level=request.search_expansion // 3, # Distribute initial search
            filters=SearchFilter(project_id=request.project_id)
        )
        retrieved_memories_ids = [res.payload.get("memory_id") for level_res in initial_search_results.values() for res in level_res]
        retrieved_memories = [await memory_repo.get_by_id(mid) for mid in retrieved_memories_ids if mid]
        retrieved_memories = [m for m in retrieved_memories if m] # Filter out None

        # 3. Initialize and run Bridge Discovery engine
        bridge_discovery_engine = SimpleBridgeDiscovery(
            memory_repo=memory_repo,
            vector_store=vector_store,
            novelty_weight=request.novelty_weight,
            connection_weight=request.connection_weight,
            min_bridge_score=request.min_bridge_score,
        )

        bridges: List[BridgeMemory] = await bridge_discovery_engine.discover_bridges(
            query_context=query_context,
            retrieved_memories=retrieved_memories,
            max_bridges=request.max_bridges,
            search_expansion=request.search_expansion
        )

        # Prepare response
        for bridge_mem in bridges:
            discovered_bridges_response.append(DiscoveredBridge(
                memory_id=bridge_mem.memory.id,
                content=bridge_mem.memory.content,
                speaker=bridge_mem.memory.speaker,
                meeting_id=bridge_mem.memory.meeting_id,
                memory_type=bridge_mem.memory.memory_type.value,
                novelty_score=bridge_mem.novelty_score,
                connection_potential=bridge_mem.connection_potential,
                surprise_score=bridge_mem.surprise_score,
                bridge_score=bridge_mem.bridge_score,
                explanation=bridge_mem.explanation,
                connected_concepts=bridge_mem.connected_concepts
            ))

        return BridgeDiscoveryResult(
            query=request.query,
            discovered_bridges=discovered_bridges_response,
            discovery_time_ms=(time.time() - start_time) * 1000,
            status="success"
        )

    except Exception as e:
        logger.error(f"Bridge discovery failed: {e}", exc_info=True)
        errors.append(str(e))
        raise HTTPException(status_code=500, detail=f"Bridge discovery failed: {str(e)}")
