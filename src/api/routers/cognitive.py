import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import numpy as np

from ..dependencies import get_db_connection, get_vector_store_instance
from ...embedding.onnx_encoder import get_encoder
from ...embedding.vector_manager import get_vector_manager
from ...cognitive.activation.basic_activation_engine import BasicActivationEngine, ActivationResult
from ...storage.sqlite.repositories import get_memory_repository, get_memory_connection_repository
from ...models.entities import Memory

logger = logging.getLogger(__name__)
router = APIRouter()


class CognitiveQueryRequest(BaseModel):
    """Request model for cognitive query."""
    query: str = Field(..., description="Natural language query for cognitive search")
    project_id: Optional[str] = Field(None, description="Optional project ID to filter memories")
    max_activations: int = Field(50, ge=1, le=200, description="Maximum number of memories to activate")
    activation_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum activation strength to include a memory")

    class Config:
        schema_extra = {
            "example": {
                "query": "What is the latest status on the UAT Testing Project?",
                "project_id": "proj-uat-123",
                "max_activations": 100,
                "activation_threshold": 0.6
            }
        }


class ActivatedMemory(BaseModel):
    """Details of an activated memory."""
    memory_id: str
    content: str
    speaker: Optional[str]
    meeting_id: str
    memory_type: str
    activation_strength: float
    activation_path: List[str]
    explanation: str


class CognitiveQueryResult(BaseModel):
    """Response model for cognitive query results."""
    query: str
    total_activated_memories: int
    core_memories: List[ActivatedMemory]
    peripheral_memories: List[ActivatedMemory]
    activation_time_ms: float
    status: str
    errors: List[str] = []


@router.post("/cognitive/query", response_model=CognitiveQueryResult)
async def cognitive_query(
    request: CognitiveQueryRequest,
    db=Depends(get_db_connection),
    vector_store=Depends(get_vector_store_instance),
) -> CognitiveQueryResult:
    """
    Performs a cognitive query using activation spreading to retrieve relevant memories.

    Args:
        request: Cognitive query parameters

    Returns:
        CognitiveQueryResult with activated memories and explanations
    """
    try:
        # Initialize components
        encoder = get_encoder()
        vector_manager = get_vector_manager()
        memory_repo = get_memory_repository(db)
        connection_repo = get_memory_connection_repository(db)

        # Encode query to get context vector
        query_embedding = encoder.encode(request.query, normalize=True)
        # Use default dimensions for query context for now, can be enhanced later
        default_dimensions = np.full(16, 0.5)
        context_vector = vector_manager.compose_vector(query_embedding, default_dimensions)

        # Initialize and run activation engine
        activation_engine = BasicActivationEngine(
            memory_repo=memory_repo,
            connection_repo=connection_repo,
            vector_store=vector_store,
            core_threshold=0.7, # Default, can be made configurable
            peripheral_threshold=0.5, # Default, can be made configurable
            decay_factor=0.8 # Default, can be made configurable
        )

        activation_result: ActivationResult = await activation_engine.activate_memories(
            context=context_vector,
            threshold=request.activation_threshold,
            max_activations=request.max_activations,
            project_id=request.project_id
        )

        # Prepare response
        core_memories_response = []
        for mem in activation_result.core_memories:
            core_memories_response.append(ActivatedMemory(
                memory_id=mem.id,
                content=mem.content,
                speaker=mem.speaker,
                meeting_id=mem.meeting_id,
                memory_type=mem.memory_type.value,
                activation_strength=activation_result.activation_strengths.get(mem.id, 0.0),
                activation_path=activation_result.activation_paths.get(mem.id, []),
                explanation=activation_result.activation_explanations.get(mem.id, "")
            ))
        
        peripheral_memories_response = []
        for mem in activation_result.peripheral_memories:
            peripheral_memories_response.append(ActivatedMemory(
                memory_id=mem.id,
                content=mem.content,
                speaker=mem.speaker,
                meeting_id=mem.meeting_id,
                memory_type=mem.memory_type.value,
                activation_strength=activation_result.activation_strengths.get(mem.id, 0.0),
                activation_path=activation_result.activation_paths.get(mem.id, []),
                explanation=activation_result.activation_explanations.get(mem.id, "")
            ))

        return CognitiveQueryResult(
            query=request.query,
            total_activated_memories=activation_result.total_activated,
            core_memories=core_memories_response,
            peripheral_memories=peripheral_memories_response,
            activation_time_ms=activation_result.activation_time_ms,
            status="success"
        )

    except Exception as e:
        logger.error(f"Cognitive query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cognitive query failed: {str(e)}")
