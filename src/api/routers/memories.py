"""
Memory management API endpoints.

This module provides endpoints for ingesting meeting transcripts,
searching memories, and managing the memory lifecycle.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Query
from pydantic import BaseModel, Field

from ..dependencies import get_db_connection, get_vector_store_instance
from ...models.entities import (
    Meeting,
    Memory,
    Project,
    MeetingType,
    ContentType,
    ProjectType,
    ProjectStatus,
    MeetingCategory,
)
from ...pipeline.ingestion_pipeline import (
    create_ingestion_pipeline,
    IngestionResult,
    PipelineConfig,
)
from ...storage.sqlite.repositories import (
    get_meeting_repository,
    get_memory_repository,
    get_project_repository,
)
from ...storage.qdrant.vector_store import SearchFilter, VectorSearchResult
from ...embedding.onnx_encoder import get_encoder
from ...embedding.vector_manager import get_vector_manager

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class ProjectCreate(BaseModel):
    """Request model for creating a project."""

    name: str = Field(..., description="Project name")
    client_name: str = Field(..., description="Client name")
    project_type: str = Field(default="other", description="Project type")
    project_manager: Optional[str] = Field(None, description="Project manager name")
    engagement_code: Optional[str] = Field(None, description="Unique engagement code")

    class Config:
        schema_extra = {
            "example": {
                "name": "Digital Transformation Strategy",
                "client_name": "Acme Corp",
                "project_type": "transformation",
                "project_manager": "John Smith",
            }
        }


class MeetingIngest(BaseModel):
    """Request model for ingesting a meeting."""

    project_id: str = Field(..., description="Project ID")
    title: str = Field(..., description="Meeting title")
    meeting_type: str = Field(default="working_session", description="Type of meeting")
    start_time: datetime = Field(..., description="Meeting start time")
    end_time: datetime = Field(..., description="Meeting end time")
    participants: List[Dict[str, Any]] = Field(
        default_factory=list, description="Meeting participants"
    )
    transcript: str = Field(..., description="Meeting transcript text")

    class Config:
        schema_extra = {
            "example": {
                "project_id": "proj-123",
                "title": "Weekly Project Sync",
                "meeting_type": "internal_team",
                "start_time": "2024-01-15T09:00:00",
                "end_time": "2024-01-15T10:00:00",
                "participants": [
                    {"name": "John Smith", "role": "Project Manager"},
                    {"name": "Jane Doe", "role": "Tech Lead"},
                ],
                "transcript": "John Smith: Let's start with status updates...",
            }
        }


class MemorySearch(BaseModel):
    """Request model for searching memories."""

    query: str = Field(..., description="Search query")
    project_id: Optional[str] = Field(None, description="Filter by project")
    meeting_id: Optional[str] = Field(None, description="Filter by meeting")
    content_types: Optional[List[str]] = Field(None, description="Filter by content types")
    limit: int = Field(10, ge=1, le=100, description="Maximum results")
    min_score: Optional[float] = Field(None, ge=0, le=1, description="Minimum similarity score")

    class Config:
        schema_extra = {
            "example": {
                "query": "What are the main project risks?",
                "project_id": "proj-123",
                "content_types": ["risk", "issue"],
                "limit": 20,
            }
        }


class SearchResult(BaseModel):
    """Response model for search results."""

    memory_id: str
    content: str
    speaker: Optional[str]
    meeting_id: str
    meeting_title: str
    content_type: str
    score: float
    created_at: datetime

    class Config:
        schema_extra = {
            "example": {
                "memory_id": "mem-123",
                "content": "The main risk is the tight timeline for API integration",
                "speaker": "John Smith",
                "meeting_id": "meet-456",
                "meeting_title": "Risk Assessment Meeting",
                "content_type": "risk",
                "score": 0.92,
                "created_at": "2024-01-15T10:30:00",
            }
        }


class IngestResponse(BaseModel):
    """Response model for ingestion results."""

    meeting_id: str
    memories_extracted: int
    memories_stored: int
    connections_created: int
    processing_time_ms: float
    status: str
    errors: List[str]
    warnings: List[str]


# Endpoints


@router.post("/projects", status_code=201)
async def create_project(project: ProjectCreate, db=Depends(get_db_connection)) -> Dict[str, Any]:
    """
    Create a new project.

    Args:
        project: Project creation data

    Returns:
        Created project details
    """
    try:
        # Create project entity
        project_entity = Project(
            name=project.name,
            client_name=project.client_name,
            project_type=ProjectType(project.project_type),
            project_manager=project.project_manager,
            engagement_code=project.engagement_code,
            status=ProjectStatus.ACTIVE,
            start_date=datetime.now(),
        )

        # Save to database
        project_repo = get_project_repository(db)
        project_id = await project_repo.create(project_entity)

        logger.info(f"Created project {project_id}: {project.name}")

        return {
            "project_id": project_id,
            "name": project.name,
            "client_name": project.client_name,
            "status": "active",
            "created_at": project_entity.created_at,
        }

    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_meeting(
    meeting_data: MeetingIngest,
    db=Depends(get_db_connection),
    vector_store=Depends(get_vector_store_instance),
) -> IngestResponse:
    """
    Ingest a meeting transcript.

    This endpoint processes a meeting transcript through the full pipeline:
    1. Extract memories
    2. Generate embeddings
    3. Extract dimensions
    4. Store in database and vector store
    5. Create connections

    Args:
        meeting_data: Meeting information and transcript

    Returns:
        Ingestion results with statistics
    """
    try:
        # Validate project exists
        project_repo = get_project_repository(db)
        project = await project_repo.get_by_id(meeting_data.project_id)
        if not project:
            raise HTTPException(
                status_code=404, detail=f"Project {meeting_data.project_id} not found"
            )

        # Create meeting entity
        meeting = Meeting(
            project_id=meeting_data.project_id,
            title=meeting_data.title,
            meeting_type=MeetingType(meeting_data.meeting_type),
            meeting_category=(
                MeetingCategory.EXTERNAL
                if meeting_data.meeting_type.startswith("client_")
                else MeetingCategory.INTERNAL
            ),
            start_time=meeting_data.start_time,
            end_time=meeting_data.end_time,
            participants=meeting_data.participants,
            transcript_path="inline",  # Transcript provided inline
        )

        # Save meeting
        meeting_repo = get_meeting_repository(db)
        meeting_id = await meeting_repo.create(meeting)
        meeting.id = meeting_id

        # Create and run pipeline
        pipeline_config = PipelineConfig()
        pipeline = await create_ingestion_pipeline(
            db, vector_store.client.host, vector_store.client.port, pipeline_config
        )

        # Ingest the meeting
        result = await pipeline.ingest_meeting(meeting, meeting_data.transcript)

        # Determine status
        if result.errors:
            status = "failed"
        elif result.warnings:
            status = "completed_with_warnings"
        else:
            status = "success"

        logger.info(
            f"Ingested meeting {meeting_id}: "
            f"{result.memories_extracted} extracted, "
            f"{result.memories_stored} stored, "
            f"{result.processing_time_ms:.0f}ms"
        )

        return IngestResponse(
            meeting_id=result.meeting_id,
            memories_extracted=result.memories_extracted,
            memories_stored=result.memories_stored,
            connections_created=result.connections_created,
            processing_time_ms=result.processing_time_ms,
            status=status,
            errors=result.errors,
            warnings=result.warnings,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/search", response_model=List[SearchResult])
async def search_memories(
    search_request: MemorySearch,
    db=Depends(get_db_connection),
    vector_store=Depends(get_vector_store_instance),
) -> List[SearchResult]:
    """
    Search memories using vector similarity.

    This endpoint performs semantic search across memories using
    the embedded vectors and optional filters.

    Args:
        search_request: Search parameters

    Returns:
        List of matching memories with similarity scores
    """
    try:
        # Generate query embedding
        encoder = get_encoder()
        query_embedding = encoder.encode(search_request.query, normalize=True)

        # Extract cognitive dimensions from the search query itself
        from ...extraction.dimensions.dimension_analyzer import get_dimension_analyzer, DimensionExtractionContext
        dimension_analyzer = get_dimension_analyzer()
        # Create a minimal context for the query (e.g., content_type="query")
        query_dim_context = DimensionExtractionContext(content_type="query")
        query_cognitive_dimensions = await dimension_analyzer.analyze(search_request.query, query_dim_context)

        # Compose query vector
        vector_manager = get_vector_manager()
        # Use the extracted cognitive dimensions for the query vector
        query_vector_obj = vector_manager.compose_vector(query_embedding, query_cognitive_dimensions)
        query_vector = query_vector_obj.full_vector

        # Create search filter
        search_filter = SearchFilter(
            project_id=search_request.project_id,
            meeting_id=search_request.meeting_id,
            content_type=search_request.content_types[0] if search_request.content_types else None,
        )

        # Search across all levels
        all_results = await vector_store.search_all_levels(
            query_vector=query_vector,
            limit_per_level=search_request.limit,
            filters=search_filter,
            score_threshold=search_request.min_score,
        )

        # Combine results from all levels
        combined_results = []
        for level_results in all_results.values():
            combined_results.extend(level_results)

        # Sort by score
        combined_results.sort(key=lambda x: x.score, reverse=True)

        # Limit results
        combined_results = combined_results[: search_request.limit]

        # Get memory details from database
        memory_repo = get_memory_repository(db)
        meeting_repo = get_meeting_repository(db)

        search_results = []
        for result in combined_results:
            # Get memory details
            memory = await memory_repo.get_by_id(result.payload.get("memory_id"))
            if not memory:
                continue

            # Get meeting details
            meeting = await meeting_repo.get_by_id(memory.meeting_id)

            search_results.append(
                SearchResult(
                    memory_id=memory.id,
                    content=memory.content,
                    speaker=memory.speaker,
                    meeting_id=memory.meeting_id,
                    meeting_title=meeting.title if meeting else "Unknown",
                    content_type=memory.content_type.value,
                    score=result.score,
                    created_at=memory.created_at,
                )
            )

        logger.info(f"Search for '{search_request.query}' returned {len(search_results)} results")

        return search_results

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/memories/{memory_id}")
async def get_memory(memory_id: str, db=Depends(get_db_connection)) -> Dict[str, Any]:
    """
    Get a specific memory by ID.

    Args:
        memory_id: Memory ID

    Returns:
        Memory details
    """
    try:
        memory_repo = get_memory_repository(db)
        memory = await memory_repo.get_by_id(memory_id)

        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

        # Update access tracking
        await memory_repo.update_access_tracking(memory_id)

        return memory.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory: {str(e)}")


@router.get("/projects/{project_id}/memories")
async def get_project_memories(
    project_id: str,
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db=Depends(get_db_connection),
) -> Dict[str, Any]:
    """
    Get memories for a project.

    Args:
        project_id: Project ID
        content_type: Optional content type filter
        limit: Maximum results
        offset: Pagination offset

    Returns:
        List of memories with pagination info
    """
    try:
        memory_repo = get_memory_repository(db)

        # Get memories
        if content_type:
            memories = await memory_repo.get_by_content_type(
                ContentType(content_type), project_id=project_id
            )
            # Apply pagination manually
            total = len(memories)
            memories = memories[offset : offset + limit]
        else:
            memories = await memory_repo.get_by_project(
                project_id=project_id, limit=limit, offset=offset
            )
            # Get total count
            total = await memory_repo.count(f"project_id = '{project_id}'")

        return {
            "memories": [m.to_dict() for m in memories],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.error(f"Failed to get project memories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get project memories: {str(e)}")


@router.post("/upload-transcript")
async def upload_transcript(
    file: UploadFile = File(..., description="Transcript file"),
    project_id: str = Form(..., description="Project ID"),
    meeting_title: str = Form(..., description="Meeting title"),
    meeting_type: str = Form("working_session", description="Meeting type"),
    meeting_date: datetime = Form(..., description="Meeting date"),
    duration_minutes: int = Form(..., description="Meeting duration in minutes"),
    db=Depends(get_db_connection),
    vector_store=Depends(get_vector_store_instance),
) -> IngestResponse:
    """
    Upload and process a transcript file.

    Accepts text files containing meeting transcripts and processes
    them through the ingestion pipeline.

    Args:
        file: Transcript file (txt format)
        project_id: Project ID
        meeting_title: Meeting title
        meeting_type: Type of meeting
        meeting_date: When the meeting occurred
        duration_minutes: Meeting duration

    Returns:
        Ingestion results
    """
    try:
        # Validate file type
        if not file.filename.endswith(".txt"):
            raise HTTPException(status_code=400, detail="Only .txt files are supported")

        # Read transcript
        content = await file.read()
        transcript = content.decode("utf-8")

        # Create meeting data
        end_time = meeting_date.replace(minute=meeting_date.minute + duration_minutes)

        meeting_data = MeetingIngest(
            project_id=project_id,
            title=meeting_title,
            meeting_type=meeting_type,
            start_time=meeting_date,
            end_time=end_time,
            participants=[],  # Would need to extract from transcript
            transcript=transcript,
        )

        # Process through regular ingestion
        return await ingest_meeting(meeting_data, db, vector_store)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/stats/project/{project_id}")
async def get_project_statistics(project_id: str, db=Depends(get_db_connection)) -> Dict[str, Any]:
    """
    Get statistics for a project.

    Args:
        project_id: Project ID

    Returns:
        Project statistics including memory counts, meeting info, etc.
    """
    try:
        memory_repo = get_memory_repository(db)
        meeting_repo = get_meeting_repository(db)

        # Get memory statistics
        memory_stats = await memory_repo.get_memory_statistics(project_id)

        # Get meeting statistics
        meeting_stats = await meeting_repo.get_meeting_statistics(project_id)

        return {
            "project_id": project_id,
            "memories": memory_stats,
            "meetings": meeting_stats,
            "generated_at": datetime.now(),
        }

    except Exception as e:
        logger.error(f"Failed to get project statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")
