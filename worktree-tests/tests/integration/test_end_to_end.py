"""
End-to-end integration tests for the complete pipeline.

These tests verify that the entire system works together correctly,
from transcript ingestion to memory search.
"""

import pytest
import asyncio
from datetime import datetime
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.entities import (
    Project, Meeting, ProjectType, ProjectStatus,
    MeetingType, MeetingCategory
)
from src.storage.sqlite.connection import DatabaseConnection
from src.storage.qdrant.vector_store import QdrantVectorStore
from src.pipeline.ingestion_pipeline import create_ingestion_pipeline
from src.storage.sqlite.repositories import (
    get_project_repository, get_meeting_repository,
    get_memory_repository
)


# Test data
SAMPLE_TRANSCRIPT = """
John Smith: Good morning everyone. Let's start with our project status update.

Sarah Johnson: Thanks John. I'm excited to report that we've completed the API integration ahead of schedule. This is a major milestone for the project.

John Smith: That's excellent news! This will definitely help us stay on track for the December deadline.

Mike Chen: I have a concern though. Our performance tests show that the database queries are taking longer than expected. We might have a bottleneck there.

Sarah Johnson: That's a valid concern. I suggest we schedule a technical review session this week to deep dive into the performance issues.

John Smith: Agreed. Mike, can you lead the performance review and provide recommendations by Friday?

Mike Chen: I'll take care of it. I'll analyze the query patterns and propose optimization strategies.

John Smith: Perfect. One more critical decision - we need to decide on our deployment strategy for next month's release.

Sarah Johnson: Given the risks we've identified, I strongly recommend a phased rollout approach. We can start with 10% of users and gradually increase.

John Smith: That makes sense. Let's go with the phased rollout. This is our final decision on the deployment strategy.

Mike Chen: I'll update the deployment plan accordingly. Also, I found an interesting insight from our user analytics - engagement peaks on Tuesdays and Thursdays.

Sarah Johnson: That's valuable information! We should schedule our rollout phases to align with these high-engagement days.

John Smith: Excellent point. Let's make sure that's reflected in our plan. Any other critical items?

Sarah Johnson: Just a reminder that we have a dependency on the infrastructure team for the load balancers. We need their confirmation by next week.

John Smith: I'll follow up with them today. Thanks everyone for the productive discussion!
"""


@pytest.fixture
async def db_connection():
    """Create test database connection."""
    db = DatabaseConnection(db_path="data/test_cognitive.db")
    await db.execute_schema()
    yield db
    await db.close()


@pytest.fixture
async def vector_store():
    """Create test vector store connection."""
    store = QdrantVectorStore(host="localhost", port=6333)
    yield store
    await store.close()


@pytest.fixture
async def test_project(db_connection):
    """Create a test project."""
    project_repo = get_project_repository(db_connection)
    
    project = Project(
        name="Test Digital Transformation",
        client_name="Test Corp",
        project_type=ProjectType.TRANSFORMATION,
        status=ProjectStatus.ACTIVE,
        project_manager="Test Manager",
        start_date=datetime.now()
    )
    
    project_id = await project_repo.create(project)
    project.id = project_id
    
    return project


@pytest.fixture
async def test_meeting(db_connection, test_project):
    """Create a test meeting."""
    meeting_repo = get_meeting_repository(db_connection)
    
    meeting = Meeting(
        project_id=test_project.id,
        title="Test Status Update Meeting",
        meeting_type=MeetingType.INTERNAL_TEAM,
        meeting_category=MeetingCategory.INTERNAL,
        start_time=datetime.now(),
        end_time=datetime.now(),
        participants=[
            {"name": "John Smith", "role": "Project Manager"},
            {"name": "Sarah Johnson", "role": "Tech Lead"},
            {"name": "Mike Chen", "role": "Senior Developer"}
        ],
        transcript_path="inline"
    )
    
    meeting_id = await meeting_repo.create(meeting)
    meeting.id = meeting_id
    
    return meeting


class TestEndToEndPipeline:
    """Test the complete pipeline from ingestion to search."""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline(self, db_connection, vector_store, test_project, test_meeting):
        """Test the entire pipeline end-to-end."""
        # Create pipeline
        pipeline = await create_ingestion_pipeline(
            db_connection,
            qdrant_host="localhost",
            qdrant_port=6333
        )
        
        # Ingest the meeting
        result = await pipeline.ingest_meeting(test_meeting, SAMPLE_TRANSCRIPT)
        
        # Verify ingestion results
        assert result.meeting_id == test_meeting.id
        assert result.memories_extracted > 10  # Should extract multiple memories
        assert result.memories_stored == result.memories_extracted
        assert result.connections_created > 0  # Sequential connections
        assert result.processing_time_ms < 2000  # Should be under 2 seconds
        assert len(result.errors) == 0
        
        # Verify memories were stored
        memory_repo = get_memory_repository(db_connection)
        memories = await memory_repo.get_by_meeting(test_meeting.id)
        
        assert len(memories) == result.memories_stored
        
        # Check memory content types
        content_types = {m.content_type.value for m in memories}
        assert "decision" in content_types
        assert "action" in content_types
        assert "insight" in content_types
        assert "risk" in content_types
        
        # Check speakers were identified
        speakers = {m.speaker for m in memories if m.speaker}
        assert "John Smith" in speakers
        assert "Sarah Johnson" in speakers
        assert "Mike Chen" in speakers
        
        # Verify vectors were stored
        for memory in memories[:5]:  # Check first 5
            assert memory.qdrant_id is not None
            
            # Retrieve vector from Qdrant
            vector_data = await vector_store.get_by_id(memory.qdrant_id, memory.level)
            assert vector_data is not None
            
            vector, payload = vector_data
            assert vector.full_vector.shape == (400,)
            assert payload["memory_id"] == memory.id
    
    @pytest.mark.asyncio
    async def test_memory_search(self, db_connection, vector_store, test_project, test_meeting):
        """Test searching for memories after ingestion."""
        # First ingest the meeting
        pipeline = await create_ingestion_pipeline(
            db_connection,
            qdrant_host="localhost",
            qdrant_port=6333
        )
        
        await pipeline.ingest_meeting(test_meeting, SAMPLE_TRANSCRIPT)
        
        # Now test search
        from src.embedding.onnx_encoder import get_encoder
        from src.embedding.vector_manager import get_vector_manager
        from src.storage.qdrant.vector_store import SearchFilter
        
        encoder = get_encoder()
        vector_manager = get_vector_manager()
        
        # Search for performance-related memories
        query = "performance issues and optimization"
        query_embedding = encoder.encode(query, normalize=True)
        query_vector = vector_manager.compose_vector(
            query_embedding,
            np.full(16, 0.5)  # Default dimensions
        )
        
        # Search with project filter
        search_filter = SearchFilter(project_id=test_project.id)
        
        results = await vector_store.search(
            query_vector=query_vector,
            level=2,  # L2 episodic memories
            limit=5,
            filters=search_filter
        )
        
        assert len(results) > 0
        
        # Verify we found relevant memories
        memory_repo = get_memory_repository(db_connection)
        found_performance_content = False
        
        for result in results:
            memory = await memory_repo.get_by_id(result.payload["memory_id"])
            if "performance" in memory.content.lower() or "optimization" in memory.content.lower():
                found_performance_content = True
                break
        
        assert found_performance_content
    
    @pytest.mark.asyncio
    async def test_dimension_extraction(self, db_connection, vector_store, test_project, test_meeting):
        """Test that cognitive dimensions are properly extracted."""
        pipeline = await create_ingestion_pipeline(
            db_connection,
            qdrant_host="localhost",
            qdrant_port=6333
        )
        
        await pipeline.ingest_meeting(test_meeting, SAMPLE_TRANSCRIPT)
        
        # Get memories and check dimensions
        memory_repo = get_memory_repository(db_connection)
        memories = await memory_repo.get_by_meeting(test_meeting.id)
        
        # Find specific memories to check dimensions
        for memory in memories:
            if "urgent" in memory.content.lower() or "critical" in memory.content.lower():
                # This should have high urgency
                vector_json = memory.dimensions_json
                assert vector_json is not None
                
                from src.embedding.vector_manager import get_vector_manager
                vector_manager = get_vector_manager()
                vector = vector_manager.from_json(vector_json)
                
                # Check urgency dimension (index 0)
                urgency = vector.dimensions[0]
                assert urgency > 0.7  # Should be high
            
            if "excellent news" in memory.content.lower() or "excited" in memory.content.lower():
                # This should have positive polarity
                vector_json = memory.dimensions_json
                assert vector_json is not None
                
                from src.embedding.vector_manager import get_vector_manager
                vector_manager = get_vector_manager()
                vector = vector_manager.from_json(vector_json)
                
                # Check polarity dimension (index 4)
                polarity = vector.dimensions[4]
                assert polarity > 0.6  # Should be positive
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, db_connection, vector_store, test_project, test_meeting):
        """Test that performance requirements are met."""
        pipeline = await create_ingestion_pipeline(
            db_connection,
            qdrant_host="localhost",
            qdrant_port=6333
        )
        
        # Measure ingestion time
        import time
        start_time = time.time()
        
        result = await pipeline.ingest_meeting(test_meeting, SAMPLE_TRANSCRIPT)
        
        total_time = (time.time() - start_time) * 1000
        
        # Verify performance requirements
        assert total_time < 2000  # Should complete in under 2 seconds
        assert result.processing_time_ms < 2000
        
        # Check stage times if available
        stats = pipeline.get_performance_stats()
        if stats['meetings_processed'] > 0:
            assert stats['avg_processing_time_ms'] < 2000
            
            # Check individual stages
            stages = stats.get('stages', {})
            if 'embedding_avg_ms' in stages:
                # Embedding should be fast
                assert stages['embedding_avg_ms'] < 500
            
            if 'extraction_avg_ms' in stages:
                # Extraction should be reasonably fast
                assert stages['extraction_avg_ms'] < 1000
    
    @pytest.mark.asyncio
    async def test_connection_creation(self, db_connection, vector_store, test_project, test_meeting):
        """Test that memory connections are properly created."""
        pipeline = await create_ingestion_pipeline(
            db_connection,
            qdrant_host="localhost",
            qdrant_port=6333
        )
        
        result = await pipeline.ingest_meeting(test_meeting, SAMPLE_TRANSCRIPT)
        
        # Verify connections were created
        assert result.connections_created > 0
        
        # Check connections in database
        from src.storage.sqlite.repositories import get_memory_connection_repository
        
        memory_repo = get_memory_repository(db_connection)
        connection_repo = get_memory_connection_repository(db_connection)
        
        memories = await memory_repo.get_by_meeting(test_meeting.id)
        
        # Check sequential connections
        for i in range(len(memories) - 1):
            connections = await connection_repo.get_outgoing_connections(memories[i].id)
            
            # Should have at least one connection (sequential to next)
            assert len(connections) > 0
            
            # Check if sequential connection exists
            sequential_found = False
            for conn in connections:
                if conn.target_id == memories[i + 1].id:
                    sequential_found = True
                    assert conn.connection_strength >= 0.7  # Default sequential strength
                    break
            
            assert sequential_found


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
