"""
Comprehensive integration tests for the cognitive pipeline.

Tests the full flow from transcript ingestion through cognitive retrieval,
including all intermediate steps and component interactions.
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from src.models.entities import Memory, MemoryType, ContentType
from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.cognitive.retrieval.contextual_retrieval import ContextualRetrieval
from src.cognitive.activation.basic_activation_engine import BasicActivationEngine
from src.cognitive.retrieval.bridge_discovery import SimpleBridgeDiscovery


@pytest.mark.integration
class TestCognitivePipelineIntegration:
    """Test the complete cognitive intelligence pipeline."""
    
    @pytest.fixture
    async def full_pipeline(self, db_repositories, mock_encoder, mock_vector_store):
        """Create a complete pipeline with all components."""
        from src.extraction.memory_extractor import MemoryExtractor
        from src.extraction.dimensions.dimension_analyzer import get_dimension_analyzer
        from src.embedding.vector_manager import VectorManager
        
        # Create components
        memory_extractor = MemoryExtractor()
        dimension_analyzer = get_dimension_analyzer()
        vector_manager = VectorManager()
        
        # Create pipeline
        pipeline = IngestionPipeline(
            memory_extractor=memory_extractor,
            dimension_analyzer=dimension_analyzer,
            encoder=mock_encoder,
            vector_manager=vector_manager,
            memory_repo=db_repositories['memory'],
            vector_store=mock_vector_store
        )
        
        # Create cognitive components
        activation_engine = BasicActivationEngine(
            memory_repo=db_repositories['memory'],
            connection_repo=db_repositories['connection'],
            vector_store=mock_vector_store
        )
        
        bridge_discovery = SimpleBridgeDiscovery(
            memory_repo=db_repositories['memory'],
            vector_store=mock_vector_store
        )
        
        contextual_retrieval = ContextualRetrieval(
            activation_engine=activation_engine,
            bridge_discovery=bridge_discovery,
            similarity_search=mock_vector_store
        )
        
        return {
            'pipeline': pipeline,
            'retrieval': contextual_retrieval,
            'repos': db_repositories
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, full_pipeline):
        """Test complete flow from transcript to cognitive query."""
        # Step 1: Create project and meeting
        project_id = await full_pipeline['repos']['project'].create(
            name="Test Project",
            description="Integration test project"
        )
        
        meeting_id = await full_pipeline['repos']['meeting'].create(
            project_id=project_id,
            title="Sprint Planning Meeting",
            start_time=datetime.now(),
            participants=["Sarah", "Tom", "Emily"]
        )
        
        # Step 2: Ingest transcript
        transcript = """
        Sarah: Welcome to sprint planning. Our velocity last sprint was 45 points.
        
        Tom: We have a critical performance issue in the search API. It's affecting customers.
        
        Sarah: That's urgent. Let's make it our top priority.
        
        Emily: I can fix that today. It's a caching issue.
        
        Tom: Also, the vendor API is still slow. It's been a problem for weeks.
        
        Sarah: We need to escalate that to management. It's becoming a major risk.
        
        Emily: Should we consider switching vendors?
        
        Sarah: Let's create a contingency plan. This dependency is blocking us.
        """
        
        result = await full_pipeline['pipeline'].ingest(
            transcript=transcript,
            meeting_id=meeting_id,
            metadata={"project_id": project_id}
        )
        
        # Verify ingestion
        assert result.memories_extracted > 5
        assert result.vectors_stored == result.memories_extracted
        assert result.processing_time_ms > 0
        
        # Step 3: Verify memories were created with all fields
        memories = await full_pipeline['repos']['memory'].find_by_meeting(meeting_id)
        assert len(memories) > 5
        
        for memory in memories:
            # Check required fields
            assert memory.id
            assert memory.content
            assert memory.meeting_id == meeting_id
            assert memory.project_id == project_id
            assert memory.speaker in ["Sarah", "Tom", "Emily"]
            assert memory.timestamp_ms >= 0
            
            # Check cognitive fields
            assert memory.dimensions_json is not None
            assert memory.qdrant_id is not None
            
            # Verify dimensions
            import json
            dims = json.loads(memory.dimensions_json)
            assert 'temporal' in dims
            assert 'emotional' in dims
            assert 'social' in dims
            assert 'causal' in dims
            assert 'evolutionary' in dims
        
        # Step 4: Create memory connections
        conn_repo = full_pipeline['repos']['connection']
        
        # Find specific memories to connect
        perf_memory = next(m for m in memories if "performance issue" in m.content)
        vendor_memory = next(m for m in memories if "vendor API" in m.content)
        risk_memory = next(m for m in memories if "major risk" in m.content)
        
        # Create connections
        await conn_repo.create(
            source_id=vendor_memory.id,
            target_id=risk_memory.id,
            connection_strength=0.9,
            connection_type="causal"
        )
        
        await conn_repo.create(
            source_id=perf_memory.id,
            target_id=vendor_memory.id,
            connection_strength=0.6,
            connection_type="thematic"
        )
        
        # Step 5: Test cognitive retrieval
        # Mock vector store search to return relevant memories
        full_pipeline['retrieval'].similarity_search.search = AsyncMock(
            return_value=[
                Mock(id=perf_memory.qdrant_id, score=0.9),
                Mock(id=vendor_memory.qdrant_id, score=0.85),
            ]
        )
        
        query = "What are the main technical risks we discussed?"
        
        retrieval_result = await full_pipeline['retrieval'].retrieve(
            query=query,
            enable_activation=True,
            enable_bridges=True,
            max_results=10
        )
        
        # Verify retrieval results
        assert len(retrieval_result.memories) >= 2
        assert retrieval_result.query == query
        assert retrieval_result.processing_time_ms > 0
        
        # Check that high-urgency memories are included
        urgent_contents = [m.content for m in retrieval_result.memories 
                          if "urgent" in m.content.lower() or "critical" in m.content.lower()]
        assert len(urgent_contents) > 0
    
    @pytest.mark.asyncio
    async def test_memory_lifecycle(self, full_pipeline):
        """Test memory creation, update, and retrieval lifecycle."""
        # Create test data
        project_id = await full_pipeline['repos']['project'].create(
            name="Lifecycle Test"
        )
        
        meeting_id = await full_pipeline['repos']['meeting'].create(
            project_id=project_id,
            title="Status Update"
        )
        
        # Ingest initial transcript
        transcript_v1 = """
        Alice: The new feature is 50% complete.
        Bob: We might miss the deadline next week.
        """
        
        result1 = await full_pipeline['pipeline'].ingest(
            transcript=transcript_v1,
            meeting_id=meeting_id
        )
        
        memories_v1 = await full_pipeline['repos']['memory'].find_by_meeting(meeting_id)
        assert len(memories_v1) >= 2
        
        # Check memory importance scores
        deadline_memory = next(m for m in memories_v1 if "deadline" in m.content)
        assert deadline_memory.importance_score > 0.7  # Should be high due to urgency
        
        # Update importance through access
        await full_pipeline['repos']['memory'].increment_access_count(deadline_memory.id)
        
        updated_memory = await full_pipeline['repos']['memory'].get(deadline_memory.id)
        assert updated_memory.access_count == 1
        assert updated_memory.last_accessed is not None
    
    @pytest.mark.asyncio  
    async def test_cross_meeting_connections(self, full_pipeline):
        """Test connections and retrieval across multiple meetings."""
        project_id = await full_pipeline['repos']['project'].create(
            name="Multi-Meeting Test"
        )
        
        # Meeting 1: Identify problem
        meeting1_id = await full_pipeline['repos']['meeting'].create(
            project_id=project_id,
            title="Problem Identification"
        )
        
        await full_pipeline['pipeline'].ingest(
            transcript="Sarah: We have serious performance issues with the database.",
            meeting_id=meeting1_id
        )
        
        # Meeting 2: Discuss solution
        meeting2_id = await full_pipeline['repos']['meeting'].create(
            project_id=project_id,
            title="Solution Planning"
        )
        
        await full_pipeline['pipeline'].ingest(
            transcript="Tom: To fix the performance issues, we should implement caching.",
            meeting_id=meeting2_id
        )
        
        # Meeting 3: Implementation update
        meeting3_id = await full_pipeline['repos']['meeting'].create(
            project_id=project_id,
            title="Implementation Update"
        )
        
        await full_pipeline['pipeline'].ingest(
            transcript="Emily: The caching solution improved performance by 80%.",
            meeting_id=meeting3_id
        )
        
        # Get all memories
        all_memories = []
        for meeting_id in [meeting1_id, meeting2_id, meeting3_id]:
            memories = await full_pipeline['repos']['memory'].find_by_meeting(meeting_id)
            all_memories.extend(memories)
        
        # Create cross-meeting connections
        problem_memory = next(m for m in all_memories if "performance issues" in m.content)
        solution_memory = next(m for m in all_memories if "implement caching" in m.content)
        result_memory = next(m for m in all_memories if "improved performance" in m.content)
        
        conn_repo = full_pipeline['repos']['connection']
        await conn_repo.create(
            source_id=problem_memory.id,
            target_id=solution_memory.id,
            connection_strength=0.85,
            connection_type="problem_solution"
        )
        
        await conn_repo.create(
            source_id=solution_memory.id,
            target_id=result_memory.id,
            connection_strength=0.9,
            connection_type="implementation_result"
        )
        
        # Test retrieval across meetings
        # This would use activation spreading to find the full story
        connections = await conn_repo.get_connections(problem_memory.id)
        assert len(connections) == 1
        assert connections[0].target_id == solution_memory.id
    
    @pytest.mark.asyncio
    async def test_dimension_impact_on_retrieval(self, full_pipeline):
        """Test how different dimensions affect retrieval priority."""
        project_id = await full_pipeline['repos']['project'].create(
            name="Dimension Test"
        )
        
        meeting_id = await full_pipeline['repos']['meeting'].create(
            project_id=project_id,
            title="Mixed Priority Meeting"
        )
        
        # Ingest transcript with varied urgency/importance
        transcript = """
        Manager: URGENT: System is down! We need all hands on deck immediately!
        
        Developer: I found an interesting optimization that could help next quarter.
        
        Manager: CRITICAL: Customer data might be affected. This is our top priority!
        
        Developer: Also, we should update our documentation when we have time.
        
        Support: Multiple customers are complaining. Revenue impact is significant!
        """
        
        result = await full_pipeline['pipeline'].ingest(
            transcript=transcript,
            meeting_id=meeting_id
        )
        
        memories = await full_pipeline['repos']['memory'].find_by_meeting(meeting_id)
        
        # Verify dimension-based importance
        import json
        
        # High urgency memories should have high importance
        for memory in memories:
            dims = json.loads(memory.dimensions_json)
            urgency = dims['temporal']['urgency']
            
            if urgency > 0.8:
                assert memory.importance_score > 0.8, (
                    f"High urgency memory should have high importance: {memory.content}"
                )
            elif urgency < 0.3:
                assert memory.importance_score < 0.6, (
                    f"Low urgency memory should have lower importance: {memory.content}"
                )
        
        # Test that urgent memories are retrieved first
        urgent_memories = sorted(
            [m for m in memories if "URGENT" in m.content or "CRITICAL" in m.content],
            key=lambda m: m.importance_score,
            reverse=True
        )
        
        assert len(urgent_memories) >= 2
        assert all(m.importance_score > 0.8 for m in urgent_memories)
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, full_pipeline):
        """Test pipeline error handling and recovery."""
        project_id = await full_pipeline['repos']['project'].create(
            name="Error Test"
        )
        
        meeting_id = await full_pipeline['repos']['meeting'].create(
            project_id=project_id,
            title="Error Test Meeting"
        )
        
        # Test with problematic inputs
        problematic_transcripts = [
            "",  # Empty
            "NoSpeaker",  # No speaker attribution
            "Speaker: " + "word " * 1000,  # Very long
            "Alice: Hello\n" * 100,  # Very repetitive
            "Bob: 😀🎉🚀" * 10,  # Emojis
        ]
        
        for transcript in problematic_transcripts:
            # Should not crash
            result = await full_pipeline['pipeline'].ingest(
                transcript=transcript,
                meeting_id=meeting_id
            )
            
            # Should handle gracefully
            assert result is not None
            assert result.processing_time_ms >= 0
            assert result.errors is None or len(result.errors) == 0
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_pipeline_performance(self, full_pipeline, benchmark_timer):
        """Test that pipeline meets performance requirements."""
        project_id = await full_pipeline['repos']['project'].create(
            name="Performance Test"
        )
        
        meeting_id = await full_pipeline['repos']['meeting'].create(
            project_id=project_id,
            title="Performance Test Meeting"
        )
        
        # Create a substantial transcript (~500 words)
        speakers = ["Alice", "Bob", "Charlie"]
        statements = [
            "We need to discuss the quarterly results.",
            "The performance metrics show improvement.",
            "Customer satisfaction is up by 15%.",
            "However, costs have increased significantly.",
            "We should focus on optimization.",
            "The new feature rollout was successful.",
            "Bug reports decreased by 30% this month.",
            "Team velocity is consistent at 45 points.",
            "We have three critical issues to address.",
            "The infrastructure upgrade is on schedule.",
        ]
        
        transcript_lines = []
        for i in range(50):  # 50 statements = ~500 words
            speaker = speakers[i % len(speakers)]
            statement = statements[i % len(statements)]
            transcript_lines.append(f"{speaker}: {statement}")
        
        large_transcript = "\n\n".join(transcript_lines)
        
        # Warm up
        await full_pipeline['pipeline'].ingest(
            transcript="Warm up",
            meeting_id=meeting_id
        )
        
        # Measure ingestion performance
        with benchmark_timer:
            result = await full_pipeline['pipeline'].ingest(
                transcript=large_transcript,
                meeting_id=meeting_id
            )
        
        # Check performance
        ingestion_time = benchmark_timer.last
        memories_per_second = result.memories_extracted / ingestion_time
        
        print(f"\nPerformance Results:")
        print(f"  Memories extracted: {result.memories_extracted}")
        print(f"  Ingestion time: {ingestion_time:.2f}s")
        print(f"  Rate: {memories_per_second:.1f} memories/second")
        
        # Should meet 10-15 memories/second target
        assert memories_per_second >= 10, (
            f"Ingestion too slow: {memories_per_second:.1f} memories/second"
        )
        
        # End-to-end should be reasonable
        assert result.processing_time_ms < 5000, (
            f"Total processing too slow: {result.processing_time_ms}ms"
        )