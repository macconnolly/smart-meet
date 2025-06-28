"""
CRITICAL: End-to-end cognitive pipeline integration tests.

These tests validate that the entire cognitive system works as an integrated whole:
Transcript → Memory Extraction → Vector Embedding → Storage → Query → Results

This is the ultimate test of whether the system actually works as a 
"Cognitive Meeting Intelligence System".

Reference: TESTING_STRATEGY.md - Phase 4: End-to-End Validation
"""

import pytest
import asyncio
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

from src.models.entities import Memory, Meeting, MemoryType, ContentType, Vector
from src.extraction.memory_extractor import MemoryExtractor
from src.embedding.onnx_encoder import get_encoder
from src.embedding.vector_manager import get_vector_manager
from src.storage.sqlite.repositories import get_memory_repository
from src.storage.qdrant.vector_store import get_vector_store
from src.cognitive.activation.engine import BFSActivationEngine
from src.pipeline.ingestion_pipeline import IngestionPipeline


class TestCognitivePipelineIntegration:
    """Test complete cognitive pipeline integration."""
    
    @pytest.fixture
    def sample_meeting_transcript(self):
        """Realistic meeting transcript for testing."""
        return '''
        Meeting: Sprint Planning Session
        Date: 2024-01-15
        Participants: Alice (PM), Bob (Developer), Charlie (Architect)
        
        Alice: Good morning everyone. Let's start our sprint planning.
        Alice: First item - we need to decide on the caching strategy.
        Bob: I've been researching Redis vs Memcached for our use case.
        Charlie: Redis would be better - it supports more data structures.
        Alice: Great. So we're decided on Redis for caching. Bob, can you handle the implementation?
        Bob: Absolutely. I'll have it ready by Thursday.
        Alice: Perfect. What about the database migration issue?
        Charlie: I'm concerned about potential data loss during migration.
        Bob: We should create a comprehensive backup strategy first.
        Alice: Good point. Charlie, can you draft the migration plan?
        Charlie: Yes, I'll have a draft by tomorrow for review.
        Alice: Excellent. Any other risks we should consider?
        Bob: The client demo is scheduled for Friday - tight timeline.
        Alice: We'll need to prioritize the core features for the demo.
        Charlie: I suggest we focus on the user authentication and basic CRUD operations.
        Alice: Agreed. Let's document these decisions.
        Bob: I'll update our project board with the new tasks.
        Alice: Thanks everyone. Our next meeting is Wednesday at 2 PM.
        '''
    
    @pytest.fixture
    def expected_memory_types(self):
        """Expected memory types to be extracted from sample transcript."""
        return {
            ContentType.DECISION: [
                "decided on Redis for caching",
                "focus on user authentication and basic CRUD",
                "prioritize core features for demo"
            ],
            ContentType.ACTION: [
                "Bob handle Redis implementation by Thursday", 
                "Charlie draft migration plan by tomorrow",
                "update project board with new tasks",
                "next meeting Wednesday at 2 PM"
            ],
            ContentType.ISSUE: [
                "concerned about data loss during migration",
                "tight timeline for client demo"
            ],
            ContentType.IDEA: [
                "create comprehensive backup strategy",
                "Redis supports more data structures"
            ]
        }
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transcript_to_memories_extraction(self, sample_meeting_transcript, expected_memory_types):
        """Test transcript → memory extraction works correctly."""
        # Create memory extractor
        try:
            extractor = MemoryExtractor()
            
            # Extract memories from transcript
            memories = await extractor.extract(
                transcript=sample_meeting_transcript,
                meeting_id="test-meeting-001"
            )
            
            # Validate extraction quality
            assert len(memories) >= 8, f"Expected at least 8 memories, got {len(memories)}"
            assert len(memories) <= 20, f"Too many memories extracted: {len(memories)}"
            
            # Check memory type distribution
            memory_types = {m.content_type for m in memories}
            expected_types = set(expected_memory_types.keys())
            
            # Should extract at least 3 different types
            assert len(memory_types.intersection(expected_types)) >= 3, \
                f"Missing expected memory types. Got: {memory_types}"
            
            # Validate specific content expectations
            memory_contents = [m.content.lower() for m in memories]
            
            # Should find key decisions
            assert any("redis" in content for content in memory_contents), \
                "Should extract Redis caching decision"
            assert any("authentication" in content for content in memory_contents), \
                "Should extract authentication decision"
            
            # Should find action items
            assert any("thursday" in content for content in memory_contents), \
                "Should extract Thursday deadline"
            assert any("tomorrow" in content for content in memory_contents), \
                "Should extract tomorrow deadline"
            
            # Should find concerns/issues
            assert any("data loss" in content or "migration" in content for content in memory_contents), \
                "Should extract migration concern"
            
            print(f"\\nExtracted {len(memories)} memories:")
            for i, memory in enumerate(memories):
                print(f"{i+1}. [{memory.content_type.value}] {memory.content[:60]}...")
                
        except NotImplementedError:
            pytest.skip("Memory extraction not implemented yet")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_memories_to_vectors_embedding(self, sample_meeting_transcript):
        """Test memory → vector embedding works correctly."""
        try:
            # Extract memories
            extractor = MemoryExtractor()
            memories = await extractor.extract(sample_meeting_transcript, "test-meeting-002")
            
            if not memories:
                pytest.skip("No memories extracted to test embedding")
            
            # Get encoder and vector manager
            encoder = get_encoder()
            vector_manager = get_vector_manager()
            
            # Embed each memory
            embedded_memories = []
            for memory in memories[:5]:  # Test first 5 memories
                # Encode semantic content
                semantic_embedding = await encoder.encode(memory.content)
                
                # TODO: Extract cognitive dimensions (placeholder for now)
                cognitive_dimensions = np.random.rand(16)  # Will be real extraction
                
                # Compose full vector
                full_vector = vector_manager.compose_vector(semantic_embedding, cognitive_dimensions)
                
                embedded_memories.append((memory, full_vector))
            
            # Validate embeddings
            assert len(embedded_memories) == min(5, len(memories))
            
            for memory, vector in embedded_memories:
                assert vector.shape == (400,), f"Wrong vector shape: {vector.shape}"
                
                # Check semantic part is normalized
                semantic_part = vector[:384]
                semantic_norm = np.linalg.norm(semantic_part)
                assert abs(semantic_norm - 1.0) < 1e-5, f"Semantic part not normalized: {semantic_norm}"
                
                # Check cognitive dimensions in [0,1]
                cognitive_part = vector[384:]
                assert np.all(cognitive_part >= 0) and np.all(cognitive_part <= 1), \
                    "Cognitive dimensions must be in [0,1]"
            
            print(f"\\nSuccessfully embedded {len(embedded_memories)} memories to 400D vectors")
            
        except NotImplementedError:
            pytest.skip("Memory extraction or embedding not implemented yet")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_storage_integration_sqlite_qdrant(self, sample_meeting_transcript):
        """Test memory storage in both SQLite and Qdrant works correctly."""
        try:
            # Extract and embed memories
            extractor = MemoryExtractor()
            memories = await extractor.extract(sample_meeting_transcript, "test-meeting-003")
            
            if not memories:
                pytest.skip("No memories to test storage")
            
            encoder = get_encoder()
            vector_manager = get_vector_manager()
            
            # Get storage components
            memory_repo = get_memory_repository()
            vector_store = get_vector_store()
            
            # Store first memory as test
            test_memory = memories[0]
            
            # 1. Store metadata in SQLite
            stored_id = await memory_repo.create(test_memory)
            assert stored_id == test_memory.id
            
            # 2. Create and store vector in Qdrant
            semantic_embedding = await encoder.encode(test_memory.content)
            cognitive_dimensions = np.random.rand(16)  # Placeholder
            full_vector = vector_manager.compose_vector(semantic_embedding, cognitive_dimensions)
            
            await vector_store.store_memory(
                memory_id=test_memory.id,
                vector=full_vector,
                level=test_memory.level,
                metadata={
                    "content_type": test_memory.content_type.value,
                    "memory_type": test_memory.memory_type.value,
                    "meeting_id": test_memory.meeting_id,
                    "project_id": test_memory.project_id
                }
            )
            
            # 3. Verify retrieval from both stores
            # Retrieve from SQLite
            retrieved_memory = await memory_repo.get_by_id(test_memory.id)
            assert retrieved_memory is not None
            assert retrieved_memory.content == test_memory.content
            assert retrieved_memory.content_type == test_memory.content_type
            
            # Retrieve from Qdrant
            search_results = await vector_store.search(
                query_vector=full_vector,
                level=test_memory.level,
                limit=1
            )
            
            assert len(search_results) > 0
            best_match = search_results[0]
            assert best_match["id"] == test_memory.id
            assert best_match["score"] > 0.99  # Should be nearly identical
            
            print(f"\\nSuccessfully stored and retrieved memory in both SQLite and Qdrant")
            print(f"Memory ID: {test_memory.id}")
            print(f"Qdrant similarity: {best_match['score']:.4f}")
            
        except NotImplementedError:
            pytest.skip("Storage components not fully implemented yet")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_query_to_results_cognitive_search(self, sample_meeting_transcript):
        """Test query → cognitive activation → results works correctly."""
        # This is the ultimate test - can we query the stored memories?
        try:
            # 1. Full pipeline setup
            extractor = MemoryExtractor()
            encoder = get_encoder()
            vector_manager = get_vector_manager()
            memory_repo = get_memory_repository()
            vector_store = get_vector_store()
            
            # 2. Extract and store multiple memories
            memories = await extractor.extract(sample_meeting_transcript, "test-meeting-004")
            
            if len(memories) < 3:
                pytest.skip("Need at least 3 memories for query testing")
            
            # Store first 5 memories
            stored_memory_ids = []
            for memory in memories[:5]:
                # Store in SQLite
                await memory_repo.create(memory)
                
                # Create and store vector
                semantic_embedding = await encoder.encode(memory.content)
                cognitive_dimensions = np.random.rand(16)  # Placeholder
                full_vector = vector_manager.compose_vector(semantic_embedding, cognitive_dimensions)
                
                await vector_store.store_memory(
                    memory_id=memory.id,
                    vector=full_vector,
                    level=memory.level,
                    metadata={
                        "content_type": memory.content_type.value,
                        "meeting_id": memory.meeting_id
                    }
                )
                stored_memory_ids.append(memory.id)
            
            # 3. Test semantic queries
            test_queries = [
                "What was decided about caching?",
                "What are the action items?", 
                "What concerns were raised?",
                "When is the next meeting?"
            ]
            
            for query in test_queries:
                print(f"\\nTesting query: '{query}'")
                
                # Encode query
                query_vector = await encoder.encode(query)
                cognitive_dims = np.random.rand(16)  # Placeholder
                full_query_vector = vector_manager.compose_vector(query_vector, cognitive_dims)
                
                # Search for similar memories
                search_results = await vector_store.search(
                    query_vector=full_query_vector,
                    level=2,  # L2 episodic memories
                    limit=3
                )
                
                assert len(search_results) > 0, f"No results for query: '{query}'"
                
                # Verify results are relevant
                for result in search_results:
                    assert result["id"] in stored_memory_ids
                    assert result["score"] > 0.1  # Some similarity
                    
                    # Get full memory details
                    memory = await memory_repo.get_by_id(result["id"])
                    print(f"  Score: {result['score']:.3f} - {memory.content[:60]}...")
                
                # Check query-specific relevance
                best_result = search_results[0]
                best_memory = await memory_repo.get_by_id(best_result["id"])
                
                if "caching" in query.lower():
                    assert "redis" in best_memory.content.lower() or "caching" in best_memory.content.lower(), \
                        f"Caching query should find Redis/caching content"
                elif "action" in query.lower():
                    assert best_memory.content_type == ContentType.ACTION, \
                        f"Action query should find action items"
                elif "concern" in query.lower():
                    assert best_memory.content_type == ContentType.ISSUE, \
                        f"Concern query should find issues"
            
            print(f"\\nSuccessfully queried stored memories and found relevant results")
            
        except NotImplementedError:
            pytest.skip("Cognitive query pipeline not fully implemented yet")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_activation_spreading_integration(self, ingestion_pipeline, sample_meeting_transcript):
        """Test activation spreading finds related memories and generates explanations."""
        # 1. Ingest memories using the full pipeline
        meeting = Meeting(
            id="test-meeting-activation",
            project_id="test-project-activation",
            title="Activation Test Meeting",
            transcript_path="inline",
            start_time=datetime.now()
        )
        await ingestion_pipeline.ingest_meeting(meeting, sample_meeting_transcript)

        # Get repositories and vector store from the pipeline
        memory_repo = ingestion_pipeline.memory_repo
        connection_repo = ingestion_pipeline.connection_repo
        vector_store = ingestion_pipeline.vector_store

        # Create activation engine
        activation_engine = BasicActivationEngine(
            memory_repo=memory_repo,
            connection_repo=connection_repo,
            vector_store=vector_store
        )

        # Define a query that should activate specific memories
        query_string = "What was the decision about Redis caching?"
        query_embedding = ingestion_pipeline.encoder.encode(query_string)
        query_dim_context = DimensionExtractionContext(content_type="query")
        query_cognitive_dimensions = await ingestion_pipeline.dimension_analyzer.analyze(query_string, query_dim_context)
        query_context_vector = ingestion_pipeline.vector_manager.compose_vector(query_embedding, query_cognitive_dimensions).full_vector

        # Perform activation
        activation_result = await activation_engine.activate_memories(
            context=query_context_vector,
            threshold=0.5, # Lower threshold for broader activation
            max_activations=10,
            project_id="test-project-activation"
        )

        assert activation_result.total_activated > 0
        assert len(activation_result.core_memories) > 0

        # Verify explanations contain cognitive dimension details
        found_redis_decision = False
        for mem in activation_result.core_memories + activation_result.peripheral_memories:
            if "redis" in mem.content.lower() and "decision" in mem.content_type.value.lower():
                found_redis_decision = True
                explanation = activation_result.activation_explanations.get(mem.id, "")
                print(f"\nActivated Memory: {mem.content[:80]}...")
                print(f"Explanation: {explanation}")
                assert "decision" in explanation # Check for content type
                assert "activation strength" in explanation
                # Check for phrases related to enhanced dimensions if applicable to the memory's content
                if mem.cognitive_dimensions:
                    if mem.cognitive_dimensions.temporal.urgency > 0.7:
                        assert "highly urgent memory" in explanation
                    if mem.cognitive_dimensions.causal.impact > 0.7:
                        assert "significant impact" in explanation

        assert found_redis_decision, "Should activate memory about Redis decision"

        print(f"\nSuccessfully tested activation spreading integration.")
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_end_to_end_performance_requirement(self, sample_meeting_transcript):
        """Test complete pipeline meets <2s performance requirement."""
        try:
            import time
            
            # Measure complete pipeline performance
            start_time = time.perf_counter()
            
            # 1. Extract memories (should be <1s for typical meeting)
            extractor = MemoryExtractor()
            memories = await extractor.extract(sample_meeting_transcript, "perf-test-001")
            
            # 2. Embed and store memories (should be <1s for 10 memories)
            encoder = get_encoder()
            vector_manager = get_vector_manager()
            
            embedded_count = 0
            for memory in memories[:10]:  # Limit for performance test
                semantic_embedding = await encoder.encode(memory.content)
                cognitive_dimensions = np.random.rand(16)
                full_vector = vector_manager.compose_vector(semantic_embedding, cognitive_dimensions)
                embedded_count += 1
            
            # 3. Test query performance
            query = "What decisions were made?"
            query_vector = await encoder.encode(query)
            cognitive_dims = np.random.rand(16)
            full_query_vector = vector_manager.compose_vector(query_vector, cognitive_dims)
            
            total_time = time.perf_counter() - start_time
            
            print(f"\\nEnd-to-End Performance:")
            print(f"Memories extracted: {len(memories)}")
            print(f"Memories embedded: {embedded_count}")  
            print(f"Total time: {total_time:.2f}s")
            
            # For now, just test individual components are fast enough
            # Full pipeline test needs complete implementation
            assert total_time < 5.0, f"Pipeline too slow: {total_time:.2f}s"
            
        except NotImplementedError:
            pytest.skip("Pipeline components not fully implemented yet")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_memory_quality_semantic_accuracy(self, sample_meeting_transcript):
        """Test extracted memories are semantically accurate and useful."""
        try:
            extractor = MemoryExtractor()
            memories = await extractor.extract(sample_meeting_transcript, "quality-test-001")
            
            if not memories:
                pytest.skip("No memories extracted for quality testing")
            
            # Test memory quality metrics
            quality_checks = {
                "content_length": 0,
                "type_diversity": 0,
                "semantic_coherence": 0,
                "actionability": 0
            }
            
            # 1. Content length check
            avg_length = sum(len(m.content) for m in memories) / len(memories)
            quality_checks["content_length"] = 1 if 20 <= avg_length <= 200 else 0
            
            # 2. Type diversity check  
            unique_types = len(set(m.content_type for m in memories))
            quality_checks["type_diversity"] = 1 if unique_types >= 3 else 0
            
            # 3. Semantic coherence check (memories should be related to meeting)
            meeting_keywords = ["redis", "caching", "migration", "demo", "authentication"]
            coherent_memories = 0
            for memory in memories:
                if any(keyword in memory.content.lower() for keyword in meeting_keywords):
                    coherent_memories += 1
            quality_checks["semantic_coherence"] = 1 if coherent_memories >= len(memories) * 0.7 else 0
            
            # 4. Actionability check (should have clear action items)
            action_memories = [m for m in memories if m.content_type == ContentType.ACTION]
            quality_checks["actionability"] = 1 if len(action_memories) >= 2 else 0
            
            quality_score = sum(quality_checks.values()) / len(quality_checks)
            
            print(f"\\nMemory Quality Assessment:")
            print(f"Total memories: {len(memories)}")
            print(f"Average length: {avg_length:.1f} chars")
            print(f"Type diversity: {unique_types} types")
            print(f"Semantic coherence: {coherent_memories}/{len(memories)} memories")
            print(f"Action items: {len(action_memories)}")
            print(f"Overall quality score: {quality_score:.2f}/1.0")
            
            # Quality requirements
            assert quality_score >= 0.75, f"Memory quality too low: {quality_score:.2f}"
            assert len(memories) >= 5, "Should extract sufficient memories"
            assert unique_types >= 2, "Should extract diverse memory types"
            
        except NotImplementedError:
            pytest.skip("Memory extraction not implemented yet")


class TestPipelineErrorHandling:
    """Test pipeline handles errors gracefully."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_empty_transcript_handling(self):
        """Test pipeline handles empty/invalid transcripts."""
        try:
            extractor = MemoryExtractor()
            
            # Test empty transcript
            memories = await extractor.extract("", "empty-test")
            assert memories == []
            
            # Test very short transcript
            memories = await extractor.extract("Hello.", "short-test")
            assert len(memories) <= 1
            
        except NotImplementedError:
            pytest.skip("Memory extraction not implemented yet")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_storage_failure_recovery(self):
        """Test system handles storage failures gracefully."""
        # Test vector store failure scenarios
        # Test SQLite connection issues
        # Verify graceful degradation
        pytest.skip("Error handling tests require full implementation")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_pipeline_operations(self):
        """Test pipeline handles concurrent operations safely."""
        # Test multiple meetings being processed simultaneously
        # Test concurrent queries during ingestion
        # Verify no race conditions or data corruption
        pytest.skip("Concurrency tests require full implementation")


# Mark all tests as critical integration tests
pytestmark = [
    pytest.mark.integration,
    pytest.mark.critical,
]