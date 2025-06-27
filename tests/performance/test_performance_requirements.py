"""
CRITICAL: Performance validation tests.

These tests validate that the system meets its stated performance requirements:
- Memory extraction: 10-15/second  
- Embedding generation: <100ms
- Full cognitive query: <2s
- Support 10K+ memories

FAILURE OF THESE TESTS MEANS THE SYSTEM DOES NOT MEET SPECS.

Reference: TESTING_STRATEGY.md - Phase 2: Performance Validation
"""

import pytest
import time
import statistics
import asyncio
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.embedding.onnx_encoder import ONNXEncoder, get_encoder
from src.embedding.vector_manager import VectorManager, get_vector_manager  
from src.extraction.memory_extractor import MemoryExtractor
from src.models.entities import Memory, Meeting, Vector


class TestEncodingPerformance:
    """Test encoding performance requirements."""
    
    @pytest.fixture
    def encoder(self):
        """Get real encoder instance."""
        return get_encoder()
    
    @pytest.fixture
    def test_texts(self):
        """Generate realistic meeting text samples."""
        return [
            "We need to implement caching for better performance.",
            "The decision was made to use Redis for our caching layer.",
            "I'll handle the implementation and have it ready by Friday.",
            "What are the potential risks with this approach?",
            "The team agreed on the new architecture design.",
            "We should conduct a security review before deployment.",
            "The client wants the feature delivered next quarter.",
            "Let's schedule a follow-up meeting to discuss progress.",
            "The budget has been approved for the new infrastructure.",
            "We need to update the documentation with these changes.",
        ] * 10  # 100 samples
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_encoding_under_100ms(self, encoder, test_texts):
        """Test single text encoding meets <100ms requirement."""
        sample_text = test_texts[0]
        
        # Warm up (first encoding may be slower due to model loading)
        await encoder.encode(sample_text)
        
        # Measure 50 encoding operations
        times = []
        for _ in range(50):
            start = time.perf_counter()
            embedding = await encoder.encode(sample_text)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
            
            # Verify embedding quality
            assert embedding.shape == (384,)
            assert np.abs(np.linalg.norm(embedding) - 1.0) < 1e-5
        
        # Statistical analysis
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        percentile_95 = sorted(times)[int(0.95 * len(times))]
        
        print(f"\\nEncoding Performance Statistics:")
        print(f"Mean: {mean_time:.1f}ms")
        print(f"Median: {median_time:.1f}ms") 
        print(f"95th percentile: {percentile_95:.1f}ms")
        
        # CRITICAL: 95% of encodings must be under 100ms
        assert percentile_95 < 100, f"95th percentile {percentile_95:.1f}ms exceeds 100ms limit"
        assert mean_time < 50, f"Mean time {mean_time:.1f}ms should be well under limit"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_encoding_efficiency(self, encoder, test_texts):
        """Test batch encoding is more efficient than individual encoding."""
        batch_sizes = [1, 5, 10, 25, 50]
        
        results = {}
        
        for batch_size in batch_sizes:
            batch = test_texts[:batch_size]
            
            # Measure batch encoding time
            start = time.perf_counter()
            embeddings = await encoder.encode_batch(batch)
            batch_time = time.perf_counter() - start
            
            # Verify results
            assert embeddings.shape == (batch_size, 384)
            
            # Calculate time per text
            time_per_text = (batch_time * 1000) / batch_size
            results[batch_size] = time_per_text
            
            print(f"Batch {batch_size}: {time_per_text:.1f}ms per text")
        
        # Batch encoding should be more efficient for larger batches
        assert results[50] < results[1], "Batch encoding should be more efficient"
        assert results[50] < 50, f"Batch encoding {results[50]:.1f}ms per text too slow"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_encoding_performance(self, encoder, test_texts):
        """Test concurrent encoding operations don't degrade performance."""
        # Test 20 concurrent encoding operations
        async def encode_text(text):
            start = time.perf_counter()
            embedding = await encoder.encode(text)
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, embedding
        
        # Create 20 concurrent tasks
        tasks = [encode_text(text) for text in test_texts[:20]]
        
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start
        
        times, embeddings = zip(*results)
        
        # Verify all encodings completed
        assert len(results) == 20
        assert all(emb.shape == (384,) for emb in embeddings)
        
        # Performance checks
        max_time = max(times)
        mean_time = statistics.mean(times)
        
        print(f"\\nConcurrent Encoding Performance:")
        print(f"Total time: {total_time:.2f}s")
        print(f"Mean individual time: {mean_time:.1f}ms")
        print(f"Max individual time: {max_time:.1f}ms")
        
        # No individual encoding should exceed 200ms even under concurrency
        assert max_time < 200, f"Max time {max_time:.1f}ms too high under concurrency"
        assert mean_time < 100, f"Mean time {mean_time:.1f}ms degraded under concurrency"


class TestVectorPerformance:
    """Test vector operations performance."""
    
    @pytest.fixture
    def vector_manager(self):
        """Get real vector manager instance."""
        return get_vector_manager()
    
    @pytest.mark.performance
    def test_vector_composition_performance(self, vector_manager):
        """Test vector composition is fast enough for real-time use."""
        # Generate test vectors
        semantic_vectors = [np.random.randn(384) for _ in range(1000)]
        dimension_vectors = [np.random.rand(16) for _ in range(1000)]
        
        # Normalize semantic vectors
        semantic_vectors = [v / np.linalg.norm(v) for v in semantic_vectors]
        
        # Measure composition time
        start = time.perf_counter()
        
        composed_vectors = []
        for semantic, dimensions in zip(semantic_vectors, dimension_vectors):
            composed = vector_manager.compose_vector(semantic, dimensions)
            composed_vectors.append(composed)
        
        elapsed = time.perf_counter() - start
        time_per_composition = (elapsed * 1000) / 1000  # ms per composition
        
        print(f"Vector composition: {time_per_composition:.3f}ms per vector")
        
        # Should be very fast - under 1ms per composition
        assert time_per_composition < 1.0, f"Composition too slow: {time_per_composition:.3f}ms"
        
        # Verify all compositions are correct
        assert len(composed_vectors) == 1000
        assert all(v.shape == (400,) for v in composed_vectors)
    
    @pytest.mark.performance
    def test_similarity_calculation_performance(self, vector_manager):
        """Test vector similarity calculations are fast enough."""
        # Generate 1000 test vectors
        vectors = []
        for _ in range(1000):
            semantic = np.random.randn(384)
            semantic = semantic / np.linalg.norm(semantic)
            dimensions = np.random.rand(16)
            vector = vector_manager.compose_vector(semantic, dimensions)
            vectors.append(vector)
        
        query_vector = vectors[0]
        
        # Measure similarity calculation time
        start = time.perf_counter()
        
        similarities = []
        for vector in vectors[1:]:
            # Use cosine similarity for semantic part
            semantic_sim = np.dot(query_vector[:384], vector[:384])
            similarities.append(semantic_sim)
        
        elapsed = time.perf_counter() - start
        time_per_similarity = (elapsed * 1000) / 999  # ms per similarity calc
        
        print(f"Similarity calculation: {time_per_similarity:.4f}ms per pair")
        
        # Should be very fast - under 0.1ms per calculation
        assert time_per_similarity < 0.1, f"Similarity calc too slow: {time_per_similarity:.4f}ms"
        
        # Verify similarity range
        assert all(-1 <= s <= 1 for s in similarities)


class TestMemoryExtractionPerformance:
    """Test memory extraction performance requirements."""
    
    @pytest.fixture
    def memory_extractor(self):
        """Get memory extractor instance."""
        return MemoryExtractor()
    
    @pytest.fixture
    def sample_transcripts(self):
        """Generate sample meeting transcripts."""
        base_transcript = '''
        Alice: Good morning everyone. Let's start our weekly sync meeting.
        Bob: I've completed the caching implementation using Redis.
        Alice: That's great! How's the performance looking?
        Bob: We're seeing a 40% improvement in response times.
        Charlie: Excellent work. What about the security review?
        Alice: I'll schedule that for next week with the security team.
        Bob: We should also update our documentation.
        Charlie: Agreed. I can help with that.
        Alice: Perfect. Any other urgent items?
        Bob: The client wants a demo next Friday.
        Alice: I'll coordinate with them to set that up.
        Charlie: Should we prepare any specific metrics?
        Bob: Yes, let's show the performance improvements.
        Alice: Great. Let's wrap up and schedule our next meeting.
        '''
        
        # Create variations for different transcript lengths
        transcripts = []
        for i in range(10):
            # Vary transcript length and content
            transcript = base_transcript + f"\\nAdditional context {i}."
            transcripts.append(transcript)
        
        return transcripts
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_extraction_rate_10_15_per_second(self, memory_extractor, sample_transcripts):
        """Test memory extraction meets 10-15 memories/second requirement."""
        # This test will fail until memory extractor is properly implemented
        if not hasattr(memory_extractor, 'extract') or not callable(memory_extractor.extract):
            pytest.skip("Memory extractor not implemented yet")
        
        total_memories_extracted = 0
        
        # Process multiple transcripts
        start = time.perf_counter()
        
        for transcript in sample_transcripts:
            try:
                memories = await memory_extractor.extract(transcript, meeting_id=f"test-{hash(transcript)}")
                total_memories_extracted += len(memories)
            except NotImplementedError:
                pytest.skip("Memory extraction not implemented yet")
        
        elapsed = time.perf_counter() - start
        
        if total_memories_extracted == 0:
            pytest.fail("No memories extracted - check extractor implementation")
        
        extraction_rate = total_memories_extracted / elapsed
        
        print(f"\\nMemory Extraction Performance:")
        print(f"Total memories: {total_memories_extracted}")
        print(f"Time: {elapsed:.2f}s") 
        print(f"Rate: {extraction_rate:.1f} memories/second")
        
        # CRITICAL: Must meet 10-15 memories/second requirement
        assert extraction_rate >= 10, f"Extraction rate {extraction_rate:.1f}/s below 10/s minimum"
        assert extraction_rate <= 50, f"Extraction rate {extraction_rate:.1f}/s suspiciously high - check quality"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_extraction_quality_vs_speed(self, memory_extractor, sample_transcripts):
        """Test that fast extraction doesn't compromise quality."""
        # This ensures we're not just counting words as "memories"
        if not hasattr(memory_extractor, 'extract'):
            pytest.skip("Memory extractor not implemented yet")
        
        transcript = sample_transcripts[0]
        
        try:
            memories = await memory_extractor.extract(transcript, meeting_id="quality-test")
        except NotImplementedError:
            pytest.skip("Memory extraction not implemented yet")
        
        if not memories:
            pytest.fail("No memories extracted for quality testing")
        
        # Quality checks
        assert len(memories) >= 3, "Should extract multiple meaningful memories"
        assert len(memories) <= 20, "Should not over-extract trivial content"
        
        # Check memory types diversity
        memory_types = {m.content_type for m in memories}
        assert len(memory_types) >= 2, "Should extract diverse memory types"
        
        # Check content quality - memories should be substantial
        for memory in memories:
            assert len(memory.content.strip()) >= 10, f"Memory too short: '{memory.content}'"
            assert memory.content != transcript, "Memory should not be entire transcript"


class TestFullSystemPerformance:
    """Test end-to-end system performance requirements."""
    
    @pytest.mark.performance
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_cognitive_query_under_2_seconds(self):
        """Test complete cognitive query cycle meets <2s requirement."""
        # This is the most critical performance test
        # Query → Vector → Activation → Results must be under 2 seconds
        
        # Skip until full pipeline is implemented
        pytest.skip("Full cognitive pipeline not implemented yet")
        
        # When implemented, test:
        # 1. Create query "What decisions were made about caching?"
        # 2. Vector encoding
        # 3. Activation spreading through 1000+ memories
        # 4. Results ranking and return
        # Total time must be < 2 seconds
    
    @pytest.mark.performance
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_10k_memory_support(self):
        """Test system handles 10K+ memories without degradation."""
        pytest.skip("Large-scale testing requires full implementation")
        
        # When implemented:
        # 1. Load 10,000 memories into system
        # 2. Test query performance remains under 2s
        # 3. Test memory insertion performance
        # 4. Test concurrent query handling
    
    @pytest.mark.performance
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_concurrent_operations_no_degradation(self):
        """Test concurrent operations don't cause performance degradation."""
        pytest.skip("Stress testing requires full implementation")
        
        # When implemented:
        # 1. Run 10 concurrent queries
        # 2. Run 5 concurrent memory insertions
        # 3. Verify no deadlocks or timeouts
        # 4. Verify performance targets still met


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    @pytest.mark.performance
    def test_performance_baseline_documentation(self):
        """Document current performance baselines for regression testing."""
        # This test documents expected performance levels
        # Update these values as improvements are made
        
        performance_targets = {
            "encoding_mean_ms": 50,
            "encoding_95th_percentile_ms": 100,
            "vector_composition_ms": 1.0,
            "similarity_calculation_ms": 0.1,
            "memory_extraction_per_second": 10,
            "full_query_seconds": 2.0,
            "max_supported_memories": 10000,
        }
        
        print("\\nPerformance Targets:")
        for metric, target in performance_targets.items():
            print(f"{metric}: {target}")
        
        # This always passes but serves as documentation
        assert True


# Configure pytest for performance tests
pytestmark = [
    pytest.mark.performance,
    pytest.mark.slow,  # These tests may take longer
]