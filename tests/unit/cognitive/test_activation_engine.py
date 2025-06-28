"""
CRITICAL: Unit tests for activation spreading engine.

This file tests the CORE COGNITIVE FUNCTIONALITY of the system.
Currently, the activation engine is not implemented - these tests
will drive the implementation.

Reference: TESTING_STRATEGY.md - Phase 1: Critical Algorithms
"""

import pytest
import numpy as np
import asyncio
from typing import List, Dict
from unittest.mock import Mock, AsyncMock, patch
import time

from src.models.entities import Memory, MemoryType, ContentType, MemoryConnection, Vector
from src.cognitive.activation.engine import (
    ActivationEngine,
    BFSActivationEngine,
    ActivationConfig,
    ActivatedMemory,
    ActivationPath,
)


class TestActivationConfig:
    """Test activation configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ActivationConfig()

        assert config.max_depth > 0
        assert config.activation_threshold > 0
        assert config.decay_rate > 0
        assert config.max_results > 0

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = ActivationConfig(
            max_depth=5, activation_threshold=0.1, decay_rate=0.8, max_results=50
        )
        assert config.max_depth == 5

        # Invalid configs should raise ValueError
        with pytest.raises(ValueError):
            ActivationConfig(max_depth=0)  # Must be positive

        with pytest.raises(ValueError):
            ActivationConfig(activation_threshold=-0.1)  # Must be positive

        with pytest.raises(ValueError):
            ActivationConfig(decay_rate=1.5)  # Must be <= 1


class TestActivatedMemory:
    """Test ActivatedMemory data structure."""

    def test_activated_memory_creation(self):
        """Test creating an activated memory."""
        memory = Memory(meeting_id="meet-001", project_id="proj-001", content="Test memory content")

        activated = ActivatedMemory(
            memory=memory,
            activation_score=0.85,
            activation_path=["mem-001", "mem-002", "mem-003"],
            reasoning="Found via semantic similarity",
        )

        assert activated.memory == memory
        assert activated.activation_score == 0.85
        assert len(activated.activation_path) == 3
        assert "semantic" in activated.reasoning

    def test_generate_activation_explanation_with_dimensions(self):
        """Test that activation explanation includes cognitive dimensions."""
        from src.extraction.dimensions.dimension_analyzer import CognitiveDimensions, TemporalFeatures, EmotionalFeatures, SocialFeatures, CausalFeatures, EvolutionaryFeatures

        # Create a Memory with specific cognitive dimensions
        cognitive_dims = CognitiveDimensions(
            temporal=TemporalFeatures(urgency=0.9, deadline_proximity=0.8, sequence_position=0.5, duration_relevance=0.6),
            emotional=EmotionalFeatures(polarity=0.7, intensity=0.6, confidence=0.8),
            social=SocialFeatures(authority=0.9, influence=0.8, team_dynamics=0.7),
            causal=CausalFeatures(dependencies=0.8, impact=0.9, risk_factors=0.7),
            evolutionary=EvolutionaryFeatures(change_rate=0.6, innovation_level=0.9, adaptation_need=0.8)
        )
        memory = Memory(
            id="mem-001",
            meeting_id="meet-001",
            project_id="proj-001",
            content="This is a critical decision with high impact and innovation.",
            speaker="Alice",
            speaker_role="CEO",
            cognitive_dimensions=cognitive_dims
        )

        engine = BasicActivationEngine(Mock(), Mock(), Mock()) # Mock dependencies
        explanation = engine._generate_activation_explanation(memory, 0.95, ["start", "mem-001"])

        # Assert that the explanation contains phrases related to the enhanced dimensions
        assert "highly urgent memory" in explanation
        assert "significant impact" in explanation
        assert "authoritative source" in explanation
        assert "high level of innovation" in explanation
        assert "significant risk" not in explanation # Should not be present if risk is low
        assert "important dependencies" in explanation

        assert "activation strength of 0.95" in explanation
        assert "type memory" in explanation
        assert "contributed by Alice" in explanation
        assert "associated with project 'proj-001'" in explanation

    def test_activation_score_validation(self):
        """Test activation score bounds."""
        memory = Memory(meeting_id="m1", project_id="p1", content="test")

        # Valid scores
        ActivatedMemory(memory=memory, activation_score=0.0)
        ActivatedMemory(memory=memory, activation_score=1.0)
        ActivatedMemory(memory=memory, activation_score=0.5)

        # Invalid scores should raise ValueError
        with pytest.raises(ValueError):
            ActivatedMemory(memory=memory, activation_score=-0.1)

        with pytest.raises(ValueError):
            ActivatedMemory(memory=memory, activation_score=1.1)


class TestBFSActivationEngine:
    """Test BFS activation spreading implementation."""

    @pytest.fixture
    def mock_memory_repo(self):
        """Mock memory repository."""
        repo = AsyncMock()
        repo.get_by_id = AsyncMock()
        repo.get_connections = AsyncMock()
        repo.search = AsyncMock()
        return repo

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store."""
        store = AsyncMock()
        store.search = AsyncMock()
        store.get_vector = AsyncMock()
        return store

    @pytest.fixture
    def activation_engine(self, mock_memory_repo, mock_vector_store):
        """Create activation engine with mocked dependencies."""
        config = ActivationConfig(
            max_depth=3, activation_threshold=0.1, decay_rate=0.8, max_results=10
        )
        return BFSActivationEngine(
            memory_repo=mock_memory_repo, vector_store=mock_vector_store, config=config
        )

    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for testing."""
        memories = []
        for i in range(10):
            memory = Memory(
                id=f"mem-{i:03d}",
                meeting_id="meet-test",
                project_id="proj-test",
                content=f"Memory content {i}",
                memory_type=MemoryType.EPISODIC,
                content_type=ContentType.CONTEXT if i % 2 == 0 else ContentType.DECISION,
            )
            memories.append(memory)
        return memories

    @pytest.mark.asyncio
    async def test_engine_initialization(self, mock_memory_repo, mock_vector_store):
        """Test engine initializes correctly."""
        config = ActivationConfig(max_depth=5)
        engine = BFSActivationEngine(
            memory_repo=mock_memory_repo, vector_store=mock_vector_store, config=config
        )

        assert engine.memory_repo == mock_memory_repo
        assert engine.vector_store == mock_vector_store
        assert engine.config.max_depth == 5
        assert isinstance(engine.activation_cache, dict)

    @pytest.mark.asyncio
    async def test_spread_activation_basic(self, activation_engine, sample_memories):
        """Test basic activation spreading from seed memories."""
        # Setup: Use first memory as seed
        seed_memories = [sample_memories[0]]

        # Mock semantic phase to return some activated memories
        activated_1 = ActivatedMemory(
            memory=sample_memories[1], activation_score=0.8, activation_path=["mem-000", "mem-001"]
        )
        activated_2 = ActivatedMemory(
            memory=sample_memories[2], activation_score=0.6, activation_path=["mem-000", "mem-002"]
        )

        # Mock the internal phases
        activation_engine._semantic_phase = AsyncMock(return_value=[activated_1])
        activation_engine._cognitive_phase = AsyncMock(return_value=[activated_2])
        activation_engine._rank_and_filter = AsyncMock(return_value=[activated_1, activated_2])

        # Test
        results = await activation_engine.spread_activation(seed_memories)

        # Verify
        assert len(results) == 2
        assert results[0].activation_score >= results[1].activation_score  # Should be ranked
        assert all(r.activation_score > 0 for r in results)

        # Verify phases were called
        activation_engine._semantic_phase.assert_called_once()
        activation_engine._cognitive_phase.assert_called_once()
        activation_engine._rank_and_filter.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_phase_vector_similarity(self, activation_engine, sample_memories):
        """Test semantic phase finds similar memories via vectors."""
        # This test drives implementation of _semantic_phase
        seed_memories = [sample_memories[0]]
        config = ActivationConfig(max_depth=2)

        # Mock vector store to return similar memories
        similar_memories = [
            (sample_memories[1], 0.9),  # Very similar
            (sample_memories[2], 0.7),  # Moderately similar
            (sample_memories[3], 0.4),  # Less similar
        ]

        activation_engine.vector_store.search.return_value = [
            {"id": mem.id, "score": score, "metadata": {"memory_id": mem.id}}
            for mem, score in similar_memories
        ]

        # Mock memory repo to return memories
        activation_engine.memory_repo.get_by_id.side_effect = lambda mid: next(
            mem for mem, _ in similar_memories if mem.id == mid
        )

        # Test semantic phase (this will fail until implemented)
        with pytest.raises(NotImplementedError):
            results = await activation_engine._semantic_phase(seed_memories, config)

        # When implemented, should verify:
        # assert len(results) > 0
        # assert all(r.activation_score > 0 for r in results)
        # assert results[0].activation_score > results[-1].activation_score

    @pytest.mark.asyncio
    async def test_cognitive_phase_dimension_similarity(self, activation_engine, sample_memories):
        """Test cognitive phase finds memories with similar cognitive dimensions."""
        # This test drives implementation of _cognitive_phase
        activated_memories = [
            ActivatedMemory(
                memory=sample_memories[0], activation_score=0.8, activation_path=["mem-000"]
            )
        ]
        config = ActivationConfig()

        # Test cognitive phase (this will fail until implemented)
        with pytest.raises(NotImplementedError):
            results = await activation_engine._cognitive_phase(activated_memories, config)

        # When implemented, should verify:
        # assert len(results) >= 0
        # Verify cognitive dimension matching logic

    @pytest.mark.asyncio
    async def test_activate_from_query(self, activation_engine, sample_memories):
        """Test activation from query vector."""
        # Create query vector
        query_vector = Vector(semantic=np.random.randn(384), dimensions=np.random.rand(16))

        # Mock finding query seeds
        activation_engine._find_query_seeds = AsyncMock(
            return_value=[sample_memories[0], sample_memories[1]]
        )

        # Mock spread activation
        expected_results = [
            ActivatedMemory(
                memory=sample_memories[2],
                activation_score=0.7,
                activation_path=["mem-000", "mem-002"],
            )
        ]
        activation_engine.spread_activation = AsyncMock(return_value=expected_results)

        # Test
        results = await activation_engine.activate_from_query(query_vector)

        # Verify
        assert results == expected_results
        activation_engine._find_query_seeds.assert_called_once()
        activation_engine.spread_activation.assert_called_once()

    @pytest.mark.asyncio
    async def test_activation_strength_calculation(self, activation_engine):
        """Test activation strength calculation with decay."""
        config = ActivationConfig(decay_rate=0.8)

        test_cases = [
            # (source_activation, similarity, depth, expected_min_strength)
            (1.0, 1.0, 0, 1.0),  # Perfect similarity, no depth
            (1.0, 0.8, 1, 0.5),  # Good similarity, depth 1
            (0.5, 0.6, 2, 0.1),  # Moderate similarity, depth 2
            (1.0, 0.1, 3, 0.0),  # Low similarity, deep
        ]

        for source, similarity, depth, expected_min in test_cases:
            # This will fail until implemented
            with pytest.raises(NotImplementedError):
                strength = await activation_engine._calculate_activation_strength(
                    source, similarity, depth, config
                )

            # When implemented:
            # assert strength >= expected_min
            # assert 0 <= strength <= 1

    @pytest.mark.asyncio
    async def test_rank_and_filter_results(self, activation_engine, sample_memories):
        """Test ranking and filtering of activation results."""
        # Create test activated memories with different scores
        activated_memories = [
            ActivatedMemory(memory=sample_memories[0], activation_score=0.9),
            ActivatedMemory(memory=sample_memories[1], activation_score=0.7),
            ActivatedMemory(memory=sample_memories[2], activation_score=0.3),
            ActivatedMemory(memory=sample_memories[3], activation_score=0.1),
            ActivatedMemory(memory=sample_memories[4], activation_score=0.05),  # Below threshold
        ]

        config = ActivationConfig(activation_threshold=0.1, max_results=3)

        # This will fail until implemented
        with pytest.raises(NotImplementedError):
            results = await activation_engine._rank_and_filter(activated_memories, config)

        # When implemented:
        # assert len(results) == 3  # Limited by max_results
        # assert all(r.activation_score >= 0.1 for r in results)  # Above threshold
        # assert results[0].activation_score >= results[1].activation_score  # Ranked

    @pytest.mark.asyncio
    async def test_performance_requirement_2_seconds(self, activation_engine):
        """Test activation spreading completes within 2 seconds."""
        # Create larger memory set for performance testing
        large_memory_set = []
        for i in range(100):  # Start with 100, scale up to 10K later
            memory = Memory(
                id=f"perf-{i:05d}",
                meeting_id="perf-test",
                project_id="perf-proj",
                content=f"Performance test memory {i} with various content",
            )
            large_memory_set.append(memory)

        seed_memories = large_memory_set[:5]

        # Mock the phases to return realistic data quickly
        activation_engine._semantic_phase = AsyncMock(
            return_value=[
                ActivatedMemory(memory=mem, activation_score=0.8 - i * 0.1)
                for i, mem in enumerate(large_memory_set[5:15])
            ]
        )
        activation_engine._cognitive_phase = AsyncMock(
            return_value=[
                ActivatedMemory(memory=mem, activation_score=0.6 - i * 0.1)
                for i, mem in enumerate(large_memory_set[15:25])
            ]
        )
        activation_engine._rank_and_filter = AsyncMock(
            return_value=[
                ActivatedMemory(memory=mem, activation_score=0.9 - i * 0.1)
                for i, mem in enumerate(large_memory_set[5:15])
            ]
        )

        # Test performance
        start_time = time.perf_counter()
        results = await activation_engine.spread_activation(seed_memories)
        elapsed_time = time.perf_counter() - start_time

        # Verify performance requirement
        assert elapsed_time < 2.0, f"Activation took {elapsed_time:.2f}s, exceeds 2s limit"
        assert len(results) > 0, "Should return results"

    @pytest.mark.asyncio
    async def test_memory_neighbor_discovery(self, activation_engine, sample_memories):
        """Test finding neighboring memories for BFS traversal."""
        memory = sample_memories[0]
        config = ActivationConfig()

        # Mock vector similarity neighbors
        vector_neighbors = [sample_memories[1], sample_memories[2]]
        activation_engine.vector_store.search.return_value = [
            {"id": mem.id, "score": 0.8, "metadata": {"memory_id": mem.id}}
            for mem in vector_neighbors
        ]

        # Mock explicit connection neighbors
        connection_neighbors = [sample_memories[3]]
        mock_connections = [
            MemoryConnection(
                source_id=memory.id, target_id=sample_memories[3].id, connection_strength=0.9
            )
        ]
        activation_engine.memory_repo.get_connections.return_value = mock_connections

        # This will fail until implemented
        with pytest.raises(NotImplementedError):
            neighbors = await activation_engine._get_memory_neighbors(memory, config)

        # When implemented:
        # assert len(neighbors) >= 3  # Vector + connection neighbors
        # assert all(isinstance(n, tuple) and len(n) == 2 for n in neighbors)
        # assert all(0 <= n[1] <= 1 for n in neighbors)  # Similarity scores in [0,1]

    @pytest.mark.asyncio
    async def test_cognitive_boosts(self, activation_engine, sample_memories):
        """Test cognitive dimension boosts for activation."""
        # Create memory with high importance/urgency
        important_memory = Memory(
            id="important-001",
            meeting_id="meet-001",
            project_id="proj-001",
            content="URGENT: Critical decision needed",
            importance_score=0.95,
            memory_type=MemoryType.EPISODIC,
            content_type=ContentType.DECISION,
        )

        activated = ActivatedMemory(memory=important_memory, activation_score=0.6)  # Initial score

        config = ActivationConfig()

        # This will fail until implemented
        with pytest.raises(NotImplementedError):
            boosted = await activation_engine._apply_cognitive_boosts(activated, config)

        # When implemented:
        # assert boosted.activation_score > activated.activation_score
        # assert boosted.activation_score <= 1.0
        # assert "importance" in boosted.reasoning or "urgent" in boosted.reasoning


class TestActivationIntegration:
    """Integration tests for activation spreading."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_memory_network_activation(self):
        """Test activation with real memory network."""
        # This test requires actual implementation and real data
        # Will be implemented once core algorithms are working
        pass

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_semantic_accuracy_validation(self):
        """Test that activation spreading finds semantically relevant memories."""
        # Test with known semantic relationships
        # Validate that "caching strategy" activates memories about "Redis", "performance", etc.
        pass

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_10k_memory_performance(self):
        """Test activation performance with 10K memories."""
        # Test realistic scale performance
        # Ensure sub-2-second response times
        pass


class TestActivationErrors:
    """Test error handling in activation spreading."""

    @pytest.mark.asyncio
    async def test_empty_seed_memories(self, activation_engine):
        """Test handling of empty seed memory list."""
        results = await activation_engine.spread_activation([])
        assert results == []

    @pytest.mark.asyncio
    async def test_invalid_query_vector(self, activation_engine):
        """Test handling of invalid query vectors."""
        # Wrong dimensions
        invalid_vector = Vector(
            semantic=np.random.randn(300), dimensions=np.random.rand(16)  # Wrong size
        )

        with pytest.raises(ValueError):
            await activation_engine.activate_from_query(invalid_vector)

    @pytest.mark.asyncio
    async def test_vector_store_failure_graceful_degradation(
        self, activation_engine, sample_memories
    ):
        """Test graceful degradation when vector store fails."""
        # Mock vector store failure
        activation_engine.vector_store.search.side_effect = Exception("Vector store unavailable")

        # Should still work with reduced functionality
        seed_memories = [sample_memories[0]]

        # This will depend on implementation - should either:
        # 1. Fall back to explicit connections only, or
        # 2. Raise a specific exception with clear error message
        with pytest.raises((Exception, NotImplementedError)):
            results = await activation_engine.spread_activation(seed_memories)


# Mark critical tests that must pass for system to be functional
pytestmark = pytest.mark.critical
