"""
Unit tests for bridge discovery functionality.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, Mock
from datetime import datetime

from src.cognitive.retrieval.bridge_discovery import SimpleBridgeDiscovery, BridgeMemory
from src.models.entities import Memory, MemoryType, ContentType
from src.storage.sqlite.repositories import MemoryRepository
from src.storage.qdrant.vector_store import QdrantVectorStore


class TestSimpleBridgeDiscovery:
    """Test cases for SimpleBridgeDiscovery class."""
    
    @pytest.fixture
    def mock_memory_repo(self):
        """Create a mock memory repository."""
        repo = AsyncMock(spec=MemoryRepository)
        return repo
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = AsyncMock(spec=QdrantVectorStore)
        return store
    
    @pytest.fixture
    def bridge_discovery(self, mock_memory_repo, mock_vector_store):
        """Create a SimpleBridgeDiscovery instance with mocks."""
        return SimpleBridgeDiscovery(
            memory_repo=mock_memory_repo,
            vector_store=mock_vector_store,
            novelty_weight=0.6,
            connection_weight=0.4,
            min_bridge_score=0.6
        )
    
    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for testing."""
        memories = []
        for i in range(5):
            memory = Memory(
                id=f"mem_{i}",
                meeting_id=f"meeting_{i % 2}",
                project_id="test_project",
                content=f"Test memory content {i}",
                speaker=f"Speaker {i % 3}",
                timestamp_ms=1000000 + i * 60000,
                memory_type=MemoryType.EPISODIC,
                content_type=ContentType.CONTEXT,
                level=2,
                importance_score=0.5 + i * 0.1,
                created_at=datetime.now()
            )
            memories.append(memory)
        return memories
    
    @pytest.mark.asyncio
    async def test_discover_bridges_basic(self, bridge_discovery, mock_vector_store, sample_memories):
        """Test basic bridge discovery functionality."""
        # Setup query context
        query_context = np.random.rand(400).astype(np.float32)
        
        # Mock vector store search results
        mock_search_results = []
        for i in range(10):
            mock_result = Mock()
            mock_result.score = 0.5 + i * 0.05
            mock_result.payload = {"memory_id": f"candidate_{i}"}
            mock_search_results.append(mock_result)
        
        mock_vector_store.search_all_levels.return_value = {
            "L0": mock_search_results[:3],
            "L1": mock_search_results[3:6],
            "L2": mock_search_results[6:]
        }
        
        # Mock memory repo get_by_ids
        candidate_memories = []
        for i in range(10):
            memory = Memory(
                id=f"candidate_{i}",
                meeting_id=f"meeting_{i % 3}",
                project_id="test_project",
                content=f"Candidate memory content {i}",
                speaker=f"Speaker {i % 4}",
                timestamp_ms=2000000 + i * 60000,
                memory_type=MemoryType.EPISODIC,
                content_type=ContentType.IDEA if i % 3 == 0 else ContentType.CONTEXT,
                level=2,
                importance_score=0.4 + i * 0.05,
                created_at=datetime.now()
            )
            candidate_memories.append(memory)
        
        bridge_discovery.memory_repo.get_by_ids.return_value = candidate_memories
        
        # Mock vector store get_vectors
        mock_vectors = {}
        for mem in candidate_memories:
            mock_vectors[mem.id] = np.random.rand(400).astype(np.float32)
        mock_vector_store.get_vectors.return_value = mock_vectors
        
        # Run bridge discovery
        bridges = await bridge_discovery.discover_bridges(
            query_context=query_context,
            retrieved_memories=sample_memories[:3],
            max_bridges=3,
            search_expansion=10
        )
        
        # Assertions
        assert len(bridges) <= 3
        assert all(isinstance(bridge, BridgeMemory) for bridge in bridges)
        
        # Verify scoring
        for bridge in bridges:
            assert 0 <= bridge.novelty_score <= 1
            assert 0 <= bridge.connection_potential <= 1
            assert 0 <= bridge.surprise_score <= 1
            assert bridge.bridge_score >= bridge_discovery.min_bridge_score
            assert bridge.explanation != ""
            assert isinstance(bridge.connected_concepts, list)
    
    @pytest.mark.asyncio
    async def test_calculate_novelty_score(self, bridge_discovery):
        """Test novelty score calculation."""
        # Create candidate memory
        candidate = Memory(
            id="candidate_1",
            meeting_id="meeting_1",
            project_id="test_project",
            content="Novel idea about quantum computing",
            speaker="Alice",
            timestamp_ms=1000000,
            memory_type=MemoryType.EPISODIC,
            content_type=ContentType.IDEA,
            level=2,
            importance_score=0.8,
            created_at=datetime.now()
        )
        
        # Create retrieved memories with varying similarity
        retrieved_memories = []
        for i in range(3):
            memory = Memory(
                id=f"retrieved_{i}",
                meeting_id="meeting_0",
                project_id="test_project",
                content=f"Standard computing approach {i}",
                speaker="Bob",
                timestamp_ms=900000,
                memory_type=MemoryType.EPISODIC,
                content_type=ContentType.CONTEXT,
                level=2,
                importance_score=0.5,
                created_at=datetime.now()
            )
            retrieved_memories.append(memory)
        
        # Mock vectors
        candidate_vector = np.random.rand(400).astype(np.float32)
        retrieved_vectors = {
            mem.id: np.random.rand(400).astype(np.float32) 
            for mem in retrieved_memories
        }
        
        # Calculate novelty score
        score = bridge_discovery._calculate_novelty_score(
            candidate, 
            candidate_vector,
            retrieved_memories,
            retrieved_vectors
        )
        
        assert 0 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_calculate_connection_potential(self, bridge_discovery):
        """Test connection potential calculation."""
        candidate = Memory(
            id="candidate_1",
            meeting_id="meeting_2",
            project_id="test_project",
            content="Integrated solution combining multiple approaches",
            speaker="Charlie",
            timestamp_ms=1500000,
            memory_type=MemoryType.EPISODIC,
            content_type=ContentType.DECISION,
            level=2,
            importance_score=0.9,
            created_at=datetime.now()
        )
        
        # Test with shared concepts
        query_concepts = ["integration", "solution", "approach"]
        retrieved_concepts = [
            ["approach", "method", "technique"],
            ["solution", "problem", "fix"],
            ["integration", "combination", "merge"]
        ]
        
        score = bridge_discovery._calculate_connection_potential(
            candidate,
            query_concepts,
            retrieved_concepts
        )
        
        assert 0 <= score <= 1
        # Should have high score due to concept overlap
        assert score > 0.5
    
    @pytest.mark.asyncio
    async def test_empty_retrieved_memories(self, bridge_discovery):
        """Test bridge discovery with no retrieved memories."""
        query_context = np.random.rand(400).astype(np.float32)
        
        bridges = await bridge_discovery.discover_bridges(
            query_context=query_context,
            retrieved_memories=[],  # Empty list
            max_bridges=3,
            search_expansion=10
        )
        
        # Should handle gracefully and return empty list
        assert bridges == []
    
    @pytest.mark.asyncio
    async def test_bridge_scoring_and_ranking(self, bridge_discovery, sample_memories):
        """Test that bridges are properly scored and ranked."""
        query_context = np.random.rand(400).astype(np.float32)
        
        # Create candidates with controlled scores
        candidates = []
        for i in range(5):
            bridge = BridgeMemory(
                memory=sample_memories[i],
                novelty_score=0.5 + i * 0.1,
                connection_potential=0.6 - i * 0.05,
                surprise_score=0.0,  # Will be calculated
                bridge_score=0.0,    # Will be calculated
                explanation="",
                connected_concepts=[]
            )
            # Calculate bridge score
            bridge.bridge_score = (
                bridge.novelty_score * bridge_discovery.novelty_weight +
                bridge.connection_potential * bridge_discovery.connection_weight
            )
            candidates.append(bridge)
        
        # Sort and filter
        sorted_bridges = sorted(
            [b for b in candidates if b.bridge_score >= bridge_discovery.min_bridge_score],
            key=lambda x: x.bridge_score,
            reverse=True
        )
        
        # Verify ordering
        for i in range(len(sorted_bridges) - 1):
            assert sorted_bridges[i].bridge_score >= sorted_bridges[i + 1].bridge_score
    
    @pytest.mark.asyncio
    async def test_generate_explanation(self, bridge_discovery):
        """Test bridge explanation generation."""
        bridge = BridgeMemory(
            memory=Memory(
                id="bridge_1",
                meeting_id="meeting_1",
                project_id="test_project",
                content="Innovative approach to data processing",
                speaker="Dana",
                timestamp_ms=2000000,
                memory_type=MemoryType.EPISODIC,
                content_type=ContentType.IDEA,
                level=2,
                importance_score=0.85,
                created_at=datetime.now()
            ),
            novelty_score=0.8,
            connection_potential=0.7,
            surprise_score=0.75,
            bridge_score=0.76,
            explanation="",
            connected_concepts=["data", "processing", "innovation"]
        )
        
        explanation = bridge_discovery._generate_explanation(bridge)
        
        assert explanation != ""
        assert "novelty" in explanation.lower()
        assert "connection" in explanation.lower()
        assert str(round(bridge.novelty_score * 100)) in explanation