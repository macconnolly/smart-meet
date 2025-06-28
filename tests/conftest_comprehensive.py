"""
Shared test fixtures and configuration for pytest.

This file is automatically loaded by pytest and provides common fixtures
used across all test files.
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock

import pytest
import numpy as np

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ===== Database Fixtures =====

@pytest.fixture
async def test_db():
    """Create a temporary SQLite database for testing."""
    from src.storage.sqlite.connection import DatabaseConnection
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = DatabaseConnection(str(db_path))
        await db.initialize()
        yield db
        await db.close()


@pytest.fixture
async def db_repositories(test_db):
    """Create all repository instances with test database."""
    from src.storage.sqlite.repositories import (
        MemoryRepository, MeetingRepository, ProjectRepository,
        StakeholderRepository, DeliverableRepository, 
        MemoryConnectionRepository
    )
    
    return {
        'memory': MemoryRepository(test_db),
        'meeting': MeetingRepository(test_db),
        'project': ProjectRepository(test_db),
        'stakeholder': StakeholderRepository(test_db),
        'deliverable': DeliverableRepository(test_db),
        'connection': MemoryConnectionRepository(test_db)
    }


# ===== Vector Store Fixtures =====

@pytest.fixture
def mock_vector_store():
    """Create a mock Qdrant vector store."""
    store = Mock()
    store.initialize = AsyncMock()
    store.upsert = AsyncMock()
    store.search = AsyncMock(return_value=[])
    store.delete = AsyncMock()
    store.get_collection_info = AsyncMock(return_value={'vectors_count': 0})
    return store


@pytest.fixture
async def real_vector_store():
    """Create a real Qdrant vector store (requires Qdrant running)."""
    pytest.importorskip("qdrant_client")
    
    from src.storage.qdrant.vector_store import QdrantVectorStore
    
    store = QdrantVectorStore(
        host="localhost",
        port=6333,
        collection_name="test_collection"
    )
    
    try:
        await store.initialize()
        yield store
        # Cleanup
        await store.delete_collection()
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")


# ===== Component Fixtures =====

@pytest.fixture
def memory_extractor():
    """Create a memory extractor instance."""
    from src.extraction.memory_extractor import MemoryExtractor
    return MemoryExtractor()


@pytest.fixture
def dimension_analyzer():
    """Create a dimension analyzer instance."""
    from src.extraction.dimensions.dimension_analyzer import get_dimension_analyzer
    return get_dimension_analyzer()


@pytest.fixture
def mock_encoder():
    """Create a mock ONNX encoder."""
    encoder = Mock()
    encoder.encode = AsyncMock(
        return_value=np.random.rand(384).astype(np.float32)
    )
    encoder.encode_batch = AsyncMock(
        side_effect=lambda texts: np.random.rand(len(texts), 384).astype(np.float32)
    )
    return encoder


@pytest.fixture
def vector_manager():
    """Create a vector manager instance."""
    from src.embedding.vector_manager import VectorManager
    return VectorManager()


# ===== Test Data Fixtures =====

@pytest.fixture
def sample_transcript():
    """Provide a sample meeting transcript."""
    return """
    Sarah: Welcome everyone to our sprint planning session. Let's start by reviewing last sprint's velocity.
    
    Tom: We completed 42 story points last sprint, which is slightly below our average of 45.
    
    Sarah: That's still good progress. Now, let's look at the backlog. The authentication refactor is our top priority.
    
    Emily: I'm concerned about the timeline for that. It's at least a 13-point story, and we have the client demo next week.
    
    Tom: We also need to urgently fix the performance regression in the search API. Users are complaining.
    
    Sarah: Good point. Let's make the performance fix our immediate priority. Emily, can you take that?
    
    Emily: Yes, I'll start on it right after this meeting. Should have it done by tomorrow.
    
    Sarah: Perfect. For the auth refactor, let's break it down into smaller pieces. Tom, any suggestions?
    
    Tom: We could separate the OAuth implementation from the session management. That way we can tackle them independently.
    
    Sarah: Excellent idea. Let's estimate each piece separately and see what we can fit into this sprint.
    """


@pytest.fixture
def sample_memories():
    """Provide sample memory objects."""
    from src.models.entities import Memory, MemoryType, ContentType
    
    return [
        Memory(
            id="mem-001",
            meeting_id="meeting-001",
            project_id="project-001",
            content="We completed 42 story points last sprint",
            memory_type=MemoryType.EPISODIC,
            content_type=ContentType.METRIC,
            speaker="Tom",
            timestamp_ms=10000,
            importance_score=0.7
        ),
        Memory(
            id="mem-002",
            meeting_id="meeting-001",
            project_id="project-001",
            content="The authentication refactor is our top priority",
            memory_type=MemoryType.EPISODIC,
            content_type=ContentType.DECISION,
            speaker="Sarah",
            timestamp_ms=30000,
            importance_score=0.9
        ),
        Memory(
            id="mem-003",
            meeting_id="meeting-001", 
            project_id="project-001",
            content="We need to urgently fix the performance regression",
            memory_type=MemoryType.EPISODIC,
            content_type=ContentType.RISK,
            speaker="Tom",
            timestamp_ms=50000,
            importance_score=0.95
        )
    ]


@pytest.fixture
def sample_connections():
    """Provide sample memory connections."""
    from src.models.entities import MemoryConnection
    
    return [
        MemoryConnection(
            source_id="mem-002",
            target_id="mem-003",
            connection_strength=0.8,
            connection_type="causal"
        ),
        MemoryConnection(
            source_id="mem-001",
            target_id="mem-002",
            connection_strength=0.6,
            connection_type="temporal"
        )
    ]


@pytest.fixture
async def populated_db(db_repositories, sample_memories, sample_connections):
    """Create a database populated with test data."""
    # Create project
    project_repo = db_repositories['project']
    project_id = await project_repo.create(
        name="Test Project",
        description="Test project for unit tests"
    )
    
    # Create meeting
    meeting_repo = db_repositories['meeting']
    meeting_id = await meeting_repo.create(
        project_id=project_id,
        title="Sprint Planning",
        start_time=datetime.now(),
        participants=["Sarah", "Tom", "Emily"]
    )
    
    # Update memory IDs to match
    for memory in sample_memories:
        memory.project_id = project_id
        memory.meeting_id = meeting_id
    
    # Create memories
    memory_repo = db_repositories['memory']
    for memory in sample_memories:
        await memory_repo.create(memory)
    
    # Create connections
    connection_repo = db_repositories['connection']
    for connection in sample_connections:
        await connection_repo.create(
            source_id=connection.source_id,
            target_id=connection.target_id,
            connection_strength=connection.connection_strength,
            connection_type=connection.connection_type
        )
    
    # Return all repos with data
    return {
        'project_id': project_id,
        'meeting_id': meeting_id,
        'memories': sample_memories,
        'connections': sample_connections,
        **db_repositories
    }


# ===== Test Helpers =====

class TestDataFactory:
    """Factory for creating test data consistently."""
    
    @staticmethod
    def create_vector(dim: int = 400, normalize: bool = False) -> np.ndarray:
        """Create a random vector of specified dimension."""
        vector = np.random.rand(dim).astype(np.float32)
        if normalize:
            vector = vector / np.linalg.norm(vector)
        return vector
    
    @staticmethod
    def create_meeting_data(**kwargs) -> Dict[str, Any]:
        """Create meeting data with defaults."""
        return {
            "title": kwargs.get("title", "Test Meeting"),
            "start_time": kwargs.get("start_time", datetime.now()),
            "participants": kwargs.get("participants", ["Alice", "Bob"]),
            "transcript": kwargs.get("transcript", "Alice: Test content."),
            **kwargs
        }
    
    @staticmethod
    def create_memory_data(**kwargs) -> Dict[str, Any]:
        """Create memory data with defaults."""
        return {
            "content": kwargs.get("content", "Test memory content"),
            "memory_type": kwargs.get("memory_type", "episodic"),
            "content_type": kwargs.get("content_type", "insight"),
            "speaker": kwargs.get("speaker", "Alice"),
            "timestamp_ms": kwargs.get("timestamp_ms", 1000),
            "importance_score": kwargs.get("importance_score", 0.5),
            **kwargs
        }


@pytest.fixture
def test_factory():
    """Provide test data factory."""
    return TestDataFactory()


# ===== Async Test Configuration =====

@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for async tests."""
    if sys.platform == "win32":
        # Windows requires special handling
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )


# ===== Performance Testing Helpers =====

@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.times = []
        
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.end = time.perf_counter()
            self.times.append(self.end - self.start)
        
        @property
        def last(self):
            return self.times[-1] if self.times else 0
        
        @property
        def average(self):
            return sum(self.times) / len(self.times) if self.times else 0
    
    return Timer()


# ===== Markers =====

def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_qdrant: mark test as requiring Qdrant"
    )


# ===== Cleanup =====

@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Reset singleton instances between tests."""
    yield
    # Reset dimension analyzer singleton
    from src.extraction.dimensions import dimension_analyzer
    dimension_analyzer._analyzer_instance = None