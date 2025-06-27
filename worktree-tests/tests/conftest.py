"""
Pytest configuration and shared fixtures for Cognitive Meeting Intelligence System.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Generator, AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
TEST_QDRANT_URL = os.getenv("TEST_QDRANT_URL", "http://localhost:6333")
TEST_QDRANT_COLLECTION = "test_memories"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_db_engine():
    """Create test database engine."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    # Create tables
    from src.models.entities import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest.fixture
async def db_session(test_db_engine):
    """Create a test database session."""
    async_session = sessionmaker(
        test_db_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
def sample_meeting_data():
    """Sample meeting data for testing."""
    return {
        "id": "test-meeting-123",
        "title": "Weekly Team Sync",
        "start_time": datetime.now(),
        "end_time": datetime.now(),
        "participants": ["Alice", "Bob", "Charlie"],
        "transcript": "Alice: Let's discuss the caching strategy...",
        "metadata": {"platform": "zoom", "recording_available": True}
    }


@pytest.fixture
def sample_memory_data():
    """Sample memory data for testing."""
    return {
        "meeting_id": "test-meeting-123",
        "content": "We need to implement caching for better performance",
        "speaker": "Alice",
        "timestamp": datetime.now(),
        "memory_type": "decision",
        "content_type": "statement",
        "level": 2,
        "importance_score": 0.8,
        "decay_rate": 0.1,
        "dimensions": {
            "temporal": [0.8, 0.5, 0.3, 0.1],
            "emotional": [0.6, 0.4, 0.7],
            "social": [0.9, 0.5, 0.3],
            "causal": [0.7, 0.6, 0.5],
            "evolutionary": [0.4, 0.3, 0.2]
        }
    }


@pytest.fixture
def mock_qdrant_client(mocker):
    """Mock Qdrant client for testing."""
    client = mocker.Mock()
    client.get_collections.return_value = mocker.Mock(collections=[])
    client.create_collection.return_value = True
    client.upsert.return_value = mocker.Mock(status="ok")
    client.search.return_value = []
    return client


@pytest.fixture
def mock_encoder(mocker):
    """Mock ONNX encoder for testing."""
    import numpy as np
    
    encoder = mocker.Mock()
    encoder.encode.return_value = np.random.rand(384).astype(np.float32)
    encoder.encode_batch.return_value = np.random.rand(10, 384).astype(np.float32)
    return encoder


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary directory for test files."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    return test_dir


@pytest.fixture
def sample_transcript_file(temp_test_dir):
    """Create a sample transcript file for testing."""
    transcript = {
        "meeting_id": "test-123",
        "segments": [
            {
                "speaker": "Alice",
                "text": "We need to optimize our caching strategy",
                "timestamp": "00:00:10"
            },
            {
                "speaker": "Bob",
                "text": "I agree, let's use Redis for distributed caching",
                "timestamp": "00:00:25"
            }
        ]
    }
    
    file_path = temp_test_dir / "sample_transcript.json"
    with open(file_path, "w") as f:
        json.dump(transcript, f)
    
    return file_path


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    # Add any singleton reset logic here
    yield


@pytest.fixture
def mock_llm_response(mocker):
    """Mock LLM API responses for testing."""
    def _mock_response(content):
        response = mocker.Mock()
        response.choices = [mocker.Mock(message=mocker.Mock(content=content))]
        return response
    return _mock_response


@pytest.fixture
async def populated_vector_store(mock_qdrant_client, mock_encoder):
    """Create a vector store with sample data."""
    from src.storage.qdrant.vector_store import VectorStore
    
    store = VectorStore(client=mock_qdrant_client, encoder=mock_encoder)
    
    # Add sample vectors
    sample_vectors = [
        {"id": f"vec-{i}", "vector": np.random.rand(400).tolist(), "payload": {"content": f"Memory {i}"}}
        for i in range(10)
    ]
    
    mock_qdrant_client.search.return_value = sample_vectors[:3]
    
    return store


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "api: API tests")
    config.addinivalue_line("markers", "cognitive: Cognitive algorithm tests")
    config.addinivalue_line("markers", "database: Database tests")
    config.addinivalue_line("markers", "vector: Vector store tests")
    config.addinivalue_line("markers", "skip_ci: Skip in CI")


# Performance tracking
@pytest.fixture(autouse=True)
def track_test_performance(request):
    """Track test execution time."""
    import time
    start_time = time.time()
    yield
    duration = time.time() - start_time
    
    # Log slow tests
    if duration > 1.0 and hasattr(request.node, "rep_call"):
        print(f"\nSlow test: {request.node.nodeid} took {duration:.2f}s")