# Unit & Integration Testing Strategy for Cognitive Meeting Intelligence

## Overview

This guide provides a comprehensive testing strategy for the Cognitive Meeting Intelligence system, covering unit tests, integration tests, and best practices specific to this architecture.

## Testing Philosophy

### 1. Test Pyramid
```
         /\
        /  \  E2E Tests (5%)
       /----\
      /      \ Integration Tests (25%)
     /--------\
    /          \ Unit Tests (70%)
   /____________\
```

- **Unit Tests**: Fast, isolated, test single components
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test complete user workflows

### 2. Key Principles
- **Fast Feedback**: Unit tests should run in < 1 second
- **Isolation**: Mock external dependencies
- **Deterministic**: Same input = same output
- **Readable**: Tests document behavior
- **Maintainable**: DRY principle, good test helpers

## Unit Testing Strategy

### 1. Component-Specific Unit Tests

#### Dimension Extractors
```python
# tests/unit/extraction/test_temporal_extractor.py
import pytest
from src.extraction.dimensions.temporal_extractor import TemporalDimensionExtractor

class TestTemporalExtractor:
    @pytest.fixture
    def extractor(self):
        return TemporalDimensionExtractor()
    
    @pytest.mark.parametrize("content,expected_urgency", [
        ("This is urgent!", 1.0),
        ("ASAP - critical issue", 1.0),
        ("Regular meeting notes", 0.0),
        ("Due tomorrow morning", 0.8),
    ])
    def test_urgency_detection(self, extractor, content, expected_urgency):
        result = extractor.extract(content)
        assert abs(result.urgency - expected_urgency) < 0.1
    
    def test_deadline_proximity_calculation(self, extractor):
        # Test with explicit deadline
        result = extractor.extract("Meeting tomorrow at 2 PM")
        assert result.deadline_proximity > 0.5
        
    def test_empty_content(self, extractor):
        result = extractor.extract("")
        assert result.urgency == 0.0
        assert result.deadline_proximity == 0.0
```

#### Memory Extraction
```python
# tests/unit/extraction/test_memory_extractor.py
import pytest
from unittest.mock import Mock, AsyncMock
from src.extraction.memory_extractor import MemoryExtractor
from src.models.entities import ContentType

class TestMemoryExtractor:
    @pytest.fixture
    def extractor(self):
        return MemoryExtractor()
    
    @pytest.mark.asyncio
    async def test_extract_decision(self, extractor):
        transcript = "John: We've decided to implement caching."
        memories = await extractor.extract_memories(transcript)
        
        assert len(memories) == 1
        assert memories[0].content_type == ContentType.DECISION
        assert "caching" in memories[0].content
    
    @pytest.mark.asyncio
    async def test_speaker_attribution(self, extractor):
        transcript = """
        Sarah: This is my first point.
        Tom: I disagree with that approach.
        """
        memories = await extractor.extract_memories(transcript)
        
        assert memories[0].speaker == "Sarah"
        assert memories[1].speaker == "Tom"
    
    @pytest.mark.asyncio
    async def test_long_transcript_chunking(self, extractor):
        # Test that long statements are properly chunked
        long_statement = "This is important. " * 50
        transcript = f"Speaker: {long_statement}"
        
        memories = await extractor.extract_memories(transcript)
        assert len(memories) > 1  # Should be chunked
        assert all(len(m.content) < 500 for m in memories)
```

#### Vector Manager
```python
# tests/unit/embedding/test_vector_manager.py
import numpy as np
import pytest
from src.embedding.vector_manager import VectorManager
from src.models.entities import Vector

class TestVectorManager:
    @pytest.fixture
    def manager(self):
        return VectorManager()
    
    def test_compose_vector(self, manager):
        semantic = np.random.rand(384).astype(np.float32)
        cognitive = np.random.rand(16).astype(np.float32)
        
        vector = manager.compose(semantic, cognitive)
        
        assert isinstance(vector, Vector)
        assert vector.semantic.shape == (384,)
        assert vector.dimensions.shape == (16,)
        assert vector.full.shape == (400,)
    
    def test_decompose_vector(self, manager):
        full_vector = np.random.rand(400).astype(np.float32)
        
        vector = manager.decompose(full_vector)
        
        assert vector.semantic.shape == (384,)
        assert vector.dimensions.shape == (16,)
        np.testing.assert_array_equal(
            np.concatenate([vector.semantic, vector.dimensions]),
            full_vector
        )
    
    def test_normalize_semantic(self, manager):
        semantic = np.array([3, 4], dtype=np.float32)  # Norm = 5
        cognitive = np.array([0.5, 0.5], dtype=np.float32)
        
        vector = manager.compose(semantic, cognitive, normalize_semantic=True)
        
        # Check semantic is normalized
        assert np.allclose(np.linalg.norm(vector.semantic), 1.0)
        # Check cognitive unchanged
        np.testing.assert_array_equal(vector.dimensions, cognitive)
```

### 2. Mocking Strategies

#### Mock External Dependencies
```python
# tests/unit/cognitive/test_activation_engine.py
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from src.cognitive.activation.basic_activation_engine import BasicActivationEngine
from src.models.entities import Memory, MemoryConnection

class TestActivationEngine:
    @pytest.fixture
    def mock_memory_repo(self):
        repo = Mock()
        repo.get = AsyncMock()
        repo.get_by_ids = AsyncMock()
        return repo
    
    @pytest.fixture
    def mock_connection_repo(self):
        repo = Mock()
        repo.get_connections = AsyncMock()
        return repo
    
    @pytest.fixture
    def mock_vector_store(self):
        store = Mock()
        store.search = AsyncMock()
        return store
    
    @pytest.fixture
    def engine(self, mock_memory_repo, mock_connection_repo, mock_vector_store):
        return BasicActivationEngine(
            memory_repo=mock_memory_repo,
            connection_repo=mock_connection_repo,
            vector_store=mock_vector_store
        )
    
    @pytest.mark.asyncio
    async def test_activation_spreading(self, engine, mock_memory_repo, mock_connection_repo):
        # Setup test data
        start_memory = Memory(id="1", content="Start", importance_score=0.9)
        connected_memory = Memory(id="2", content="Connected", importance_score=0.7)
        
        mock_memory_repo.get_by_ids.return_value = [start_memory]
        mock_connection_repo.get_connections.return_value = [
            MemoryConnection(source_id="1", target_id="2", connection_strength=0.8)
        ]
        mock_memory_repo.get.return_value = connected_memory
        
        # Run activation
        result = await engine.activate_memories(
            query="test",
            starting_memory_ids=["1"],
            max_hops=1
        )
        
        # Verify activation spread
        assert len(result.activated_memories) == 2
        assert result.activated_memories[0].memory.id == "1"
        assert result.activated_memories[1].memory.id == "2"
        assert result.activated_memories[1].activation_score < 0.9  # Decayed
```

#### Mock Async Operations
```python
# tests/unit/test_async_components.py
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_dimension_analyzer_caching():
    with patch('src.extraction.dimensions.dimension_analyzer.DimensionCache') as MockCache:
        cache_instance = MockCache.return_value
        cache_instance.get = Mock(return_value=None)
        cache_instance.put = Mock()
        
        from src.extraction.dimensions.dimension_analyzer import get_dimension_analyzer
        analyzer = get_dimension_analyzer()
        
        # First call - cache miss
        result1 = await analyzer.analyze("test content")
        assert cache_instance.get.called
        assert cache_instance.put.called
        
        # Verify result
        assert result1.temporal.urgency >= 0
        assert result1.temporal.urgency <= 1
```

### 3. Testing Async Code

```python
# tests/unit/test_async_patterns.py
import pytest
import asyncio
from asyncio import TimeoutError

class TestAsyncPatterns:
    @pytest.mark.asyncio
    async def test_concurrent_extraction(self):
        from src.extraction.dimensions.dimension_analyzer import get_dimension_analyzer
        analyzer = get_dimension_analyzer()
        
        # Test concurrent analysis
        contents = ["urgent task", "normal update", "critical issue"]
        
        # Should complete within reasonable time
        tasks = [analyzer.analyze(c) for c in contents]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert results[0].temporal.urgency > results[1].temporal.urgency
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        # Test that long-running operations timeout properly
        async def slow_operation():
            await asyncio.sleep(10)
            
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(slow_operation(), timeout=0.1)
```

## Integration Testing Strategy

### 1. Database Integration Tests

```python
# tests/integration/test_sqlite_integration.py
import pytest
import tempfile
from pathlib import Path
from src.storage.sqlite.connection import DatabaseConnection
from src.storage.sqlite.repositories import MemoryRepository, MeetingRepository

@pytest.fixture
async def test_db():
    """Create a temporary test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = DatabaseConnection(str(db_path))
        await db.initialize()
        yield db
        await db.close()

@pytest.fixture
async def repositories(test_db):
    """Create repository instances."""
    return {
        'memory': MemoryRepository(test_db),
        'meeting': MeetingRepository(test_db)
    }

class TestDatabaseIntegration:
    @pytest.mark.asyncio
    async def test_memory_persistence(self, repositories):
        memory_repo = repositories['memory']
        
        # Create memory
        memory_data = {
            "meeting_id": "test-meeting",
            "content": "Important decision",
            "memory_type": "episodic",
            "content_type": "decision"
        }
        
        memory_id = await memory_repo.create(**memory_data)
        
        # Retrieve and verify
        retrieved = await memory_repo.get(memory_id)
        assert retrieved is not None
        assert retrieved.content == "Important decision"
    
    @pytest.mark.asyncio
    async def test_cascade_delete(self, repositories):
        meeting_repo = repositories['meeting']
        memory_repo = repositories['memory']
        
        # Create meeting with memories
        meeting_id = await meeting_repo.create(title="Test Meeting")
        memory_id = await memory_repo.create(
            meeting_id=meeting_id,
            content="Test memory"
        )
        
        # Delete meeting should cascade
        await meeting_repo.delete(meeting_id)
        
        # Memory should be gone
        memory = await memory_repo.get(memory_id)
        assert memory is None
```

### 2. Vector Store Integration

```python
# tests/integration/test_qdrant_integration.py
import pytest
import numpy as np
from src.storage.qdrant.vector_store import QdrantVectorStore
from src.models.entities import Memory
import docker

def qdrant_available():
    """Check if Qdrant is running."""
    try:
        client = docker.from_env()
        containers = client.containers.list()
        return any('qdrant' in c.name for c in containers)
    except:
        return False

@pytest.mark.skipif(not qdrant_available(), reason="Qdrant not running")
class TestQdrantIntegration:
    @pytest.fixture
    async def vector_store(self):
        store = QdrantVectorStore(
            host="localhost",
            port=6333,
            collection_name="test_memories"
        )
        await store.initialize()
        yield store
        # Cleanup
        await store.delete_collection()
    
    @pytest.mark.asyncio
    async def test_vector_storage_retrieval(self, vector_store):
        # Create test memory with vector
        memory = Memory(
            id="test-1",
            content="Test memory",
            qdrant_id="qdrant-test-1"
        )
        
        vector = np.random.rand(400).astype(np.float32)
        
        # Store
        await vector_store.upsert(
            memory_id=memory.id,
            vector=vector,
            metadata={"content": memory.content}
        )
        
        # Search
        results = await vector_store.search(
            query_vector=vector,
            limit=1
        )
        
        assert len(results) == 1
        assert results[0].id == "qdrant-test-1"
        assert results[0].score > 0.99  # Should be very similar
```

### 3. Pipeline Integration Tests

```python
# tests/integration/test_pipeline_integration.py
import pytest
from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.extraction.memory_extractor import MemoryExtractor
from src.extraction.dimensions.dimension_analyzer import get_dimension_analyzer
from src.embedding.onnx_encoder import ONNXEncoder
from src.storage.sqlite.repositories import MemoryRepository
from src.storage.qdrant.vector_store import QdrantVectorStore

@pytest.mark.integration
class TestPipelineIntegration:
    @pytest.fixture
    async def pipeline(self, test_db, mock_qdrant):
        # Create all components
        memory_extractor = MemoryExtractor()
        dimension_analyzer = get_dimension_analyzer()
        encoder = MockEncoder()  # Use mock for ONNX
        memory_repo = MemoryRepository(test_db)
        vector_store = mock_qdrant
        
        pipeline = IngestionPipeline(
            memory_extractor=memory_extractor,
            dimension_analyzer=dimension_analyzer,
            encoder=encoder,
            memory_repo=memory_repo,
            vector_store=vector_store
        )
        
        return pipeline
    
    @pytest.mark.asyncio
    async def test_full_ingestion_flow(self, pipeline):
        transcript = """
        Sarah: We need to address the performance issues urgently.
        Tom: I agree. The system is too slow for our users.
        Sarah: Let's implement caching as a quick fix.
        Tom: Good idea. I'll start on that tomorrow.
        """
        
        meeting_id = "test-meeting-001"
        
        # Run ingestion
        result = await pipeline.ingest(
            transcript=transcript,
            meeting_id=meeting_id,
            metadata={"title": "Performance Review"}
        )
        
        # Verify results
        assert result.memories_extracted > 0
        assert result.vectors_stored == result.memories_extracted
        assert result.processing_time_ms > 0
        
        # Verify memories have all required fields
        memories = await pipeline.memory_repo.find_by_meeting(meeting_id)
        for memory in memories:
            assert memory.content
            assert memory.dimensions_json
            assert memory.qdrant_id
            assert memory.semantic_embedding is not None
```

### 4. Cognitive Feature Integration

```python
# tests/integration/test_cognitive_integration.py
import pytest
from src.cognitive.activation.basic_activation_engine import BasicActivationEngine
from src.cognitive.retrieval.bridge_discovery import SimpleBridgeDiscovery
from src.cognitive.retrieval.contextual_retrieval import ContextualRetrieval

@pytest.mark.integration
class TestCognitiveIntegration:
    @pytest.fixture
    async def cognitive_system(self, populated_db, vector_store):
        """Create cognitive system with test data."""
        activation_engine = BasicActivationEngine(
            memory_repo=populated_db['memory_repo'],
            connection_repo=populated_db['connection_repo'],
            vector_store=vector_store
        )
        
        bridge_discovery = SimpleBridgeDiscovery(
            memory_repo=populated_db['memory_repo'],
            vector_store=vector_store
        )
        
        contextual_retrieval = ContextualRetrieval(
            activation_engine=activation_engine,
            bridge_discovery=bridge_discovery,
            similarity_search=vector_store
        )
        
        return contextual_retrieval
    
    @pytest.mark.asyncio
    async def test_cognitive_query(self, cognitive_system):
        query = "What are the performance issues we discussed?"
        
        result = await cognitive_system.retrieve(
            query=query,
            enable_activation=True,
            enable_bridges=True,
            max_results=10
        )
        
        # Should find relevant memories
        assert len(result.memories) > 0
        
        # Should have activation paths
        assert any(m.activation_path for m in result.memories)
        
        # Should find bridges
        assert result.bridge_memories is not None
        assert len(result.bridge_memories) > 0
```

## Testing Best Practices

### 1. Test Data Management

```python
# tests/fixtures/test_data.py
import json
from datetime import datetime
from pathlib import Path

class TestDataFactory:
    """Factory for creating consistent test data."""
    
    @staticmethod
    def create_meeting(title="Test Meeting", **kwargs):
        return {
            "id": kwargs.get("id", "test-meeting-001"),
            "title": title,
            "start_time": kwargs.get("start_time", datetime.now()),
            "participants": kwargs.get("participants", ["Alice", "Bob"]),
            "transcript": kwargs.get("transcript", "Alice: Test content.")
        }
    
    @staticmethod
    def create_memory(content="Test memory", **kwargs):
        return {
            "id": kwargs.get("id", "test-memory-001"),
            "meeting_id": kwargs.get("meeting_id", "test-meeting-001"),
            "content": content,
            "memory_type": kwargs.get("memory_type", "episodic"),
            "content_type": kwargs.get("content_type", "insight"),
            "timestamp_ms": kwargs.get("timestamp_ms", 0)
        }
    
    @staticmethod
    def load_transcript(filename):
        """Load test transcript from file."""
        path = Path(__file__).parent / "transcripts" / filename
        with open(path, 'r') as f:
            return json.load(f)
```

### 2. Performance Testing

```python
# tests/performance/test_performance.py
import time
import pytest
import asyncio
from statistics import mean, stdev

class TestPerformance:
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_extraction_speed(self, memory_extractor):
        """Test extraction meets 10-15 memories/second target."""
        transcript = self.load_large_transcript()  # ~1000 words
        
        # Warm up
        await memory_extractor.extract_memories(transcript[:100])
        
        # Measure
        start = time.perf_counter()
        memories = await memory_extractor.extract_memories(transcript)
        duration = time.perf_counter() - start
        
        rate = len(memories) / duration
        
        assert rate >= 10, f"Extraction too slow: {rate:.1f} memories/sec"
        assert rate <= 20, f"Extraction suspiciously fast: {rate:.1f} memories/sec"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cognitive_query_latency(self, cognitive_system):
        """Test end-to-end query meets <2s target."""
        queries = [
            "What decisions were made?",
            "Show me urgent tasks",
            "Find vendor-related issues"
        ]
        
        latencies = []
        
        for query in queries:
            start = time.perf_counter()
            await cognitive_system.retrieve(query)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
        
        avg_latency = mean(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 1500, f"Average latency too high: {avg_latency:.0f}ms"
        assert max_latency < 2000, f"Max latency too high: {max_latency:.0f}ms"
```

### 3. Error Handling Tests

```python
# tests/unit/test_error_handling.py
import pytest
from src.extraction.memory_extractor import MemoryExtractor

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_extractor_handles_malformed_input(self):
        extractor = MemoryExtractor()
        
        # Should not crash on bad input
        bad_inputs = [
            None,
            "",
            "NoSpeaker",
            "Speaker:" * 1000,  # Repetitive
            "\n\n\n",  # Only newlines
            "😀" * 100,  # Emojis
        ]
        
        for bad_input in bad_inputs:
            # Should handle gracefully
            result = await extractor.extract_memories(bad_input or "")
            assert isinstance(result, list)
    
    def test_dimension_extractor_bounds(self):
        from src.extraction.dimensions.temporal_extractor import TemporalDimensionExtractor
        extractor = TemporalDimensionExtractor()
        
        # Test extreme inputs
        result = extractor.extract("URGENT! " * 100)
        assert 0 <= result.urgency <= 1  # Should be clamped
        
        result = extractor.extract("")
        assert result.urgency == 0  # Should have sensible default
```

## Test Execution Strategy

### 1. Test Organization
```
tests/
├── unit/               # Fast, isolated tests
│   ├── extraction/
│   ├── embedding/
│   ├── cognitive/
│   └── storage/
├── integration/        # Component interaction tests
│   ├── test_pipeline.py
│   ├── test_cognitive.py
│   └── test_storage.py
├── e2e/               # Full workflow tests
├── performance/       # Performance benchmarks
├── fixtures/          # Test data and helpers
└── conftest.py        # Shared fixtures
```

### 2. Running Tests

```bash
# Run all tests
pytest

# Run only unit tests (fast)
pytest tests/unit -v

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/extraction/test_temporal_extractor.py -v

# Run tests matching pattern
pytest -k "test_urgency" -v

# Run with markers
pytest -m "not integration" -v  # Skip integration tests
pytest -m "performance" -v      # Only performance tests

# Parallel execution
pytest -n auto  # Use all CPU cores

# With specific log level
pytest --log-cli-level=DEBUG
```

### 3. CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        python -c "import nltk; nltk.download('vader_lexicon')"
    
    - name: Run unit tests
      run: pytest tests/unit -v --cov=src
    
    - name: Run integration tests
      run: pytest tests/integration -v
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Debugging Failed Tests

### 1. Using pytest debugging
```bash
# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest -l

# Verbose output
pytest -vv

# Show print statements
pytest -s
```

### 2. Test Isolation
```python
# Use pytest-xdist for test isolation
pytest --forked  # Run each test in separate process

# Clear state between tests
@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    from src.extraction.dimensions.dimension_analyzer import _analyzer_instance
    _analyzer_instance = None
    yield
    _analyzer_instance = None
```

### 3. Async Debugging
```python
import pytest
import logging

# Enable async logging
logging.getLogger("asyncio").setLevel(logging.DEBUG)

# Debug event loop issues
@pytest.fixture
def event_loop_policy():
    """Use selector event loop for better debugging."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )
```

## Summary

This testing strategy provides:

1. **Comprehensive Coverage**: Unit, integration, and E2E tests
2. **Fast Feedback**: Isolated unit tests run quickly
3. **Realistic Testing**: Integration tests with real components
4. **Performance Validation**: Tests against stated targets
5. **Maintainability**: Clear structure and good practices

Start with unit tests for fast feedback, then add integration tests for confidence, and finally E2E tests for user workflows.