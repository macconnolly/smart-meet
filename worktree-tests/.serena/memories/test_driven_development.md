# Test-Driven Development Guide

## Overview
This guide provides comprehensive test examples for each component in the Cognitive Meeting Intelligence system. Follow TDD principles: write tests first, then implement to pass.

## Test Structure

```
tests/
├── unit/                    # Isolated component tests
│   ├── test_models.py
│   ├── test_encoder.py
│   ├── test_extractors.py
│   ├── test_repositories.py
│   └── test_vector_manager.py
├── integration/             # Component interaction tests
│   ├── test_pipeline.py
│   ├── test_storage.py
│   └── test_api.py
├── performance/            # Performance benchmarks
│   ├── test_encoding_speed.py
│   └── test_search_latency.py
└── fixtures/              # Test data and helpers
    ├── sample_transcripts.py
    └── test_vectors.py
```

## Unit Test Examples

### 1. Model Tests
**File**: `tests/unit/test_models.py`
```python
import pytest
from datetime import datetime
from src.models.entities import Memory, Meeting, MemoryType, ConnectionType

class TestMemoryModel:
    def test_memory_creation_with_defaults(self):
        """Test memory creates with proper defaults"""
        memory = Memory(
            content="We decided to use Python",
            memory_type=MemoryType.DECISION
        )
        
        assert memory.id is not None
        assert len(memory.id) == 36  # UUID format
        assert memory.importance == 0.5
        assert memory.access_count == 0
        assert memory.level == 2  # L2 by default
        assert memory.created_at is not None
        
    def test_memory_type_validation(self):
        """Test memory type enum validation"""
        valid_types = [
            MemoryType.DECISION,
            MemoryType.ACTION_ITEM,
            MemoryType.IDEA,
            MemoryType.ISSUE,
            MemoryType.QUESTION,
            MemoryType.CONTEXT
        ]
        
        for mem_type in valid_types:
            memory = Memory(content="Test", memory_type=mem_type)
            assert memory.memory_type == mem_type
    
    def test_memory_importance_bounds(self):
        """Test importance score bounds"""
        memory = Memory(content="Test", importance=1.5)
        # Implementation should clamp to [0, 1]
        assert 0 <= memory.importance <= 1

class TestMeetingModel:
    def test_meeting_creation(self):
        """Test meeting creation with participants"""
        meeting = Meeting(
            title="Sprint Planning",
            participants=["Alice", "Bob", "Charlie"],
            transcript="Discussion content...",
            duration_minutes=45
        )
        
        assert meeting.id is not None
        assert len(meeting.participants) == 3
        assert meeting.duration_minutes == 45
        assert isinstance(meeting.date, datetime)
```

### 2. Encoder Tests
**File**: `tests/unit/test_encoder.py`
```python
import pytest
import numpy as np
import time
from src.embedding.onnx_encoder import ONNXEncoder

class TestONNXEncoder:
    @pytest.fixture
    def encoder(self):
        """Provide encoder instance"""
        return ONNXEncoder()
    
    def test_encoding_dimensions(self, encoder):
        """Test encoder produces correct dimensions"""
        text = "This is a test sentence for encoding"
        embedding = encoder.encode(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32
    
    def test_encoding_normalization(self, encoder):
        """Test embeddings are normalized"""
        embedding = encoder.encode("Test text")
        norm = np.linalg.norm(embedding)
        
        assert abs(norm - 1.0) < 1e-6, f"Expected norm ~1.0, got {norm}"
    
    def test_encoding_performance(self, encoder):
        """Test encoding meets performance requirements"""
        text = "Performance test for encoding speed measurement"
        
        # Warm up
        encoder.encode(text)
        
        # Measure
        times = []
        for _ in range(10):
            start = time.perf_counter()
            encoder.encode(text)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        assert avg_time < 100, f"Encoding too slow: {avg_time:.1f}ms"
    
    def test_encoding_cache(self, encoder):
        """Test caching improves performance"""
        text = "Cache test sentence"
        
        # First call - not cached
        start = time.perf_counter()
        embedding1 = encoder.encode(text)
        first_time = (time.perf_counter() - start) * 1000
        
        # Second call - should be cached
        start = time.perf_counter()
        embedding2 = encoder.encode(text)
        cached_time = (time.perf_counter() - start) * 1000
        
        assert np.array_equal(embedding1, embedding2)
        assert cached_time < first_time / 10  # At least 10x faster
    
    def test_batch_encoding(self, encoder):
        """Test batch encoding functionality"""
        texts = [
            "First sentence",
            "Second sentence", 
            "Third sentence"
        ]
        
        embeddings = encoder.batch_encode(texts)
        
        assert embeddings.shape == (3, 384)
        for i, text in enumerate(texts):
            single = encoder.encode(text)
            assert np.allclose(embeddings[i], single)
```

### 3. Dimension Extractor Tests
**File**: `tests/unit/test_extractors.py`
```python
import pytest
import numpy as np
from src.extraction.dimensions.temporal_extractor import TemporalDimensionExtractor
from src.extraction.dimensions.emotional_extractor import EmotionalDimensionExtractor
from src.extraction.dimensions.dimension_analyzer import DimensionAnalyzer

class TestTemporalExtractor:
    @pytest.fixture
    def extractor(self):
        return TemporalDimensionExtractor()
    
    def test_urgency_detection(self, extractor):
        """Test urgency keyword detection"""
        test_cases = [
            ("This is urgent!", 1.0),
            ("Please complete ASAP", 1.0),
            ("This is important", 0.6),
            ("Normal task", 0.0),
        ]
        
        for text, expected_min in test_cases:
            features = extractor.extract(text)
            urgency = features[0]
            assert urgency >= expected_min, f"'{text}' urgency {urgency} < {expected_min}"
    
    def test_deadline_detection(self, extractor):
        """Test deadline proximity detection"""
        test_cases = [
            ("Complete by May 15", 0.7),
            ("Due before June 1", 0.7),
            ("No deadline mentioned", 0.0),
        ]
        
        for text, expected in test_cases:
            features = extractor.extract(text)
            deadline = features[1]
            assert abs(deadline - expected) < 0.1
    
    def test_sequence_position(self, extractor):
        """Test sequence position with context"""
        text = "Next steps"
        context = {"position": 8, "total": 10}
        
        features = extractor.extract(text, context)
        position = features[2]
        
        assert position == 0.8  # 8/10

class TestEmotionalExtractor:
    @pytest.fixture
    def extractor(self):
        return EmotionalDimensionExtractor()
    
    def test_sentiment_analysis(self, extractor):
        """Test VADER sentiment extraction"""
        test_cases = [
            ("This is absolutely fantastic!", 0.8, 1.0),  # positive
            ("This is terrible and awful", 0.0, 0.3),     # negative
            ("This is a neutral statement", 0.4, 0.6),    # neutral
        ]
        
        for text, min_pol, max_pol in test_cases:
            features = extractor.extract(text)
            polarity = features[0]
            assert min_pol <= polarity <= max_pol
    
    def test_emotional_intensity(self, extractor):
        """Test emotional intensity detection"""
        intense_text = "I absolutely LOVE this amazing idea!!!"
        neutral_text = "The meeting is at 3pm"
        
        intense_features = extractor.extract(intense_text)
        neutral_features = extractor.extract(neutral_text)
        
        assert intense_features[1] > neutral_features[1]

class TestDimensionAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return DimensionAnalyzer()
    
    def test_full_dimension_extraction(self, analyzer):
        """Test complete 16D extraction"""
        text = "This is an urgent decision we need to make today!"
        features = analyzer.extract(text)
        
        assert features.shape == (16,)
        assert np.all(features >= 0)
        assert np.all(features <= 1)
    
    def test_dimension_names(self, analyzer):
        """Test dimension name mapping"""
        names = analyzer.get_dimension_names()
        assert len(names) == 16
        assert names[0] == "urgency"
        assert names[4] == "sentiment_polarity"
```

### 4. Repository Tests
**File**: `tests/unit/test_repositories.py`
```python
import pytest
import asyncio
from datetime import datetime
from src.models.entities import Memory, MemoryType
from src.storage.sqlite.connection import DatabaseConnection
from src.storage.sqlite.repositories.memory_repository import MemoryRepository

class TestMemoryRepository:
    @pytest.fixture
    async def repo(self, tmp_path):
        """Create repository with test database"""
        db_path = tmp_path / "test.db"
        db = DatabaseConnection(str(db_path))
        
        # Initialize schema
        with open("src/storage/sqlite/schema.sql") as f:
            schema = f.read()
        
        with db.get_connection() as conn:
            conn.executescript(schema)
        
        return MemoryRepository(db)
    
    @pytest.mark.asyncio
    async def test_create_and_retrieve(self, repo):
        """Test basic CRUD operations"""
        # Create
        memory = Memory(
            meeting_id="meet-123",
            content="Test decision",
            memory_type=MemoryType.DECISION,
            importance=0.8
        )
        
        memory_id = await repo.create(memory)
        assert memory_id == memory.id
        
        # Retrieve
        retrieved = await repo.get_by_id(memory_id)
        assert retrieved is not None
        assert retrieved.content == "Test decision"
        assert retrieved.memory_type == MemoryType.DECISION
        assert retrieved.importance == 0.8
    
    @pytest.mark.asyncio
    async def test_get_by_meeting(self, repo):
        """Test retrieval by meeting ID"""
        meeting_id = "meet-456"
        
        # Create multiple memories
        memories = [
            Memory(meeting_id=meeting_id, content=f"Memory {i}", timestamp_ms=i*1000)
            for i in range(5)
        ]
        
        for memory in memories:
            await repo.create(memory)
        
        # Retrieve by meeting
        retrieved = await repo.get_by_meeting(meeting_id)
        assert len(retrieved) == 5
        
        # Check ordering by timestamp
        for i in range(4):
            assert retrieved[i].timestamp_ms < retrieved[i+1].timestamp_ms
    
    @pytest.mark.asyncio
    async def test_update_access(self, repo):
        """Test access count updates"""
        memory = Memory(content="Test")
        memory_id = await repo.create(memory)
        
        # Initial state
        retrieved = await repo.get_by_id(memory_id)
        assert retrieved.access_count == 0
        assert retrieved.last_accessed is None
        
        # Update access
        await repo.update_access(memory_id)
        
        # Verify update
        updated = await repo.get_by_id(memory_id)
        assert updated.access_count == 1
        assert updated.last_accessed is not None
```

### 5. Vector Manager Tests
**File**: `tests/unit/test_vector_manager.py`
```python
import pytest
import numpy as np
from src.embedding.vector_manager import VectorManager

class TestVectorManager:
    @pytest.fixture
    def manager(self):
        return VectorManager()
    
    def test_vector_composition(self, manager):
        """Test 384D + 16D → 400D composition"""
        # Create normalized semantic vector
        semantic = np.random.rand(384).astype(np.float32)
        semantic = semantic / np.linalg.norm(semantic)
        
        # Create feature vector
        features = np.random.rand(16).astype(np.float32)
        
        # Compose
        composed = manager.compose_vector(semantic, features)
        
        assert composed.shape == (400,)
        assert np.array_equal(composed[:384], semantic)
        assert np.array_equal(composed[384:], features)
    
    def test_vector_decomposition(self, manager):
        """Test 400D → 384D + 16D decomposition"""
        # Create test vector
        vector = np.random.rand(400).astype(np.float32)
        vector[:384] = vector[:384] / np.linalg.norm(vector[:384])
        
        # Decompose
        semantic, features = manager.decompose_vector(vector)
        
        assert semantic.shape == (384,)
        assert features.shape == (16,)
        assert np.array_equal(semantic, vector[:384])
        assert np.array_equal(features, vector[384:])
    
    def test_composition_validation(self, manager):
        """Test input validation"""
        # Wrong dimensions should raise
        with pytest.raises(ValueError):
            manager.compose_vector(np.zeros(380), np.zeros(16))
        
        with pytest.raises(ValueError):
            manager.compose_vector(np.zeros(384), np.zeros(15))
    
    def test_feature_normalization(self, manager):
        """Test features are clipped to [0,1]"""
        semantic = np.ones(384) / np.sqrt(384)  # Normalized
        features = np.array([-0.5, 0.5, 1.5] + [0.5] * 13)  # Some out of bounds
        
        composed = manager.compose_vector(semantic, features)
        result_features = composed[384:]
        
        assert result_features[0] == 0.0  # Clipped from -0.5
        assert result_features[1] == 0.5   # Unchanged
        assert result_features[2] == 1.0   # Clipped from 1.5
```

## Integration Test Examples

### 1. Pipeline Integration Tests
**File**: `tests/integration/test_pipeline.py`
```python
import pytest
import asyncio
from src.pipeline.ingestion_pipeline import MeetingIngestionPipeline
from tests.fixtures.sample_transcripts import SAMPLE_TRANSCRIPTS

class TestIngestionPipeline:
    @pytest.fixture
    async def pipeline(self, test_db, test_qdrant):
        """Create pipeline with test dependencies"""
        # Initialize all components
        # Return configured pipeline
        pass
    
    @pytest.mark.asyncio
    async def test_end_to_end_ingestion(self, pipeline):
        """Test complete transcript processing"""
        transcript = SAMPLE_TRANSCRIPTS["sprint_planning"]
        
        result = await pipeline.ingest_meeting(
            title="Sprint Planning",
            transcript=transcript,
            participants=["Alice", "Bob", "Charlie"]
        )
        
        assert result["status"] == "success"
        assert result["memories_extracted"] >= 10
        assert result["processing_time_ms"] < 2000
        
        # Verify storage
        meeting_id = result["meeting_id"]
        memories = await pipeline.memory_repo.get_by_meeting(meeting_id)
        
        assert len(memories) == result["memories_extracted"]
        
        # Verify vectors stored
        for memory in memories[:5]:  # Check first 5
            vector_data = await pipeline.vector_store.get_by_id(memory.id)
            assert vector_data is not None
            assert vector_data["vector"].shape == (400,)
    
    @pytest.mark.asyncio
    async def test_memory_type_extraction(self, pipeline):
        """Test different memory types are extracted"""
        transcript = """
        Alice: Let's start the meeting.
        Bob: We decided to use React for the frontend.
        Charlie: I'll set up the development environment.
        Alice: Great idea! What if we also add TypeScript?
        Bob: I'm concerned about the timeline.
        Charlie: Question: Do we have the budget approved?
        """
        
        result = await pipeline.ingest_meeting(
            title="Test Meeting",
            transcript=transcript,
            participants=["Alice", "Bob", "Charlie"]
        )
        
        memories = await pipeline.memory_repo.get_by_meeting(result["meeting_id"])
        
        # Check memory types
        types = [m.memory_type for m in memories]
        assert MemoryType.DECISION in types
        assert MemoryType.ACTION_ITEM in types
        assert MemoryType.IDEA in types
        assert MemoryType.ISSUE in types
        assert MemoryType.QUESTION in types
```

### 2. Storage Integration Tests
**File**: `tests/integration/test_storage.py`
```python
import pytest
import numpy as np
from src.storage.qdrant.vector_store import QdrantVectorStore

class TestQdrantIntegration:
    @pytest.fixture
    async def vector_store(self):
        """Initialize Qdrant for testing"""
        store = QdrantVectorStore(host="localhost", port=6333)
        # Clean up test data after
        yield store
        # Cleanup code
    
    @pytest.mark.asyncio
    async def test_multi_level_storage(self, vector_store):
        """Test storing in different levels"""
        test_vector = np.random.rand(400).astype(np.float32)
        test_vector[:384] = test_vector[:384] / np.linalg.norm(test_vector[:384])
        
        # Store in each level
        for level in [0, 1, 2]:
            await vector_store.store_memory(
                memory_id=f"test-{level}",
                vector=test_vector,
                level=level,
                metadata={"test": True, "level": level}
            )
        
        # Search each level
        for level in [0, 1, 2]:
            results = await vector_store.search(
                query_vector=test_vector,
                level=level,
                limit=1
            )
            
            assert len(results) == 1
            assert results[0][0] == f"test-{level}"
            assert results[0][1] > 0.99  # High similarity
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, vector_store):
        """Test metadata filtering"""
        base_vector = np.random.rand(400).astype(np.float32)
        
        # Store memories with different metadata
        for i in range(10):
            vector = base_vector + np.random.rand(400) * 0.1
            vector[:384] = vector[:384] / np.linalg.norm(vector[:384])
            
            await vector_store.store_memory(
                memory_id=f"mem-{i}",
                vector=vector,
                level=2,
                metadata={
                    "meeting_id": "meet-1" if i < 5 else "meet-2",
                    "type": "decision" if i % 2 == 0 else "action"
                }
            )
        
        # Search with filter
        results = await vector_store.search(
            query_vector=base_vector,
            level=2,
            limit=10,
            filters={"meeting_id": "meet-1"}
        )
        
        assert len(results) == 5
        for _, _, metadata in results:
            assert metadata["meeting_id"] == "meet-1"
```

### 3. API Integration Tests
**File**: `tests/integration/test_api.py`
```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_ingest_endpoint(self, client):
        """Test meeting ingestion"""
        request_data = {
            "title": "API Test Meeting",
            "transcript": "Alice: Let's test the API. Bob: Great idea!",
            "participants": ["Alice", "Bob"]
        }
        
        response = client.post("/ingest", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["memories_extracted"] >= 2
        assert "meeting_id" in data
        assert data["processing_time_ms"] < 2000
    
    def test_search_endpoint(self, client):
        """Test memory search"""
        # First ingest some data
        ingest_data = {
            "title": "Search Test",
            "transcript": "We decided to implement search functionality.",
            "participants": ["Dev"]
        }
        
        ingest_response = client.post("/ingest", json=ingest_data)
        assert ingest_response.status_code == 200
        
        # Now search
        search_data = {
            "query": "What did we decide about search?",
            "limit": 5
        }
        
        response = client.post("/search", json=search_data)
        
        assert response.status_code == 200
        
        results = response.json()
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check result structure
        first = results[0]
        assert "memory_id" in first
        assert "content" in first
        assert "score" in first
        assert "metadata" in first
    
    def test_error_handling(self, client):
        """Test API error responses"""
        # Invalid request
        response = client.post("/ingest", json={})
        assert response.status_code == 422  # Validation error
        
        # Invalid search
        response = client.post("/search", json={"limit": 5})
        assert response.status_code == 422  # Missing query
```

## Performance Test Examples

### 1. Encoding Performance
**File**: `tests/performance/test_encoding_speed.py`
```python
import pytest
import time
import statistics
from src.embedding.onnx_encoder import ONNXEncoder

class TestEncodingPerformance:
    @pytest.fixture
    def encoder(self):
        return ONNXEncoder()
    
    @pytest.mark.benchmark
    def test_single_encoding_speed(self, encoder, benchmark):
        """Benchmark single text encoding"""
        text = "This is a typical meeting utterance that needs to be encoded."
        
        # Warm up
        encoder.encode(text)
        
        # Benchmark
        result = benchmark(encoder.encode, text)
        
        assert result.shape == (384,)
        assert benchmark.stats["mean"] < 0.1  # 100ms
        assert benchmark.stats["stddev"] < 0.02  # Low variance
    
    @pytest.mark.benchmark  
    def test_batch_encoding_speed(self, encoder, benchmark):
        """Benchmark batch encoding efficiency"""
        texts = [
            f"Meeting utterance number {i} with some content"
            for i in range(100)
        ]
        
        def batch_encode():
            return encoder.batch_encode(texts)
        
        result = benchmark(batch_encode)
        
        assert result.shape == (100, 384)
        
        # Should be more efficient than individual
        time_per_text = benchmark.stats["mean"] / 100
        assert time_per_text < 0.05  # 50ms per text in batch
    
    def test_cache_performance(self, encoder):
        """Test cache impact on performance"""
        texts = [f"Unique text {i}" for i in range(100)]
        
        # First pass - no cache
        start = time.perf_counter()
        for text in texts:
            encoder.encode(text)
        first_pass = time.perf_counter() - start
        
        # Second pass - all cached
        start = time.perf_counter()
        for text in texts:
            encoder.encode(text)
        cached_pass = time.perf_counter() - start
        
        # Cache should be at least 10x faster
        assert cached_pass < first_pass / 10
```

### 2. Search Performance
**File**: `tests/performance/test_search_latency.py`
```python
import pytest
import asyncio
import numpy as np
from src.storage.qdrant.vector_store import QdrantVectorStore

class TestSearchPerformance:
    @pytest.fixture
    async def populated_store(self):
        """Create store with test data"""
        store = QdrantVectorStore()
        
        # Populate with 10k vectors
        for i in range(10000):
            vector = np.random.rand(400).astype(np.float32)
            vector[:384] = vector[:384] / np.linalg.norm(vector[:384])
            
            await store.store_memory(
                memory_id=f"perf-{i}",
                vector=vector,
                level=2,
                metadata={"index": i}
            )
        
        return store
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_search_latency(self, populated_store, benchmark):
        """Test search performance with 10k vectors"""
        query = np.random.rand(400).astype(np.float32)
        query[:384] = query[:384] / np.linalg.norm(query[:384])
        
        async def search():
            return await populated_store.search(query, level=2, limit=10)
        
        results = await benchmark(search)
        
        assert len(results) == 10
        assert benchmark.stats["mean"] < 0.05  # 50ms average
        assert benchmark.stats["max"] < 0.1   # 100ms worst case
```

## Test Fixtures

### Sample Transcripts
**File**: `tests/fixtures/sample_transcripts.py`
```python
SAMPLE_TRANSCRIPTS = {
    "sprint_planning": """
        John: Good morning everyone. Let's start our sprint planning.
        Sarah: We decided to prioritize the authentication module.
        Mike: I'll implement the JWT tokens this week.
        Sarah: That's urgent - we need it by Friday.
        John: What about the database migrations?
        Mike: I'm concerned they might break existing data.
        Sarah: Good point. Let's create a backup strategy first.
        John: Action item: Mike to document the migration plan.
        """,
    
    "design_review": """
        Alice: Welcome to the design review session.
        Bob: I have an idea - what if we use a card-based layout?
        Charlie: That's interesting. How would it work on mobile?
        Bob: We could stack the cards vertically.
        Alice: I like it. Let's prototype this approach.
        Charlie: Question: Do we have user research supporting this?
        Alice: Yes, check the research doc I shared yesterday.
        """
}
```

### Test Utilities
**File**: `tests/fixtures/test_helpers.py`
```python
import numpy as np
from typing import List
from src.models.entities import Memory, MemoryType

def create_test_vector(seed: int = 42) -> np.ndarray:
    """Create a valid 400D test vector"""
    np.random.seed(seed)
    vector = np.random.rand(400).astype(np.float32)
    # Normalize semantic part
    vector[:384] = vector[:384] / np.linalg.norm(vector[:384])
    # Clip features to [0,1]
    vector[384:] = np.clip(vector[384:], 0, 1)
    return vector

def create_test_memories(count: int, meeting_id: str) -> List[Memory]:
    """Create test memory objects"""
    memories = []
    types = list(MemoryType)
    
    for i in range(count):
        memory = Memory(
            meeting_id=meeting_id,
            content=f"Test memory content {i}",
            memory_type=types[i % len(types)],
            timestamp_ms=i * 1000,
            speaker=f"Speaker{i % 3}"
        )
        memories.append(memory)
    
    return memories
```

## Testing Best Practices

### 1. Test Naming Convention
```python
def test_should_extract_urgent_memories_with_high_priority():
    """Test method should describe expected behavior"""
    pass
```

### 2. Arrange-Act-Assert Pattern
```python
def test_memory_extraction():
    # Arrange
    extractor = MemoryExtractor()
    transcript = "Test transcript"
    
    # Act
    memories = extractor.extract(transcript, "meeting-123")
    
    # Assert
    assert len(memories) > 0
```

### 3. Parametrized Tests
```python
@pytest.mark.parametrize("text,expected_type", [
    ("We decided to use Python", MemoryType.DECISION),
    ("I'll complete this tomorrow", MemoryType.ACTION_ITEM),
    ("What if we tried a different approach?", MemoryType.IDEA),
])
def test_memory_classification(text, expected_type):
    classifier = MemoryClassifier()
    assert classifier.classify(text) == expected_type
```

### 4. Async Test Patterns
```python
@pytest.mark.asyncio
async def test_async_operation():
    # Use pytest-asyncio for async tests
    result = await async_function()
    assert result is not None
```

### 5. Performance Assertions
```python
def test_performance_requirement():
    with Timer() as timer:
        result = expensive_operation()
    
    assert timer.elapsed < 0.1  # 100ms requirement
    assert result is not None
```

## Running Tests

### Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_encoder.py

# Run performance tests
pytest tests/performance/ -v

# Run with markers
pytest -m "not benchmark"  # Skip benchmark tests
pytest -m "asyncio"        # Only async tests
```

### Configuration
**File**: `pytest.ini`
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
markers =
    benchmark: marks tests as performance benchmarks
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## Test Coverage Goals

### Phase 1 Targets
- Unit test coverage: >90%
- Integration test coverage: >80%
- Critical paths: 100%
- Performance tests: All benchmarks passing

### Coverage Report
```
src/models/entities.py          100%
src/embedding/onnx_encoder.py    95%
src/extraction/memory_extractor  92%
src/storage/repositories         94%
src/api/main.py                  88%
```