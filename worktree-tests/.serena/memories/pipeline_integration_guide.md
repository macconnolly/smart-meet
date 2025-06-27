# Pipeline Integration Guide

## Overview
This guide explains how components connect and data flows through the Cognitive Meeting Intelligence system.

## Component Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   FastAPI       │────▶│    Pipeline     │────▶│   Storage       │
│   Endpoints     │     │   Orchestrator  │     │   (SQL+Qdrant)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
            ┌──────────────┐      ┌──────────────┐
            │  Extraction  │      │  Embedding   │
            │  Components  │      │  Components  │
            └──────────────┘      └──────────────┘
```

## Data Flow Diagrams

### 1. Meeting Ingestion Flow

```
Transcript Input
      │
      ▼
[Memory Extractor]
      │
      ├─── Extract sentences
      ├─── Identify speakers  
      └─── Classify memory types
      │
      ▼
[For each Memory]
      │
      ├──▶ [ONNX Encoder] ──────▶ 384D embedding
      │
      ├──▶ [Dimension Analyzer] ─▶ 16D features
      │           │
      │           ├── Temporal (4D)
      │           ├── Emotional (3D)
      │           └── Others (9D)
      │
      ▼
[Vector Manager]
      │
      ├─── Compose 400D vector
      │
      ▼
[Parallel Storage]
      │
      ├──▶ [SQLite] ─────▶ Memory metadata
      │
      └──▶ [Qdrant L2] ──▶ 400D vector
      │
      ▼
[Connection Creator]
      │
      └─── Create sequential links
```

### 2. Search Query Flow

```
Query Text
      │
      ▼
[ONNX Encoder] ─────▶ 384D embedding
      │
      ▼
[Dimension Analyzer] ─▶ 16D features
      │
      ▼
[Vector Manager] ────▶ 400D query vector
      │
      ▼
[Qdrant Search]
      │
      ├─── L0: Concepts (if Phase 2+)
      ├─── L1: Contexts (if Phase 4+)
      └─── L2: Episodes
      │
      ▼
[Result Aggregation]
      │
      ▼
[Memory Repository] ─▶ Full memory details
      │
      ▼
Response
```

### 3. Activation Spreading Flow (Phase 2)

```
Initial Search Results
      │
      ▼
[Phase 1: L0 Search]
      │
      ├─── Query concepts
      └─── Get top matches
      │
      ▼
[Phase 2: BFS Spreading]
      │
      ├─── Initialize queue
      ├─── Track paths
      └─── Apply decay
      │
      ▼
[For each activation]
      │
      ├──▶ [Connection Repo] ──▶ Get connections
      │
      ├──▶ [Calculate decay] ──▶ strength * 0.8^depth
      │
      └──▶ [Classify] ────────▶ Core/Contextual/Peripheral
      │
      ▼
[Activation Result]
```

## Component Interfaces

### 1. Memory Extractor
```python
class MemoryExtractor:
    def extract(self, transcript: str, meeting_id: str) -> List[Memory]:
        """
        Input: Raw transcript text
        Output: List of Memory objects with:
               - content: extracted text
               - memory_type: classification
               - speaker: identified speaker
               - timestamp_ms: position in conversation
        """
```

### 2. ONNX Encoder
```python
class ONNXEncoder:
    def encode(self, text: str) -> np.ndarray:
        """
        Input: Text string
        Output: 384-dimensional normalized embedding
        Performance: <100ms per encoding
        """
```

### 3. Dimension Analyzer
```python
class DimensionAnalyzer:
    def extract(self, text: str, context: Dict = None) -> np.ndarray:
        """
        Input: Text and optional context
        Output: 16-dimensional feature vector [0,1]
        Components:
            [0:4]   - Temporal dimensions
            [4:7]   - Emotional dimensions  
            [7:10]  - Social dimensions
            [10:13] - Causal dimensions
            [13:16] - Evolutionary dimensions
        """
```

### 4. Vector Manager
```python
class VectorManager:
    def compose_vector(self, embedding: np.ndarray, features: np.ndarray) -> np.ndarray:
        """
        Input: 384D embedding + 16D features
        Output: 400D composed vector
        """
    
    def decompose_vector(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Input: 400D vector
        Output: (384D embedding, 16D features)
        """
```

### 5. Storage Interfaces

#### SQLite Repositories
```python
class MemoryRepository:
    async def create(self, memory: Memory) -> str
    async def get_by_id(self, memory_id: str) -> Optional[Memory]
    async def get_by_meeting(self, meeting_id: str) -> List[Memory]
    async def update_access(self, memory_id: str) -> None

class ConnectionRepository:
    async def create_connection(self, source: str, target: str, type: ConnectionType) -> None
    async def get_connections(self, memory_id: str) -> List[Dict]
```

#### Qdrant Vector Store
```python
class QdrantVectorStore:
    async def store_memory(self, id: str, vector: np.ndarray, level: int, metadata: Dict) -> None
    async def search(self, vector: np.ndarray, level: int, limit: int) -> List[Tuple[str, float, Dict]]
    async def search_all_levels(self, vector: np.ndarray) -> Dict[int, List[Results]]
```

## Integration Patterns

### 1. Async/Await Pattern
All I/O operations use async/await for non-blocking execution:
```python
async def process_memory(memory: Memory):
    # Parallel operations
    embedding_task = asyncio.create_task(encoder.encode(memory.content))
    dimension_task = asyncio.create_task(analyzer.extract(memory.content))
    
    embedding = await embedding_task
    dimensions = await dimension_task
    
    vector = vector_manager.compose(embedding, dimensions)
    
    # Parallel storage
    await asyncio.gather(
        memory_repo.create(memory),
        vector_store.store_memory(memory.id, vector)
    )
```

### 2. Dependency Injection
Components are injected for testability:
```python
class MeetingIngestionPipeline:
    def __init__(
        self,
        memory_repo: MemoryRepository,
        vector_store: QdrantVectorStore,
        encoder: ONNXEncoder,
        # ... other dependencies
    ):
        self.memory_repo = memory_repo
        self.vector_store = vector_store
        self.encoder = encoder
```

### 3. Error Propagation
Errors bubble up with context:
```python
try:
    result = await pipeline.ingest_meeting(transcript)
except MemoryExtractionError as e:
    logger.error(f"Extraction failed: {e}")
    raise HTTPException(status_code=400, detail=str(e))
except VectorStorageError as e:
    logger.error(f"Storage failed: {e}")
    raise HTTPException(status_code=500, detail="Storage error")
```

### 4. Batch Processing
Optimize for bulk operations:
```python
# Instead of:
for memory in memories:
    await process_memory(memory)

# Use:
await asyncio.gather(*[process_memory(m) for m in memories])
```

## Configuration Flow

### Environment Variables
```bash
# .env file
DATABASE_PATH=data/cognitive.db
QDRANT_HOST=localhost
QDRANT_PORT=6333
MODEL_PATH=models/all-MiniLM-L6-v2.onnx
LOG_LEVEL=INFO
```

### Configuration Loading
```python
# src/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_path: str = "data/cognitive.db"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    model_path: str = "models/all-MiniLM-L6-v2.onnx"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

## Testing Integration Points

### 1. Unit Test Boundaries
Test each component in isolation:
```python
# Test encoder separately
def test_encoder():
    encoder = ONNXEncoder("mock_model.onnx")
    embedding = encoder.encode("test text")
    assert embedding.shape == (384,)

# Test dimension analyzer separately  
def test_dimension_analyzer():
    analyzer = DimensionAnalyzer()
    features = analyzer.extract("urgent task")
    assert features[0] > 0.8  # High urgency
```

### 2. Integration Test Points
Test component interactions:
```python
async def test_vector_pipeline():
    # Test encoder → analyzer → manager flow
    encoder = ONNXEncoder()
    analyzer = DimensionAnalyzer()
    manager = VectorManager()
    
    embedding = encoder.encode("test")
    features = analyzer.extract("test")
    vector = manager.compose(embedding, features)
    
    assert vector.shape == (400,)
```

### 3. End-to-End Test
Test complete flows:
```python
async def test_ingestion_flow():
    # Test API → Pipeline → Storage
    response = client.post("/ingest", json={
        "transcript": "Test meeting content",
        "title": "Test Meeting"
    })
    
    assert response.status_code == 200
    
    # Verify storage
    meeting_id = response.json()["meeting_id"]
    memories = await memory_repo.get_by_meeting(meeting_id)
    assert len(memories) > 0
```

## Performance Considerations

### 1. Bottleneck Points
- **ONNX Encoding**: Cache frequently encoded texts
- **Vector Storage**: Batch inserts when possible
- **Connection Queries**: Index on source_id and target_id

### 2. Optimization Strategies
```python
# Caching
@lru_cache(maxsize=1000)
def encode_cached(text: str) -> np.ndarray:
    return encoder.encode(text)

# Batching
async def store_vectors_batch(vectors: List[Tuple[str, np.ndarray]]):
    points = [
        PointStruct(id=id, vector=vec.tolist())
        for id, vec in vectors
    ]
    client.upsert(collection_name="L2", points=points)
```

### 3. Monitoring Points
- Encoding latency (target: <100ms)
- Storage latency (target: <50ms)
- End-to-end latency (target: <2s)
- Memory usage
- Cache hit rates

## Common Integration Issues

### 1. Vector Dimension Mismatch
```python
# Problem: Expecting 384D, got 400D
# Solution: Check if vector is already composed
if vector.shape == (400,):
    semantic, features = vector_manager.decompose(vector)
```

### 2. Async Context Errors
```python
# Problem: "RuntimeError: no running event loop"
# Solution: Use asyncio.run() or proper async context
asyncio.run(main())
```

### 3. Connection Pool Exhaustion
```python
# Problem: Too many database connections
# Solution: Use connection pooling
db = DatabaseConnection(pool_size=10)
```

### 4. Memory Leaks
```python
# Problem: Cache grows unbounded
# Solution: Use LRU cache with size limit
@lru_cache(maxsize=1000)
```

## Deployment Integration

### Docker Compose Services
```yaml
services:
  api:
    depends_on:
      - qdrant
      - init-db
    environment:
      - QDRANT_HOST=qdrant
      
  qdrant:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/"]
      
  init-db:
    command: python scripts/init_db.py
```

### Service Dependencies
1. Qdrant must be healthy before API starts
2. Database must be initialized before API starts
3. Models must be downloaded before encoding

## Future Integration Points

### Phase 2: Activation Spreading
- Hook into search results
- Add graph traversal after initial search
- Maintain activation state during request

### Phase 3: Bridge Discovery  
- Parallel processing with activation
- Cache bridge results
- Integrate with search response

### Phase 4: Consolidation
- Background task integration
- Scheduled job coordination
- Parent-child relationship management

### Phase 5: Production
- Load balancer integration
- Cache layer (Redis)
- Monitoring integration (Prometheus)