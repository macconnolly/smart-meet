# Project Architecture Patterns

## Overall Architecture
- **Layered Architecture**: Clear separation of concerns
  - API Layer → Cognitive Layer → Embedding Layer → Storage Layer
  - Each layer has defined interfaces and responsibilities
  - Dependency injection for testability

## Core Design Patterns

### Repository Pattern
- **Purpose**: Abstract database operations
- **Implementation**: `MemoryRepository`, `MeetingRepository`
- **Benefits**: Swappable storage backends, testable
```python
class MemoryRepository:
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
    
    async def create(self, memory: Memory) -> str:
        # Implementation
    
    async def get_by_id(self, memory_id: str) -> Optional[Memory]:
        # Implementation
```

### Engine Pattern
- **Purpose**: Encapsulate complex algorithms
- **Implementation**: `ActivationEngine`, `BridgeDiscoveryEngine`, `ConsolidationEngine`
- **Benefits**: Algorithm isolation, configurable behavior
```python
class ActivationEngine:
    def __init__(self, memory_repo, vector_store):
        self.memory_repo = memory_repo
        self.vector_store = vector_store
        # Configuration
        
    async def spread_activation(self, query_vector) -> ActivationResult:
        # Two-phase BFS implementation
```

### Factory Pattern
- **Purpose**: Object creation with complex initialization
- **Implementation**: Memory creation, vector composition
- **Example**: `VectorManager.compose_vector()`

### Pipeline Pattern
- **Purpose**: Multi-stage processing
- **Implementation**: `MeetingIngestionPipeline`
- **Stages**: Extract → Embed → Store → Connect

### Strategy Pattern
- **Purpose**: Interchangeable algorithms
- **Implementation**: Dimension extractors
- **Example**: Different extractors for temporal, emotional, social dimensions

## Async Architecture
- **Async/Await Throughout**: All I/O operations are async
- **Background Tasks**: Consolidation, lifecycle management
- **Task Scheduling**: `ConsolidationScheduler`, `LifecycleManager`
```python
async def start(self):
    self.running = True
    self.task = asyncio.create_task(self._run_scheduler())
```

## Data Flow Patterns

### Meeting Ingestion Flow
1. Transcript → Memory Extraction
2. Memories → Embedding Generation
3. Embeddings → Vector Storage (Qdrant)
4. Metadata → SQLite Storage
5. Connection Creation

### Query Processing Flow
1. Query → Embedding + Dimensions
2. Direct Search → Initial Results
3. Activation Spreading → Expanded Set
4. Bridge Discovery → Novel Connections
5. Result Aggregation → Response

## Storage Architecture

### Hierarchical Storage (3-Tier)
- **L0 (Concepts)**: Highest abstractions, semantic memories
- **L1 (Contexts)**: Patterns and consolidated memories
- **L2 (Episodes)**: Raw episodic memories

### Dual Storage System
- **Qdrant**: Vector storage and similarity search
- **SQLite**: Metadata, relationships, statistics

## Component Organization

### Module Structure
```
src/
├── core/           # Core utilities, cache
├── models/         # Data models (entities)
├── extraction/     # Memory extraction
│   └── dimensions/ # Dimensional analyzers
├── embedding/      # ONNX encoder, vector manager
├── cognitive/      # Cognitive features
│   ├── activation/ # Activation spreading
│   ├── bridges/    # Bridge discovery
│   └── consolidation/ # Memory consolidation
├── storage/        # Storage layer
│   ├── sqlite/     # SQLite repositories
│   └── qdrant/     # Qdrant client
├── api/           # FastAPI endpoints
└── cli/           # CLI tools

```

## Caching Strategy
- **Query Cache**: LRU with TTL for repeated queries
- **Embedding Cache**: In-memory cache for frequent texts
- **Bridge Cache**: Database cache for discovered bridges

## Error Handling Pattern
- **Graceful Degradation**: Continue with partial results
- **Retry Logic**: For transient failures
- **Logging**: Structured logging at each layer
- **Validation**: Early input validation

## Performance Patterns
- **Batch Processing**: For lifecycle operations
- **Connection Pooling**: Database connections
- **Vector Quantization**: Future optimization
- **Async I/O**: Non-blocking operations

## Security Considerations
- **Input Validation**: Pydantic models
- **SQL Injection Prevention**: Parameterized queries
- **Rate Limiting**: API endpoint protection
- **Authentication**: JWT tokens (planned)