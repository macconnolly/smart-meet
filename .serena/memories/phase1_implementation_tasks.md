# Phase 1: Complete Foundation (Week 1-1.5)

Build a working end-to-end system with 400D vectors, basic dimension extraction, and 3-tier storage.

## Day 1-2: SQLite Database Foundation

### Task 1: Create Database Schema
- **Where**: `src/storage/sqlite/schema.sql`
- **Create**: All 5 tables (meetings, memories, memory_connections, query_statistics, bridge_cache)
- **Indexes**: Create performance indexes on foreign keys and frequently queried fields
- **Validation**: Run `sqlite3 data/cognitive.db < schema.sql` and verify tables exist

### Task 2: Create Data Models
- **Where**: `src/models/entities.py`
- **Create**: Meeting, Memory, MemoryConnection dataclasses
- **Enums**: MemoryType (6 types), ConnectionType (5 types)
- **Defaults**: Proper default values (importance=0.5, level=2, etc.)
- **Validation**: Unit tests for model creation and field validation

### Task 3: Database Connection Manager
- **Where**: `src/storage/sqlite/connection.py`
- **Create**: Thread-safe DatabaseConnection class with context manager
- **Features**: Connection pooling, foreign key enforcement, row factory
- **Validation**: Test concurrent access and transaction handling

### Task 4: Repository Implementation
- **Where**: `src/storage/sqlite/repositories/`
- **Create**: MemoryRepository, MeetingRepository, ConnectionRepository
- **Methods**: CRUD operations, batch operations, search methods
- **Testing**: 100% test coverage for all repository methods

## Day 3: ONNX Embeddings & Vector Composition

### Task 1: Model Setup
- **Where**: `scripts/setup_model.py`
- **Download**: sentence-transformers/all-MiniLM-L6-v2
- **Convert**: Export to ONNX format with proper input/output names
- **Validation**: Verify model file exists and loads

### Task 2: ONNX Encoder
- **Where**: `src/embedding/onnx_encoder.py`
- **Create**: ONNXEncoder class with caching
- **Performance**: <100ms per encoding
- **Output**: 384D normalized embeddings
- **Testing**: Performance benchmarks and correctness tests

### Task 3: Vector Manager
- **Where**: `src/embedding/vector_manager.py`
- **Create**: Compose 384D + 16D → 400D vectors
- **Methods**: compose_vector(), decompose_vector(), validate_vector()
- **Validation**: Ensure semantic part normalized, features in [0,1]

## Day 4: Qdrant 3-Tier Setup

### Task 1: Docker Configuration
- **Where**: `docker-compose.yml`
- **Setup**: Qdrant service with health checks
- **Volumes**: Persistent storage in ./data/qdrant
- **Validation**: `docker-compose up -d` and verify health endpoint

### Task 2: Collection Initialization
- **Where**: `scripts/init_qdrant.py`
- **Create**: L0_cognitive_concepts, L1_cognitive_contexts, L2_cognitive_episodes
- **HNSW**: Optimized parameters for each level
- **Validation**: Verify all 3 collections created with correct settings

### Task 3: Vector Store Interface
- **Where**: `src/storage/qdrant/vector_store.py`
- **Create**: QdrantVectorStore class
- **Methods**: store_memory(), search(), search_all_levels()
- **Testing**: Storage, retrieval, and filtering tests

## Day 5: Dimension Extractors

### Task 1: Temporal Dimensions (4D)
- **Where**: `src/extraction/dimensions/temporal_extractor.py`
- **Extract**: Urgency, deadline proximity, sequence position, duration relevance
- **Keywords**: Map urgency terms to scores
- **Validation**: Test with various urgency levels

### Task 2: Emotional Dimensions (3D)
- **Where**: `src/extraction/dimensions/emotional_extractor.py`
- **Use**: VADER sentiment analyzer
- **Extract**: Polarity, intensity, confidence
- **Validation**: Test positive/negative/neutral texts

### Task 3: Placeholder Dimensions (9D)
- **Where**: `src/extraction/dimensions/placeholder_extractors.py`
- **Create**: Social (3D), Causal (3D), Evolutionary (3D)
- **Default**: Return 0.5 for all dimensions
- **Future**: Structure for later implementation

### Task 4: Dimension Analyzer
- **Where**: `src/extraction/dimensions/dimension_analyzer.py`
- **Orchestrate**: All extractors to produce 16D vector
- **Validate**: Ensure all values in [0,1]
- **Testing**: Integration tests for full extraction

## Day 6-7: Pipeline Integration & API

### Task 1: Memory Extractor
- **Where**: `src/extraction/memory_extractor.py`
- **Extract**: Split sentences, identify speakers, classify types
- **Patterns**: Regex for decisions, actions, ideas, issues, questions
- **Validation**: Test with sample transcripts

### Task 2: Ingestion Pipeline
- **Where**: `src/pipeline/ingestion_pipeline.py`
- **Process**: Extract → Embed → Compose → Store
- **Parallel**: Use asyncio for concurrent operations
- **Connections**: Create sequential links between memories
- **Performance**: <2s for typical transcript

### Task 3: FastAPI Implementation
- **Where**: `src/api/main.py`
- **Endpoints**: POST /ingest, POST /search, GET /health
- **Models**: Pydantic request/response schemas
- **Error Handling**: Proper HTTP status codes and messages
- **Validation**: Input validation and sanitization

### Task 4: End-to-End Testing
- **Where**: `tests/integration/test_pipeline.py`
- **Test**: Complete ingestion flow
- **Verify**: Memories extracted, vectors stored, search works
- **Performance**: Measure and validate <2s target

## Success Criteria

### Functionality
- ✅ Complete pipeline processes transcripts end-to-end
- ✅ 400D vectors properly composed and stored
- ✅ All 3 Qdrant tiers configured and working
- ✅ Basic API endpoints functional

### Performance
- ✅ Embedding generation <100ms
- ✅ Vector storage <50ms
- ✅ Memory extraction 10-15/second
- ✅ End-to-end processing <2s
- ✅ Search queries <200ms

### Quality
- ✅ Unit test coverage >90%
- ✅ Integration tests passing
- ✅ No critical bugs
- ✅ Clean code following patterns

## Next Phase Preview
With Phase 1 complete, Phase 2 will add:
- Two-phase BFS activation spreading
- Path tracking for transparency
- Core/contextual/peripheral classification
- <500ms performance for 50 activations