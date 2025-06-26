# Implementation Guide - Cognitive Meeting Intelligence

> **Navigation**: [Home](README.md) → Implementation Guide  
> **Related**: [Architecture](docs/architecture/system-overview.md) | [API Docs](docs/api/endpoints.md) | [Testing](docs/development/testing.md)

## 🎯 Purpose

This guide provides the authoritative implementation roadmap for the Cognitive Meeting Intelligence system. It consolidates all technical decisions, removes conflicts, and provides a clear day-by-day path to build the MVP.

## 📋 MVP Scope (Week 1)

### Core Features
- **400D Vector Architecture**: 384D semantic + 16D cognitive dimensions
- **3-Tier Qdrant Storage**: L0 (concepts), L1 (contexts), L2 (episodes)
- **6 Memory Types**: decision, action, idea, issue, question, context
- **Dimension Extraction**:
  - Real: Temporal (4D) + Emotional (3D) = 7D implemented
  - Placeholder: Social (3D) + Causal (3D) + Evolutionary (3D) = 9D return 0.5
- **Basic Pipeline**: Ingest → Extract → Embed → Store → Search
- **Simple API**: Health, Ingest, Search endpoints

### Deferred Features
- Authentication & authorization
- Activation spreading
- Bridge discovery
- Memory consolidation
- Advanced UI
- Real-time processing

## 🚀 Implementation Order

### Day 1: Core Models & Database
**Goal**: Establish data foundation

**Tasks**:
1. Create `src/models/entities.py` with all dataclasses
2. Create `src/storage/sqlite/schema.sql` with 5 tables
3. Create `src/storage/sqlite/connection.py` for DB management
4. Write `scripts/init_db.py` to initialize database
5. Create unit tests in `tests/unit/test_models.py`

**Success Criteria**:
- ✅ All models have proper type hints and defaults
- ✅ Database creates without errors
- ✅ Can insert and retrieve test data
- ✅ All model tests pass

**Key Files**:
```python
# src/models/entities.py
@dataclass
class Memory:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    meeting_id: str
    content: str
    speaker: Optional[str] = None
    timestamp_ms: int = 0
    memory_type: MemoryType = MemoryType.CONTEXT
    content_type: ContentType = ContentType.GENERAL
    level: int = 2  # L2 by default
    importance_score: float = 0.5
    # ... etc
```

### Day 2: Embeddings Infrastructure
**Goal**: Set up ONNX-based text embeddings

**Tasks**:
1. Write `scripts/download_model.py` to fetch all-MiniLM-L6-v2
2. Create `src/embedding/onnx_encoder.py` with caching
3. Implement batch encoding support
4. Add performance benchmarks
5. Create tests in `tests/unit/test_encoder.py`

**Success Criteria**:
- ✅ Model downloads and converts to ONNX
- ✅ Single encoding <100ms
- ✅ Produces normalized 384D vectors
- ✅ Cache improves performance 10x
- ✅ All encoder tests pass

### Day 3: Vector Management & Dimensions
**Goal**: Implement 400D vector composition

**Tasks**:
1. Create `src/embedding/vector_manager.py` for composition
2. Implement `src/extraction/dimensions/temporal.py` (4D)
3. Implement `src/extraction/dimensions/emotional.py` (3D) with VADER
4. Create placeholder extractors for remaining 9D
5. Create `src/extraction/dimensions/analyzer.py` to orchestrate
6. Write comprehensive tests

**Success Criteria**:
- ✅ Compose 384D + 16D = 400D correctly
- ✅ Temporal detects urgency keywords
- ✅ Emotional uses VADER successfully
- ✅ All dimensions in [0, 1] range
- ✅ Total extraction <50ms

### Day 4: Storage Layer
**Goal**: Hybrid storage system (SQLite + Qdrant)

**Tasks**:
1. Write `scripts/init_qdrant.py` for 3-tier setup
2. Create `src/storage/qdrant/vector_store.py` wrapper
3. Implement all SQLite repositories
4. Add connection pooling
5. Create integration tests

**Success Criteria**:
- ✅ All 3 Qdrant collections created
- ✅ HNSW parameters optimized
- ✅ Vector storage and retrieval works
- ✅ SQLite handles graph relationships efficiently
- ✅ Repositories handle concurrent access
- ✅ 100% repository test coverage

### Day 5: Extraction Pipeline
**Goal**: Extract structured memories from transcripts

**Tasks**:
1. Create `src/extraction/memory_extractor.py`
2. Implement pattern matching for 6 memory types
3. Add speaker identification
4. Create `src/pipeline/ingestion.py` to orchestrate
5. Add connection creation logic
6. Write integration tests

**Success Criteria**:
- ✅ Extract 10+ memories from sample
- ✅ Correctly classify types
- ✅ Identify speakers accurately
- ✅ Create sequential connections
- ✅ Process 1-hour transcript <2s

### Day 6-7: API & Integration
**Goal**: RESTful API with full pipeline

**Tasks**:
1. Create `src/api/main.py` with FastAPI
2. Implement `/health`, `/ingest`, `/search` endpoints
3. Add Pydantic models for validation
4. Create `docker-compose.yml` for services
5. Write API tests
6. Create `Makefile` for common tasks
7. Full end-to-end testing

**Success Criteria**:
- ✅ API starts without errors
- ✅ Health check returns 200
- ✅ Ingest processes transcript fully
- ✅ Search returns relevant results
- ✅ All integration tests pass
- ✅ API docs auto-generated

## 📁 Repository Structure

See complete structure in [README.md#repository-structure](README.md#repository-structure)

## 🧪 Testing Strategy

### Test Organization
```
tests/
├── unit/           # Test individual components
├── integration/    # Test component interactions  
├── performance/    # Benchmark performance
└── fixtures/       # Shared test data
```

### Running Tests
```bash
# After each component
pytest tests/unit/test_<component>.py -v

# Integration tests
pytest tests/integration/ -v

# Full test suite with coverage
pytest --cov=src --cov-report=html

# Performance benchmarks
pytest tests/performance/ -v --benchmark-only
```

## ⚡ Performance Targets

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Text encoding | <100ms | Single text |
| Dimension extraction | <50ms | All 16D |
| Vector storage | <20ms | Single vector |
| Memory extraction | 10-15/sec | From transcript |
| Full pipeline | <2s | 1-hour transcript |
| Search query | <200ms | 10k memories |

## 🔧 Configuration

### Environment Variables
```bash
# .env
DATABASE_URL=sqlite:///./data/memories.db
QDRANT_HOST=localhost
QDRANT_PORT=6333
ONNX_MODEL_PATH=models/all-MiniLM-L6-v2
LOG_LEVEL=INFO
CACHE_SIZE=10000
```

### Key Settings
```yaml
# config/default.yaml
qdrant:
  l0_collection: cognitive_concepts
  l1_collection: cognitive_contexts  
  l2_collection: cognitive_episodes
  
extraction:
  min_memory_length: 10
  max_memory_length: 500
  
dimensions:
  temporal:
    urgency_keywords: ["urgent", "asap", "immediately"]
  emotional:
    use_vader: true
```

## 🚦 Quality Checklist

Before committing any code:
- [ ] Run `black src/ tests/` for formatting
- [ ] Run `flake8 src/ tests/` for linting
- [ ] Run `mypy src/` for type checking
- [ ] Run relevant unit tests
- [ ] Update/add documentation
- [ ] Check performance targets

## 📝 Common Patterns

### Repository Pattern
```python
class MemoryRepository:
    def __init__(self, db: DatabaseConnection):
        self.db = db
    
    async def create(self, memory: Memory) -> str:
        # Implementation
```

### Engine Pattern
```python
class ExtractionEngine:
    def __init__(self, config: Config):
        self.config = config
    
    async def process(self, text: str) -> List[Memory]:
        # Implementation
```

### Async Context Manager
```python
async with db.get_connection() as conn:
    # Database operations
```

## 🐛 Troubleshooting

### ONNX Model Issues
- Ensure `onnxruntime` version matches requirements
- Check model path in config
- Verify model file integrity

### Qdrant Connection
- Confirm Docker container running: `docker ps`
- Check port 6333 is accessible
- Review Qdrant logs: `docker logs qdrant`

### Slow Performance
- Check embedding cache is enabled
- Verify batch operations are used
- Profile with `python -m cProfile`

## 📚 References

- **Technical Spec**: [docs/architecture/system-overview.md](docs/architecture/system-overview.md)
- **API Documentation**: [docs/api/endpoints.md](docs/api/endpoints.md)
- **Testing Guide**: [docs/development/testing.md](docs/development/testing.md)
- **Deployment**: [docs/development/deployment.md](docs/development/deployment.md)

## 🎯 Next Steps

After completing the MVP:

### Phase 2 - Week 2: Activation Spreading
**Goal**: Implement two-phase BFS cognitive search

**Day 8-9: Core Algorithm**
- [ ] Create `src/cognitive/activation/engine.py`
- [ ] Implement two-phase BFS algorithm
- [ ] Add activation decay functions
- [ ] Create activation visualizer
- [ ] Target: <500ms for 10k nodes

**Day 10-11: Integration**
- [ ] Add `/api/v2/cognitive-search` endpoint
- [ ] Implement result explanation generation
- [ ] Add activation path tracking
- [ ] Create performance benchmarks
- [ ] Write comprehensive tests

**Day 12-14: Optimization**
- [ ] Add activation caching layer
- [ ] Implement parallel BFS phases
- [ ] Optimize memory access patterns
- [ ] Add query result caching
- [ ] Performance tuning for scale

### Phase 3 - Week 3: Bridge Discovery
**Goal**: Find non-obvious connections via distance inversion

**Day 15-16: Bridge Algorithm**
- [ ] Create `src/cognitive/bridges/engine.py`
- [ ] Implement distance inversion logic
- [ ] Add bridge strength calculation
- [ ] Create bridge validator
- [ ] Target: <1s for bridge discovery

**Day 17-18: Serendipity Features**
- [ ] Add surprise score calculation
- [ ] Implement bridge explanation
- [ ] Create bridge ranking system
- [ ] Add context preservation
- [ ] Build bridge cache

**Day 19-21: UI & Visualization**
- [ ] Add `/api/v2/discover-bridges` endpoint
- [ ] Create bridge visualization data
- [ ] Implement interactive exploration
- [ ] Add bridge filtering options
- [ ] Write bridge discovery tests

### Phase 4 - Week 4: Memory Consolidation
**Goal**: DBSCAN clustering for semantic memory creation

**Day 22-23: Clustering Engine**
- [ ] Create `src/cognitive/consolidation/engine.py`
- [ ] Implement DBSCAN for 400D vectors
- [ ] Add cluster quality metrics
- [ ] Create consolidation scheduler
- [ ] Target: Process 10k memories in <30s

**Day 24-25: Consolidation Logic**
- [ ] Implement L2 → L1 consolidation
- [ ] Add L1 → L0 concept extraction
- [ ] Create parent-child linking
- [ ] Implement decay rate updates
- [ ] Add consolidation triggers

**Day 26-28: Automation & Testing**
- [ ] Add background consolidation worker
- [ ] Implement consolidation API endpoints
- [ ] Create consolidation monitoring
- [ ] Add rollback mechanisms
- [ ] Comprehensive testing suite

### Phase 5 - Week 5: Production Hardening
**Goal**: Scale, security, and deployment readiness

**Day 29-30: Performance & Scale**
- [ ] Add connection pooling everywhere
- [ ] Implement request queuing
- [ ] Add circuit breakers
- [ ] Create load testing suite
- [ ] Optimize for 100k+ memories

**Day 31-32: Security & Auth**
- [ ] Add JWT authentication
- [ ] Implement role-based access
- [ ] Add rate limiting
- [ ] Create API key management
- [ ] Security audit

**Day 33-35: Deployment & Monitoring**
- [ ] Create production Dockerfile
- [ ] Add Kubernetes manifests
- [ ] Implement health monitoring
- [ ] Add Prometheus metrics
- [ ] Create deployment guide
- [ ] Set up CI/CD pipeline

---

> **For AI Assistants**: This is your primary guide. Start here, follow the day-by-day implementation, and reference other docs as needed. Each component builds on the previous - don't skip steps!