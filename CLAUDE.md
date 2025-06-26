# CLAUDE.md - AI Assistant Context for Cognitive Meeting Intelligence

## 🧠 Project Overview
You are working on a **Cognitive Meeting Intelligence System** that transforms meeting transcripts into a queryable memory network using cognitive science principles. The system uses activation spreading and bridge discovery algorithms to find insights.

## 🎯 Core Concepts You Must Understand

### 1. Memory Types
- **Episodic Memory (L2)**: Raw meeting content, decays quickly (0.1 rate)
- **Semantic Memory (L1)**: Consolidated patterns, decays slowly (0.01 rate)  
- **Concepts (L0)**: Highest abstractions, rarely decay

### 2. Vector Architecture (400D)
- **384D**: Semantic embedding from all-MiniLM-L6-v2 (ONNX)
- **16D**: Dimensional features:
  - Temporal (4D): urgency, deadline, sequence, duration
  - Emotional (3D): sentiment, intensity, confidence (VADER)
  - Social (3D): authority, audience, interaction
  - Causal (3D): cause-effect relationships
  - Evolutionary (3D): change over time

### 3. Cognitive Algorithms
- **Activation Spreading**: Two-phase BFS that mimics human memory recall
- **Bridge Discovery**: Distance inversion to find serendipitous connections
- **Memory Consolidation**: DBSCAN clustering to create semantic memories

## 📁 Project Structure
```
src/
├── models/          # Data models (Memory, Meeting, Connection)
├── extraction/      # Memory extraction from transcripts
│   └── dimensions/  # 16D feature extractors
├── embedding/       # ONNX encoder & vector management
├── cognitive/       # Core algorithms
│   ├── activation/  # Spreading activation
│   ├── bridges/     # Bridge discovery
│   └── consolidation/ # Memory consolidation
├── storage/         # Dual storage system
│   ├── sqlite/      # Metadata & relationships
│   └── qdrant/      # Vector storage (3 tiers)
└── api/            # FastAPI endpoints
```

## 🛠️ CRITICAL: Session Task Management

### TodoWrite/TodoRead Usage (REQUIRED)
**You MUST use TodoWrite at the start of ANY implementation session to:**
1. Break down complex tasks into manageable steps
2. Track progress throughout your work
3. Ensure nothing is forgotten
4. Give visibility to the user

**Example Session Start:**
```python
# User: "Implement the ONNX encoder with caching"
# You should immediately:
TodoWrite([
    {"id": "1", "content": "Download and setup ONNX model", "status": "pending", "priority": "high"},
    {"id": "2", "content": "Implement ONNXEncoder class", "status": "pending", "priority": "high"},
    {"id": "3", "content": "Add LRU caching", "status": "pending", "priority": "medium"},
    {"id": "4", "content": "Write unit tests", "status": "pending", "priority": "medium"},
    {"id": "5", "content": "Run performance benchmarks", "status": "pending", "priority": "low"}
])
```

**Update todos as you work:**
- Mark "in_progress" when starting a task
- Mark "completed" IMMEDIATELY when done
- Add new todos if you discover additional work

## 🔧 Key Commands & Workflows

### Development Setup
```bash
# Always activate virtual environment first
source venv/bin/activate  # Windows: venv\Scripts\activate

# Start services
docker-compose up -d     # Qdrant vector database
python scripts/init_db.py # Initialize SQLite
python scripts/init_qdrant.py # Create collections
```

### Running Tests
```bash
# Before ANY code changes
pytest tests/unit/test_<module>.py  # Test specific module
pytest --cov=src --cov-report=html  # Full coverage
python tests/test_phase1_integration.py  # Integration tests
```

### Code Quality (MUST run before commits)
```bash
black src/ tests/ --line-length 100
flake8 src/ tests/ --max-line-length 100
mypy src/
```

### API Development
```bash
# Development with auto-reload
uvicorn src.api.cognitive_api:app --reload

# Test endpoints
curl -X POST http://localhost:8000/api/v2/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was decided about caching?"}'
```

## ⚡ Performance Targets
- Memory extraction: 10-15/second
- Embedding generation: <100ms
- Full cognitive query: <2s
- Support 10K+ memories

## 🏗️ Current Implementation Status

> **📢 IMPORTANT**: Navigation hierarchy for implementation:
> 1. Start here: [AGENT_START_HERE.md](docs/AGENT_START_HERE.md)
> 2. Day roadmap: [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) 
> 3. All tasks: [TASK_COMPLETION_CHECKLIST.md](TASK_COMPLETION_CHECKLIST.md)

### 📋 Task Tracking System
- **129 granular implementation tasks** organized by Phase (1-5) and Day (1-35)
- Each task has unique ID (e.g., IMPL-D1-001) for tracking
- See [TASK_COMPLETION_CHECKLIST.md](TASK_COMPLETION_CHECKLIST.md) for complete list
- Use TodoWrite/TodoRead for session-level task management

### Current Phase: Project Setup
- Project structure created
- Documentation consolidated
- Ready for Day 1 implementation

### Week 1 Plan (MVP)
- **Day 1**: Core Models & Database (5 tasks)
- **Day 2**: Embeddings Infrastructure (4 tasks)
- **Day 3**: Vector Management & Dimensions (6 tasks)
- **Day 4**: Storage Layer (5 tasks)
- **Day 5**: Extraction Pipeline (5 tasks)
- **Day 6-7**: API & Integration (7 tasks)

**Total Phase 1**: 35 implementation tasks + quality checks

## 💡 Key Patterns & Best Practices

### Repository Pattern
```python
# Always use repositories for data access
memory_repo = MemoryRepository(db_connection)
memory = await memory_repo.get_by_id(memory_id)
```

### Engine Pattern
```python
# Encapsulate algorithms in engines
engine = ActivationEngine(memory_repo, vector_store)
result = await engine.spread_activation(query_vector)
```

### Async Everything
```python
# All I/O operations must be async
async def process_memory(memory: Memory):
    embedding = await encoder.encode(memory.content)
    await vector_store.store(embedding)
```

### Error Handling
```python
# Always handle errors gracefully
try:
    result = await risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    return None
```

## 🐛 Common Issues & Solutions

### Issue: Vector dimension mismatch
```python
# ALWAYS use VectorManager for composition
vector = VectorManager.compose_vector(semantic_embedding, dimensions)
# This ensures 384D + 16D = 400D
```

### Issue: Slow queries
```python
# Check if caching is enabled
query_cache = QueryCache(max_size=1000, ttl_seconds=3600)
# Use batch operations
memories = await memory_repo.get_batch(memory_ids)
```

### Issue: Memory leaks
```python
# Close connections properly
async with db.get_connection() as conn:
    # Operations here
    pass  # Connection auto-closed
```

## 🎨 Code Style Essentials
- Type hints REQUIRED for all functions
- Docstrings for all public methods
- Use dataclasses for models
- Black formatting (100 char lines)
- Async for all I/O operations

## 🚀 Quick Task Templates

### Adding a New Dimension Extractor
```python
# 1. Create in src/extraction/dimensions/
class NewDimensionExtractor:
    def extract(self, text: str) -> np.ndarray:
        features = np.zeros(3)  # Your dimension size
        # Implementation
        return features

# 2. Add to DimensionAnalyzer
self.extractors['new'] = NewDimensionExtractor()

# 3. Update vector size documentation
```

### Adding a New API Endpoint
```python
# 1. Add to src/api/cognitive_api.py
@app.post("/api/v2/new-endpoint")
async def new_endpoint(request: NewRequest):
    # Implementation
    return response

# 2. Create Pydantic models
class NewRequest(BaseModel):
    field: str

# 3. Add tests
async def test_new_endpoint():
    response = await client.post("/api/v2/new-endpoint", json={})
```

## 📊 Database Schema Quick Reference
```sql
memories: id, meeting_id, content, speaker, timestamp, memory_type, 
          content_type, level, qdrant_id, dimensions_json, importance_score,
          decay_rate, access_count, last_accessed, created_at, parent_id

memory_connections: source_id, target_id, connection_strength, 
                   connection_type, created_at, last_activated, activation_count

meetings: id, title, start_time, end_time, participants_json, 
         transcript_path, metadata_json, created_at, processed_at, memory_count
```

## 🔍 Debugging Commands
```bash
# Check Qdrant collections
curl http://localhost:6333/collections

# View SQLite data
sqlite3 data/memories.db "SELECT * FROM memories LIMIT 5;"

# Monitor logs
docker-compose logs -f qdrant

# Profile performance
python -m cProfile -o profile.stats src/api/cognitive_api.py
```

## ⚠️ Critical Rules
1. NEVER commit without running tests and linters
2. ALWAYS use async for database/network operations
3. NEVER store vectors in SQLite (use Qdrant)
4. ALWAYS validate input data with Pydantic
5. NEVER expose internal errors to API responses

## 📚 Key Files to Understand
1. `src/models/entities.py` - Core data models
2. `src/cognitive/activation/engine.py` - Activation algorithm
3. `src/cognitive/bridges/engine.py` - Bridge discovery
4. `src/extraction/ingestion.py` - Meeting processing
5. `src/api/cognitive_api.py` - Main API

## 🆘 When Stuck
1. Check the memories: `mcp__serena__list_memories`
2. Read the roadmap: `docs/roadmap.md`
3. Review tests for examples
4. Check similar implementations in codebase
5. Validate against performance targets

## 🎯 Your Mission
Help implement and improve this cognitive meeting intelligence system while maintaining high code quality, meeting performance targets, and following established patterns. Focus on making the system genuinely useful for capturing and retrieving meeting insights.

Remember: This isn't just storage - it's cognitive intelligence. Every feature should enhance understanding, not just recall.