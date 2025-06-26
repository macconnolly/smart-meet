# TODO to Task ID Mapping Plan

## Overview
Map all skeleton TODO messages to specific IMPL-D*-*** task IDs from TASK_COMPLETION_CHECKLIST.md

## Mapping Strategy

### Day 1: Core Models & Database
- `src/models/entities.py` → IMPL-D1-001
- `src/storage/sqlite/schema.sql` → IMPL-D1-002
- `src/storage/sqlite/connection.py` → IMPL-D1-003
- `scripts/init_db.py` → IMPL-D1-004
- `tests/unit/test_models.py` → IMPL-D1-005

### Day 2: Embeddings Infrastructure
- `src/embedding/onnx_encoder.py` → IMPL-D2-002
  - Model loading → IMPL-D2-002
  - Encoding methods → IMPL-D2-002
  - Cache implementation → IMPL-D2-002
  - Benchmarking → IMPL-D2-003
- `scripts/download_model.py` → IMPL-D2-001
- `tests/unit/test_encoder.py` → IMPL-D2-004

### Day 3: Vector Management & Dimensions
- `src/embedding/vector_manager.py` → IMPL-D3-001
- `src/extraction/dimensions/temporal.py` → IMPL-D3-002
- `src/extraction/dimensions/emotional.py` → IMPL-D3-003
- `src/extraction/dimensions/social.py` → IMPL-D3-004
- `src/extraction/dimensions/causal.py` → IMPL-D3-004
- `src/extraction/dimensions/evolutionary.py` → IMPL-D3-004
- `src/extraction/dimensions/analyzer.py` → IMPL-D3-005
- Tests → IMPL-D3-006

### Day 4: Storage Layer
- `scripts/init_qdrant.py` → IMPL-D4-001
- `src/storage/qdrant/vector_store.py` → IMPL-D4-002
- `src/storage/sqlite/repositories.py` → IMPL-D4-003
- Connection pooling → IMPL-D4-004
- Integration tests → IMPL-D4-005

### Day 5: Extraction Pipeline
- `src/extraction/memory_extractor.py` → IMPL-D5-001
- Pattern matching → IMPL-D5-002
- `src/pipeline/ingestion.py` → IMPL-D5-003
- Connection creation → IMPL-D5-004
- Integration tests → IMPL-D5-005

### Day 6-7: API & Integration
- `src/api/main.py` → IMPL-D6-001
- API endpoints → IMPL-D6-002
- Pydantic models → IMPL-D6-003
- `docker-compose.yml` → IMPL-D6-004
- API tests → IMPL-D6-005
- `Makefile` → IMPL-D6-006
- E2E testing → IMPL-D6-007

### Day 8-9: Activation Spreading
- `src/cognitive/activation/engine.py` → IMPL-D8-001
- Activation scoring → IMPL-D8-002
- Memory classification → IMPL-D8-003
- Project filtering → IMPL-D8-004
- Tests → IMPL-D8-005

### Day 10-11: Activation Integration
- `/api/v2/cognitive-search` → IMPL-D10-001
- Result explanation → IMPL-D10-002
- Activation caching → IMPL-D10-003
- Performance benchmarks → IMPL-D10-004
- Comprehensive tests → IMPL-D10-005

### Day 12-14: Optimization
- Parallel processing → IMPL-D12-001
- Memory access optimization → IMPL-D12-002
- Query caching → IMPL-D12-003
- Performance tuning → IMPL-D12-004
- Scale testing → IMPL-D12-005

### Day 15-16: Bridge Discovery
- `src/cognitive/bridges/engine.py` → IMPL-D15-001
- Bridge scoring → IMPL-D15-002
- Explanation generation → IMPL-D15-003
- Bridge cache → IMPL-D15-004

### Day 17-18: Advanced Dimensions
- Social dimension enhancement → IMPL-D17-001
- Causal dimension implementation → IMPL-D17-002
- Strategic dimension → IMPL-D17-003
- Integration testing → IMPL-D17-004

### Day 19-21: UI & Integration
- Bridge endpoints → IMPL-D19-001
- Visualization data → IMPL-D19-002
- Filtering → IMPL-D19-003
- Tests → IMPL-D19-004

### Day 22-23: Clustering Engine
- `src/cognitive/consolidation/engine.py` → IMPL-D22-001
- Clustering implementation → IMPL-D22-002
- Consolidation scheduler → IMPL-D22-003

### Day 24-25: Memory Lifecycle
- L2→L1 consolidation → IMPL-D24-001
- L1→L0 promotion → IMPL-D24-002
- Lifecycle management → IMPL-D24-003

### Day 26-28: Automation
- Background worker → IMPL-D26-001
- Monitoring → IMPL-D26-002
- Rollback → IMPL-D26-003
- Test suite → IMPL-D26-004

### Day 29-30: Performance & Scale
- Connection pooling → IMPL-D29-001
- Request queuing → IMPL-D29-002
- Circuit breakers → IMPL-D29-003
- Load testing → IMPL-D29-004

### Day 31-32: Security & Auth
- JWT authentication → IMPL-D31-001
- RBAC → IMPL-D31-002
- Rate limiting → IMPL-D31-003
- Security audit → IMPL-D31-004

### Day 33-35: Deployment & Monitoring
- Production Dockerfile → IMPL-D33-001
- Kubernetes manifests → IMPL-D33-002
- Monitoring → IMPL-D33-003
- Deployment guide → IMPL-D33-004
- CI/CD → IMPL-D33-005

## TODO Format Update Pattern

Replace:
```python
# TODO Day 2:
# TODO:
# @TODO:
```

With:
```python
# TODO: [IMPL-D2-002] <specific task description>
```

This makes it easy to track which TODOs correspond to which tasks in TASK_COMPLETION_CHECKLIST.md