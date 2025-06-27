# 🎯 Implementation Focus: Days 1-3

## 🚀 Quick Start
```bash
# 1. Test worktree (write tests FIRST)
cd worktree-tests && source venv/bin/activate && claude code .

# 2. Implementation worktrees (after tests)
cd worktree-day1 && source venv/bin/activate && claude code .
cd worktree-day2 && source venv/bin/activate && claude code .
cd worktree-day3 && source venv/bin/activate && claude code .
```

## 📋 Day 1: Core Models & Database (Tasks 1-5)
**Branch**: `feature/day1-implementation`

### Tasks:
1. **IMPL-D1-001**: Create dataclasses in `src/models/entities.py`
   - Memory, Meeting, Project, Connection classes
   - All enums and types
   
2. **IMPL-D1-002**: Database connection in `src/storage/sqlite/connection.py`
   - Async connection management
   - Connection pooling
   
3. **IMPL-D1-003**: Base repository in `src/storage/sqlite/repositories/base.py`
   - Generic CRUD operations
   
4. **IMPL-D1-004**: Memory repository in `src/storage/sqlite/repositories/memory_repository.py`
   - Memory-specific queries
   
5. **IMPL-D1-005**: Initialize schema
   - Run `src/storage/sqlite/schema.sql`

## 📋 Day 2: Embeddings (Tasks 6-11)
**Branch**: `feature/day2-implementation`

### Tasks:
6. **IMPL-D2-001**: ONNX model setup
   - Download model to `data/models/` (SHARED!)
   - Model: all-MiniLM-L6-v2
   
7. **IMPL-D2-002**: ONNX encoder in `src/embedding/onnx_encoder.py`
   - Load and run model
   - Handle 384D output
   
8. **IMPL-D2-003**: Add LRU caching
   - Cache embeddings
   
9. **IMPL-D2-004**: Vector manager in `src/embedding/vector_manager.py`
   - Compose 400D vectors (384D + 16D)
   
10. **IMPL-D2-005**: Model warmup
    - Preload on startup
    
11. **IMPL-D2-006**: Batch encoding
    - Process multiple texts

## 📋 Day 3: Dimensions (Tasks 12-17)
**Branch**: `feature/day3-implementation`

### Tasks:
12. **IMPL-D3-001**: Temporal extractor in `src/extraction/dimensions/temporal_extractor.py`
    - Extract urgency, deadline, sequence, duration (4D)
    
13. **IMPL-D3-002**: Emotional extractor in `src/extraction/dimensions/emotional_extractor.py`
    - Use VADER for sentiment, intensity, confidence (3D)
    
14. **IMPL-D3-003**: Placeholder extractors
    - Social (3D), Causal (3D), Evolutionary (3D)
    - Return 0.5 defaults for now
    
15. **IMPL-D3-004**: Dimension analyzer in `src/extraction/dimensions/dimension_analyzer.py`
    - Coordinate all extractors
    - Return 16D array
    
16. **IMPL-D3-005**: Vector validation
    - Ensure 400D compliance
    
17. **IMPL-D3-006**: Dimension caching
    - Cache computed dimensions

## 🔧 Shared Resources Setup

### Model Directory (SHARED across all worktrees)
```bash
# Create once in main
mkdir -p data/models

# Update all encoder implementations to use:
MODEL_PATH = "/mnt/c/Users/EL436GA/dev/meet/data/models/all-MiniLM-L6-v2.onnx"
```

### Database (SHARED)
```bash
# Create once in main
mkdir -p data
# All worktrees use: data/memories.db
```

## ⚡ TDD Process for Each Task

1. **Write Test First** (in worktree-tests)
   ```bash
   cd worktree-tests
   # Create test file
   # Write failing test
   pytest tests/unit/test_xxx.py  # Should FAIL
   git commit -m "test: Add test for XXX [TEST-D1-001]"
   ```

2. **Share to Implementation**
   ```bash
   cd ..
   ./worktree-sync.sh share worktree-tests worktree-dayX
   ```

3. **Implement** (in worktree-dayX)
   ```bash
   cd worktree-dayX
   # Fix code until test passes
   pytest tests/unit/test_xxx.py  # Should PASS
   git commit -m "feat: Implement XXX [IMPL-D1-001]"
   ```

## 📊 Success Criteria
- All tests passing
- Code coverage >90%
- Performance targets met
- No linting errors

## 🚫 What NOT to Do
- Don't implement without tests
- Don't download models multiple times
- Don't create new documentation
- Don't implement Phase 2 features (cognitive engines)

## 📁 Key Files Only
- `TASK_COMPLETION_CHECKLIST.md` - Track progress
- `IMPLEMENTATION_FOCUS.md` - This guide
- `worktree-sync.sh` - Coordination tool

All other guides are archived.