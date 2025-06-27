# 📅 Day-Based Worktree Development Guide

## 🎯 Worktree Organization

### Overview
- **worktree-tests**: Write ALL tests (for all days)
- **worktree-day1**: Day 1 implementation (models, database)
- **worktree-day2**: Day 2 implementation (embeddings)
- **worktree-day3**: Day 3 implementation (dimensions, vector management)

## 🚀 How to Start Claude in Each Worktree

### 1. Test Worktree (Write ALL Tests First)
```bash
cd worktree-tests
claude code .
```
**Purpose**: Write tests for ALL days following TDD
- Day 1 tests: models, database, repositories
- Day 2 tests: ONNX encoder, vector manager
- Day 3 tests: dimension extractors, vector composition
- Integration tests for each day

### 2. Day 1 Worktree (Core Models & Database)
```bash
cd worktree-day1
claude code .
```
**Tasks** (IMPL-D1-001 to IMPL-D1-005):
- Create all dataclasses in src/models/entities.py
- Implement database connection layer
- Create base repository class
- Implement memory repository
- Initialize database schema

### 3. Day 2 Worktree (Embeddings Infrastructure)
```bash
cd worktree-day2
claude code .
```
**Tasks** (IMPL-D2-001 to IMPL-D2-006):
- Download and setup ONNX model
- Implement ONNX encoder with caching
- Create vector manager for 400D composition
- Implement embedding engine
- Setup model warmup
- Create batch processing

### 4. Day 3 Worktree (Dimensions & Vector Management)
```bash
cd worktree-day3
claude code .
```
**Tasks** (IMPL-D3-001 to IMPL-D3-006):
- Implement temporal dimension extractor
- Implement emotional dimension extractor
- Create placeholder extractors (social, causal, evolutionary)
- Implement dimension analyzer
- Create vector validation utilities
- Setup dimension caching

## 🔄 TDD Workflow by Day

### Day 1 Development Process
```bash
# Step 1: Write Day 1 tests (in test worktree)
cd worktree-tests
./worktree-sync.sh task worktree-tests test 1 001 model-tests
# Create tests/unit/test_models.py
# Create tests/unit/test_connection.py
# Create tests/unit/test_memory_repository.py
git add tests/unit/test_*.py
git commit -m "test: Add Day 1 unit tests [TEST-D1-001]"

# Step 2: Share tests to Day 1 worktree
cd /mnt/c/Users/EL436GA/dev/meet
./worktree-sync.sh share worktree-tests worktree-day1

# Step 3: Implement Day 1 (in day1 worktree)
cd worktree-day1
git merge local/test/D1-001-model-tests
# Fix code to make tests pass
./worktree-sync.sh test . tests/unit/test_models.py
git add src/models/entities.py
git commit -m "feat: Implement core data models [IMPL-D1-001]"
```

### Day 2 Development Process
```bash
# Step 1: Write Day 2 tests (in test worktree)
cd worktree-tests
./worktree-sync.sh task worktree-tests test 2 001 encoder-tests
# Create tests/unit/test_onnx_encoder.py
# Create tests/unit/test_vector_manager.py
git commit -m "test: Add Day 2 embedding tests [TEST-D2-001]"

# Step 2: Share to Day 2 worktree
./worktree-sync.sh share worktree-tests worktree-day2

# Step 3: Implement Day 2
cd worktree-day2
# Implement ONNX encoder, vector manager, etc.
```

## 📋 Task Distribution by Day

### Day 1 Tasks (Models & Database)
- IMPL-D1-001: Create Memory dataclass and related models
- IMPL-D1-002: Create database connection layer
- IMPL-D1-003: Create base repository pattern
- IMPL-D1-004: Implement memory repository
- IMPL-D1-005: Initialize database schema

### Day 2 Tasks (Embeddings)
- IMPL-D2-001: Download and setup ONNX model
- IMPL-D2-002: Implement ONNX encoder
- IMPL-D2-003: Add LRU caching to encoder
- IMPL-D2-004: Create vector manager
- IMPL-D2-005: Implement model warmup
- IMPL-D2-006: Create batch encoding

### Day 3 Tasks (Dimensions)
- IMPL-D3-001: Implement temporal extractor
- IMPL-D3-002: Implement emotional extractor
- IMPL-D3-003: Create placeholder extractors
- IMPL-D3-004: Implement dimension analyzer
- IMPL-D3-005: Create vector validation
- IMPL-D3-006: Setup dimension caching

## 🛠️ Useful Commands

### Check worktree status
```bash
./worktree-sync.sh status
```

### See what tests need writing
```bash
./worktree-sync.sh check
```

### Run tests in any worktree
```bash
./worktree-sync.sh test worktree-day1
./worktree-sync.sh test worktree-day2 tests/unit/test_encoder.py
```

### Share changes between worktrees
```bash
# Share tests to implementation worktree
./worktree-sync.sh share worktree-tests worktree-day1

# Share Day 1 completion to Day 2
./worktree-sync.sh share worktree-day1 worktree-day2
```

### Create task branches
```bash
# For tests
./worktree-sync.sh task worktree-tests test 1 001 model-tests

# For implementation
./worktree-sync.sh task worktree-day1 impl 1 001 create-models
```

## ⚡ Quick Start Sequence

1. **Start with tests**: Open Claude in `worktree-tests` and write Day 1 tests
2. **Share to Day 1**: Use sync script to share tests
3. **Implement Day 1**: Open Claude in `worktree-day1` and fix failing tests
4. **Update checklist**: Back in main, update TASK_COMPLETION_CHECKLIST.md
5. **Repeat for Day 2**: Write Day 2 tests, share, implement
6. **Continue pattern**: Each day builds on the previous

## 📊 Current Status
- ✅ Worktrees created for test + Days 1-3
- ✅ Sync script configured
- ❌ No tests written yet
- ❌ No implementation started
- 🎯 Ready to begin TDD process!

## 🔴 Important Rules
1. **ALWAYS** write tests first in the test worktree
2. **NEVER** implement without failing tests
3. **ALWAYS** verify tests pass before moving to next task
4. Each day's implementation depends on previous days
5. Update main branch checklist after each completed task

Start by opening Claude in the test worktree!