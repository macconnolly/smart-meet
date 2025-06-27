# 🚨 MANDATORY AGENT DEVELOPMENT RULES

> **THIS DOCUMENT SUPERSEDES ALL OTHERS FOR DEVELOPMENT PROCESS**

## 🔴 CRITICAL: Current State (2025-06-27)

### What's Actually Done:
- ✅ Code written for Day 1-4 features
- ❌ ZERO tests written or run
- ❌ Test-Driven Development NOT followed
- ❌ Git workflow NOT followed properly
- ❌ Progress updates NOT sequential

### What Must Change NOW:
1. **STOP** all feature development
2. **START** writing tests for existing code
3. **FOLLOW** TDD for all new work
4. **USE** git worktrees and proper branches
5. **CREATE** sequential progress updates

## 📋 Development Process (MUST FOLLOW)

### 1. Session Start Checklist
```bash
# EVERY session MUST start with:
1. pwd                                  # Verify you're in correct worktree
2. source venv/bin/activate            # Activate Python environment
3. git worktree list                   # Check active worktrees
4. git branch --show-current           # Verify correct branch
5. ls docs/progress/                   # Check latest progress number
6. mcp__serena__list_memories          # Review workflow memories
7. TodoWrite([...])                    # Plan session tasks
```

### 2. Test-Driven Development (TDD)
```
For EVERY feature:
1. WRITE TEST FIRST
   - Test fails (RED)
   - Commit test file
   
2. WRITE MINIMAL CODE
   - Make test pass (GREEN)
   - Commit implementation
   
3. REFACTOR
   - Improve code
   - Tests still pass
   - Commit improvements
```

### 3. Git Workflow
```bash
# For new feature:
git checkout -b impl/D{day}-{task}-{description}
# Example: git checkout -b impl/D1-001-create-models

# Commit format:
git commit -m "test: Add test for Memory model creation [IMPL-D1-001]"
git commit -m "feat: Implement Memory model [IMPL-D1-001]"
git commit -m "refactor: Optimize Memory validation [IMPL-D1-001]"
```

### 4. Progress Updates
```bash
# Check latest:
ls docs/progress/

# Create next:
vim docs/progress/009_test_implementation_start.md

# Follow template:
mcp__serena__read_memory progress_update_format
```

## 🌳 Git Worktree Strategy

### Current Worktrees:
1. **Main** (`/mnt/c/Users/EL436GA/dev/meet`)
   - Role: Overseer, integration, checklist updates
   - Branch: `main`
   - Venv: `source venv/bin/activate`

2. **Tests** (`worktree-tests`)
   - Role: Write ALL tests FIRST
   - Branch: `feature/test-implementation`
   - Focus: Unit tests, integration tests, performance tests
   - Venv: `cd worktree-tests && source venv/bin/activate`

3. **Day 1** (`worktree-day1`)
   - Role: Implement Day 1 features (models, database)
   - Branch: `feature/day1-implementation`
   - Focus: Models, database, repositories
   - Venv: `cd worktree-day1 && source venv/bin/activate`

4. **Day 2** (`worktree-day2`)
   - Role: Implement Day 2 features (embeddings)
   - Branch: `feature/day2-implementation`
   - Focus: ONNX encoder, vector manager
   - Venv: `cd worktree-day2 && source venv/bin/activate`

5. **Day 3** (`worktree-day3`)
   - Role: Implement Day 3 features (dimensions)
   - Branch: `feature/day3-implementation`
   - Focus: Dimension extractors, vector composition
   - Venv: `cd worktree-day3 && source venv/bin/activate`

### Worktree Commands:
```bash
# Switch between worktrees:
cd worktree-tests
cd worktree-day1-enhance
cd ../meet  # back to main

# Keep worktrees in sync:
git fetch origin
git merge origin/main
```

## 📝 Task ID Mapping

### Format: IMPL-D{day}-{number:03d}
- Day 1: IMPL-D1-001 to IMPL-D1-005
- Day 2: IMPL-D2-001 to IMPL-D2-006
- Day 3: IMPL-D3-001 to IMPL-D3-006
- Day 4: IMPL-D4-001 to IMPL-D4-005
- Day 5: IMPL-D5-001 to IMPL-D5-005
- Day 6-7: IMPL-D6-001 to IMPL-D6-007

### Test Task Format: TEST-D{day}-{number:03d}
- Example: TEST-D1-001 for testing models

## 🧪 Test Requirements

### Test Files Needed (Priority Order):
1. `tests/unit/test_models.py` - TEST-D1-001
2. `tests/unit/test_connection.py` - TEST-D1-002
3. `tests/unit/test_memory_repository.py` - TEST-D1-003
4. `tests/unit/test_onnx_encoder.py` - TEST-D2-001
5. `tests/unit/test_vector_manager.py` - TEST-D2-002
6. `tests/unit/test_memory_extractor.py` - TEST-D5-001
7. `tests/integration/test_ingestion_pipeline.py` - TEST-D5-002
8. `tests/integration/test_api_endpoints.py` - TEST-D6-001

### Test Coverage Goals:
- Unit tests: >90% coverage
- Integration tests: >80% coverage
- Performance tests: Meet all targets

## 🎯 Immediate Actions Required

### In Test Worktree:
```bash
cd worktree-tests
git checkout -b test/D1-001-model-tests
# Write tests for models
pytest tests/unit/test_models.py  # Should FAIL
git add tests/unit/test_models.py
git commit -m "test: Add unit tests for data models [TEST-D1-001]"
```

### In Day1 Worktree:
```bash
cd worktree-day1-enhance
git checkout -b impl/D1-001-fix-models
# Fix code to make tests pass
pytest tests/unit/test_models.py  # Should PASS
git add src/models/entities.py
git commit -m "fix: Update models to pass tests [IMPL-D1-001]"
```

## 📊 Success Metrics

### A task is ONLY complete when:
1. ✅ Unit tests written and passing
2. ✅ Integration tests written and passing
3. ✅ Code coverage >90%
4. ✅ Performance targets met
5. ✅ Documentation updated
6. ✅ Proper git commits with task IDs
7. ✅ Progress update created

## 🚫 Common Violations to AVOID

1. ❌ Writing code without tests first
2. ❌ Marking tasks complete without tests
3. ❌ Committing without task IDs
4. ❌ Working on wrong branch
5. ❌ Creating non-sequential progress files
6. ❌ Skipping TodoWrite at session start
7. ❌ Not checking existing memories

## 🔍 Verification Commands

```bash
# Check test coverage:
pytest --cov=src --cov-report=html

# Verify all tests pass:
pytest

# Check code quality:
black src/ tests/ --check
flake8 src/ tests/
mypy src/

# Verify commit format:
git log --oneline -5
```

## 📚 Essential Memories to Read

1. `progress_update_format` - How to create progress updates
2. `git_workflow_enforcement` - Detailed git rules
3. `phase1_implementation_tasks` - What needs to be done
4. `project_onboarding` - Project context

## 🎯 THE GOLDEN RULE

**NO CODE WITHOUT TESTS**

If you're writing implementation code without a failing test first, you're doing it WRONG.

---

**Remember**: The goal is not to write code quickly, but to write code that WORKS and is TESTED. Every shortcut taken now creates technical debt that must be paid later.