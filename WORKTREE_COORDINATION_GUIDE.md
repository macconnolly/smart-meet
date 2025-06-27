# 🎯 Worktree Coordination Guide for TDD Development

## 🚀 Quick Start Commands

### 1. Check Status of All Worktrees
```bash
./worktree-sync.sh status
```

### 2. View TDD Workflow Guide
```bash
./worktree-sync.sh workflow
```

### 3. Check Which Tests Are Needed
```bash
./worktree-sync.sh check
```

## 📂 Starting Claude in Each Worktree

### Option 1: Start Claude in Main (Overseer) - Currently Active
You're here now! Use this for:
- Updating TASK_COMPLETION_CHECKLIST.md
- Managing overall progress
- Coordinating between worktrees
- Running the sync script

### Option 2: Start Claude in Test Worktree
```bash
cd worktree-tests
claude code .
```
Use for:
- Writing ALL tests FIRST (TDD requirement)
- Creating test/D1-XXX branches
- Running `pytest` to verify tests fail initially

### Option 3: Start Claude in Day1 Enhancement Worktree
```bash
cd worktree-day1-enhance
claude code .
```
Use for:
- Fixing code to make tests pass
- Creating impl/D1-XXX branches
- Running tests to verify fixes work

### Option 4: Start Claude in Next Phase Worktree
```bash
cd worktree-next-phase
claude code .
```
Use for:
- Implementing Day 3-4 features (AFTER tests)
- Missing dimension extractors
- Additional repository methods

### Option 5: Start Claude in Documentation Worktree
```bash
cd worktree-docs
claude code .
```
Use for:
- Consolidating documentation
- Making rules clearer for agents
- Improving CLAUDE.md

## 🔄 TDD Development Process

### Step 1: Write Tests First (Test Worktree)
```bash
# In test worktree
cd worktree-tests
./worktree-sync.sh task worktree-tests test 1 001 model-tests

# Write failing tests
# Edit: tests/unit/test_models.py

# Commit when tests fail properly
git add tests/unit/test_models.py
git commit -m "test: Add unit tests for Memory model [TEST-D1-001]"
```

### Step 2: Share Tests to Implementation Worktree
```bash
# From main directory
./worktree-sync.sh share worktree-tests worktree-day1-enhance
```

### Step 3: Fix Code (Day1 Worktree)
```bash
# In day1 worktree
cd worktree-day1-enhance
git merge local/test/D1-001-model-tests

# Fix code to pass tests
# Edit: src/models/entities.py

# Run tests to verify
./worktree-sync.sh test . tests/unit/test_models.py

# Commit when tests pass
git add src/models/entities.py
git commit -m "fix: Update Memory model to pass tests [IMPL-D1-001]"
```

### Step 4: Update Main Branch
```bash
# Back in main
cd /mnt/c/Users/EL436GA/dev/meet
git merge feature/test-implementation
git merge feature/day1-enhancements

# Update checklist
# Edit: TASK_COMPLETION_CHECKLIST.md
git add TASK_COMPLETION_CHECKLIST.md
git commit -m "docs: Update task checklist - completed TEST-D1-001 and IMPL-D1-001"
```

## 🎯 Priority Tasks to Start

Based on current state, start with:

1. **TEST-D1-001**: Write tests for `src/models/entities.py`
   - Location: `tests/unit/test_models.py`
   - Test all dataclasses (Memory, Meeting, Project, etc.)

2. **TEST-D1-002**: Write tests for database connection
   - Location: `tests/unit/test_connection.py`
   - Test `src/storage/sqlite/connection.py`

3. **TEST-D1-003**: Write tests for memory repository
   - Location: `tests/unit/test_memory_repository.py`
   - Test `src/storage/sqlite/repositories/memory_repository.py`

## 🛠️ Useful Sync Commands

### Sync a worktree with main
```bash
./worktree-sync.sh sync worktree-tests
```

### Show differences between worktrees
```bash
./worktree-sync.sh diff worktree-tests worktree-day1-enhance
```

### Run tests in any worktree
```bash
./worktree-sync.sh test worktree-tests
./worktree-sync.sh test worktree-day1-enhance tests/unit/test_models.py
```

## ⚠️ Important Rules

1. **NEVER** write implementation code without failing tests first
2. **ALWAYS** verify tests fail before implementing
3. **ALWAYS** verify tests pass after implementing
4. **ALWAYS** use task IDs in commit messages
5. **ALWAYS** update TASK_COMPLETION_CHECKLIST.md after completing tasks

## 🔍 Current Status

- ✅ Worktrees created and synced
- ✅ Coordination script working
- ❌ No tests written yet (0/129 tasks complete)
- 🎯 Ready to start TDD process

Start with writing tests in the test worktree!