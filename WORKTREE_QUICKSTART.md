# 🚀 Worktree Quick Start Guide

## 📂 Your Worktrees Are Ready!

### Start Claude in Different Worktrees:

#### 1. Test Worktree (Write ALL tests)
```bash
cd worktree-tests
claude code .
```
First task: Write tests for Day 1 (models, database, repositories)

#### 2. Day 1 Implementation
```bash
cd worktree-day1
claude code .
```
Wait for tests from worktree-tests, then implement models & database

#### 3. Day 2 Implementation
```bash
cd worktree-day2
claude code .
```
After Day 1 complete, implement embeddings (ONNX encoder, vector manager)

#### 4. Day 3 Implementation
```bash
cd worktree-day3
claude code .
```
After Day 2 complete, implement dimensions & vector composition

## 🔧 Coordination Script Commands

```bash
# Check status of all worktrees
./worktree-sync.sh status

# See TDD workflow guide
./worktree-sync.sh workflow

# Check what tests need writing
./worktree-sync.sh check

# Share tests to implementation
./worktree-sync.sh share worktree-tests worktree-day1

# Run tests in a worktree
./worktree-sync.sh test worktree-day1
```

## 🎯 Next Steps

1. **Start here**: `cd worktree-tests && claude code .`
2. Write failing tests for Day 1 (TEST-D1-001 through TEST-D1-005)
3. Share tests to Day 1 worktree
4. Fix implementation in Day 1 worktree
5. Update checklist in main

## 📋 Task IDs by Day

**Day 1**: IMPL-D1-001 to IMPL-D1-005 (models, database)
**Day 2**: IMPL-D2-001 to IMPL-D2-006 (embeddings)
**Day 3**: IMPL-D3-001 to IMPL-D3-006 (dimensions)

All worktrees are synced and ready to go!