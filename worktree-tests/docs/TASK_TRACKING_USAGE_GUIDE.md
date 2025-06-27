# Task Tracking Usage Guide

## Overview

This guide explains how to effectively use the dual-layer task tracking system:
- **Session-level tracking**: Using TodoWrite/TodoRead for immediate task management
- **Project-level tracking**: Using TASK_COMPLETION_CHECKLIST.md for overall progress

## 🎯 Agent Strategy

### Single Agent vs Multiple Agents

#### Single Agent Approach (Recommended)
- **When to use**: For coherent, focused implementation sessions
- **Benefits**: 
  - Maintains context across entire session
  - Reduces handoff overhead
  - Better understanding of interconnected changes
- **Example**: Implementing the entire embedding infrastructure in one session

#### Multiple Agent Approach
- **When to use**: For complex, multi-day features requiring different expertise
- **Benefits**:
  - Specialized knowledge per agent
  - Fresh perspective on problems
  - Parallel development possible
- **Example**: One agent for database schema, another for API endpoints

### Best Practice: Hybrid Approach
```
Day 1: Agent A - Core infrastructure (DB, models)
Day 2: Agent B - Business logic (engines, algorithms)  
Day 3: Agent A - Integration and testing
```

## 📋 Mapping TODOs to Implementation Tasks

### 1. Starting a Session
When beginning work, immediately create TODOs from the implementation checklist:

```python
# User: "Implement the ONNX encoder (Day 2 tasks)"

# Agent should check TASK_COMPLETION_CHECKLIST.md and create:
TodoWrite([
    {"id": "IMPL-D2-001", "content": "Download ONNX model files", "status": "pending", "priority": "high"},
    {"id": "IMPL-D2-002", "content": "Create ONNXEncoder class", "status": "pending", "priority": "high"},
    {"id": "IMPL-D2-003", "content": "Implement encode_batch method", "status": "pending", "priority": "high"},
    {"id": "IMPL-D2-004", "content": "Add caching with functools.lru_cache", "status": "pending", "priority": "medium"}
])
```

### 2. Task ID Mapping
Always use the official task IDs from TASK_COMPLETION_CHECKLIST.md:
- Format: `IMPL-D{day}-{number}` (e.g., IMPL-D1-001)
- This enables cross-reference between session work and project tracking

### 3. Discovering Additional Tasks
If you discover work not in the checklist:

```python
# While implementing, you realize you need error handling
TodoWrite([
    {"id": "IMPL-D2-001a", "content": "Add error handling for model loading", "status": "pending", "priority": "high"},
    {"id": "IMPL-D2-001b", "content": "Create ModelLoadError exception", "status": "pending", "priority": "medium"}
])
# Use suffixes (a, b, c) for subtasks
```

## 📊 Progress Tracking Methods

### 1. Session Progress
Use TodoRead frequently to check status:

```python
# At start of session
todos = TodoRead()  # See what's pending

# After completing a task
TodoWrite([
    {"id": "IMPL-D2-001", "content": "Download ONNX model files", "status": "completed", "priority": "high"},
    # ... other todos remain pending
])

# Mid-session check
todos = TodoRead()  # Review progress
```

### 2. Daily Progress Report
At end of each session, provide summary:

```markdown
## Session Summary
- Completed: IMPL-D2-001, IMPL-D2-002, IMPL-D2-003
- In Progress: IMPL-D2-004 (80% complete, caching partially implemented)
- Blocked: None
- Additional Tasks Discovered: IMPL-D2-001a, IMPL-D2-001b (both completed)

## Updated TASK_COMPLETION_CHECKLIST.md:
- [x] IMPL-D2-001: Download ONNX model files
- [x] IMPL-D2-002: Create ONNXEncoder class  
- [x] IMPL-D2-003: Implement encode_batch method
- [ ] IMPL-D2-004: Add caching with functools.lru_cache (80%)
```

### 3. Cross-Session Handoff
When switching agents or continuing later:

```python
# New agent starts by:
1. Read TASK_COMPLETION_CHECKLIST.md
2. Check for any partial completions noted
3. TodoWrite with remaining tasks
4. Continue implementation
```

## 🔄 Task Completion Workflow

### 1. Pre-Implementation
```python
# 1. Read the implementation guide
Read("docs/IMPLEMENTATION_GUIDE.md")

# 2. Check task checklist for current day
Read("TASK_COMPLETION_CHECKLIST.md")

# 3. Create session TODOs
TodoWrite([...])  # Map checklist items to TODOs

# 4. Review existing code if continuing
Read("src/embedding/onnx_encoder.py")  # If exists
```

### 2. During Implementation
```python
# 1. Mark task as in_progress
TodoWrite([
    {"id": "IMPL-D2-001", "status": "in_progress", ...}
])

# 2. Implement the feature
# ... actual coding ...

# 3. Run tests immediately
Bash("pytest tests/unit/test_onnx_encoder.py")

# 4. Mark as completed ONLY if tests pass
TodoWrite([
    {"id": "IMPL-D2-001", "status": "completed", ...}
])
```

### 3. Post-Implementation
```python
# 1. Update the checklist file
Edit("TASK_COMPLETION_CHECKLIST.md", 
     old="- [ ] IMPL-D2-001",
     new="- [x] IMPL-D2-001")

# 2. Document any issues or notes
Edit("TASK_COMPLETION_CHECKLIST.md",
     old="- [ ] IMPL-D2-004: Add caching",
     new="- [ ] IMPL-D2-004: Add caching (80% - LRU implemented, need TTL)")

# 3. Commit with clear message
Bash("git add -A && git commit -m 'feat: implement ONNX encoder (IMPL-D2-001 to IMPL-D2-003)'")
```

## 💡 Usage Patterns & Examples

### Pattern 1: Feature Implementation
```python
# User: "Implement the dimension extractors"

# Agent approach:
1. TodoWrite with all dimension tasks (IMPL-D3-001 to IMPL-D3-006)
2. Implement each extractor sequentially
3. Create shared base class if pattern emerges
4. Test each individually, then integration test
5. Update checklist with completion status
```

### Pattern 2: Bug Fix During Implementation
```python
# While implementing IMPL-D4-001, discover a bug in Memory model

TodoWrite([
    {"id": "BUG-001", "content": "Fix Memory.decay_rate validation", "status": "pending", "priority": "high"},
    # ... existing todos
])

# Fix bug first (high priority)
# Then continue with original task
# Document in session summary
```

### Pattern 3: Complex Multi-File Changes
```python
# Task requires changes across multiple files

TodoWrite([
    {"id": "IMPL-D5-001", "content": "Create IngestionPipeline class", "status": "pending", "priority": "high"},
    {"id": "IMPL-D5-001-models", "content": "Update Memory model for ingestion", "status": "pending", "priority": "high"},
    {"id": "IMPL-D5-001-api", "content": "Add ingestion endpoint", "status": "pending", "priority": "medium"},
    {"id": "IMPL-D5-001-tests", "content": "Write integration tests", "status": "pending", "priority": "medium"}
])

# Break complex tasks into logical subtasks
```

### Pattern 4: Research Before Implementation
```python
# Some tasks require investigation first

TodoWrite([
    {"id": "RESEARCH-001", "content": "Investigate DBSCAN parameters for consolidation", "status": "pending", "priority": "high"},
    {"id": "IMPL-D12-001", "content": "Implement ConsolidationEngine", "status": "pending", "priority": "high"}
])

# Complete research task first
# Document findings in code comments
# Then implement with confidence
```

## 🚨 Common Pitfalls & Solutions

### Pitfall 1: Marking Tasks Complete Prematurely
```python
# ❌ Wrong: Marking complete without testing
TodoWrite([{"id": "IMPL-D2-001", "status": "completed", ...}])

# ✅ Correct: Test first, then mark complete
Bash("pytest tests/unit/test_feature.py")
# Only if tests pass:
TodoWrite([{"id": "IMPL-D2-001", "status": "completed", ...}])
```

### Pitfall 2: Not Tracking Discovered Work
```python
# ❌ Wrong: Doing extra work without tracking
# Just implementing without updating TODOs

# ✅ Correct: Add discovered tasks
TodoWrite([
    {"id": "IMPL-D2-001a", "content": "Handle edge case for empty input", ...}
])
```

### Pitfall 3: Losing Context Between Sessions
```python
# ❌ Wrong: Starting fresh without checking previous work

# ✅ Correct: Always start with:
1. Read TASK_COMPLETION_CHECKLIST.md
2. Check git status
3. Read any partially completed code
4. TodoWrite with remaining work
```

## 📈 Progress Metrics

### Daily Velocity Tracking
Track completion rate to estimate timeline:
```
Day 1: 5/5 tasks (100%)
Day 2: 3/4 tasks (75%) - 1 carried forward
Day 3: 6/6 tasks + 1 carried (100%)
Average: 93% daily completion
```

### Complexity Indicators
Note which tasks take longer:
```
Quick (<30 min): Simple models, basic CRUD
Medium (30-90 min): Engines, algorithms
Long (>90 min): Integration, complex algorithms
```

### Blocker Tracking
Document blockers in checklist:
```markdown
- [ ] IMPL-D4-003: Vector composition ⚠️ BLOCKED: Waiting for dimension extractors
```

## 🎯 Best Practices Summary

1. **Always start with TodoWrite** mapping checklist tasks
2. **Use official task IDs** from TASK_COMPLETION_CHECKLIST.md
3. **Test before marking complete** - no exceptions
4. **Document discoveries** as new TODOs with suffixes
5. **Update both systems** - TODOs during session, checklist after
6. **Provide session summaries** for handoffs
7. **Track blockers and partial completions** in checklist notes
8. **Commit frequently** with task IDs in commit messages

## 🔗 Related Documentation

- [TASK_COMPLETION_CHECKLIST.md](../TASK_COMPLETION_CHECKLIST.md) - Master task list
- [IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md) - Daily implementation guide
- [AGENT_START_HERE.md](AGENT_START_HERE.md) - Quick start for agents

Remember: The dual-layer system ensures both immediate task focus (TODOs) and long-term project tracking (checklist). Use both for maximum effectiveness!