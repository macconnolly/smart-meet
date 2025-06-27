# Git Workflow Enforcement Rules

## 🚨 MANDATORY: Git Workflow Requirements

### 1. **Branch Creation for Tasks**
Before starting ANY implementation task:
```bash
# Check current branch
git branch --show-current

# Create task-specific branch
git checkout -b impl/D{day}-{task_number}-{description}
# Example: git checkout -b impl/D2-001-onnx-encoder
```

### 2. **Commit Message Format**
EVERY commit MUST follow this format:
```
<type>: <description> [<task-id>]

<detailed body explaining changes>

Refs: #<task-id>
```

**Example**:
```
feat: Implement ONNX encoder with caching [IMPL-D2-001]

- Add ONNXEncoder class with warmup functionality
- Implement LRU caching for repeated encodings
- Add performance tracking and stats
- Achieve <100ms encoding target

Refs: #D2-001
```

### 3. **Pre-Commit Checklist**
ALWAYS run these before committing:
```bash
# 1. Check what you're committing
git status
git diff --staged

# 2. Verify branch name matches task
git branch --show-current

# 3. Run quality checks
black src/ tests/ --line-length 100
flake8 src/ tests/ --max-line-length 100
pytest tests/unit/test_<component>.py

# 4. Check recent commits for context
git log --oneline -5
```

### 4. **Task ID Mapping**
- Find task IDs in `TASK_COMPLETION_CHECKLIST.md`
- Day 1 tasks: IMPL-D1-001 through IMPL-D1-008
- Day 2 tasks: IMPL-D2-001 through IMPL-D2-004
- Format: IMPL-D{day}-{number:03d}

### 5. **Types for Commits**
- `feat`: New feature implementation
- `fix`: Bug fixes
- `refactor`: Code refactoring (no behavior change)
- `test`: Adding or updating tests
- `docs`: Documentation only
- `chore`: Maintenance tasks
- `perf`: Performance improvements

### 6. **Merge Strategy**
```bash
# After task completion
git checkout feature/day-{N}-{description}
git merge impl/D{N}-{task}-{description}
git branch -d impl/D{N}-{task}-{description}
```

### 7. **Common Mistakes to AVOID**
- ❌ Committing directly to feature branches without task branches
- ❌ Generic commit messages like "update files" or "fix issues"
- ❌ Missing task IDs in commits
- ❌ Not running tests before committing
- ❌ Large commits mixing multiple tasks

### 8. **Git Commit Examples by Task Type**

**Model Implementation**:
```
feat: Add Memory dataclass with all fields [IMPL-D1-001]

- Create Memory entity with 15+ fields
- Add validation in __post_init__
- Include to_dict/from_dict methods
- Support all memory types and content types

Refs: #D1-001
```

**Database Setup**:
```
feat: Initialize SQLite schema with 9 tables [IMPL-D1-002]

- Create all tables from schema.sql
- Add performance indexes
- Insert initial system metadata
- Enable foreign keys and WAL mode

Refs: #D1-002
```

**API Endpoint**:
```
feat: Add memory search endpoint [IMPL-D6-002]

- POST /api/v2/memories/search
- Vector similarity search across tiers
- Filter by project, content type
- Return SearchResult with scores

Refs: #D6-002
```

### 9. **When to Create Pull Requests**
- End of each day's work
- After completing a major feature
- When merging to develop branch
- Never commit directly to main

### 10. **Recovery from Mistakes**
```bash
# If you committed without task ID
git commit --amend -m "feat: Proper message [IMPL-D2-001]"

# If you're on wrong branch
git stash
git checkout -b impl/D2-001-proper-name
git stash pop

# If you need to split commits
git reset --soft HEAD~1
# Then recommit properly
```

## 🎯 Remember
Every commit tells a story. Make it clear:
- WHAT changed (title)
- WHY it changed (body)
- WHERE it fits (task ID)

This ensures perfect traceability from requirements → tasks → code → commits.