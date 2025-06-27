# Git Workflow and Branch Management

## Branch Strategy

### Main Branches
- `main` - Production-ready code
- `develop` - Integration branch for features

### Feature Branches
- Pattern: `feature/day-{N}-{description}`
- Example: `feature/day-1-core-models`
- Created from: `develop`
- Merged to: `develop`

### Implementation Branches (AI-Assisted)
- Pattern: `impl/{task-id}-{description}`
- Example: `impl/D1-001-memory-model`
- For individual task implementation

### Fix Branches
- Pattern: `fix/{issue-description}`
- Example: `fix/vector-dimension-mismatch`

## Workflow Commands

### Starting a New Day's Work
```bash
# Update develop branch
git checkout develop
git pull origin develop

# Create day branch
git checkout -b feature/day-1-core-models

# Create task branch
git checkout -b impl/D1-001-memory-model
```

### Committing Work
```bash
# Stage changes
git add -p  # Interactive staging
git add .   # Stage all

# Commit with task reference
git commit -m "feat: Implement Memory model [IMPL-D1-001]

- Add Memory dataclass with all fields
- Include vector composition methods
- Add validation logic

Refs: #D1-001"
```

### Merging Task to Day Branch
```bash
# Complete task branch
git checkout feature/day-1-core-models
git merge impl/D1-001-memory-model
git branch -d impl/D1-001-memory-model
```

### End of Day Integration
```bash
# Merge day's work to develop
git checkout develop
git merge feature/day-1-core-models

# Tag milestone
git tag -a "phase-1-day-1-complete" -m "Complete Day 1: Core Models"
```

## Commit Message Convention

### Format
```
<type>: <description> [<task-id>]

<body>

Refs: #<task-id>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `test`: Adding tests
- `docs`: Documentation
- `chore`: Maintenance
- `perf`: Performance improvement

### Examples
```
feat: Add activation spreading algorithm [IMPL-D8-001]

- Implement two-phase BFS activation
- Add decay calculations
- Include relevance scoring

Refs: #D8-001
```

## AI Agent Git Commands

### Check Status
```bash
git status
git diff --staged
git log --oneline -10
```

### Create Branch
```bash
git checkout -b impl/D1-001-memory-model
```

### Stage and Commit
```bash
git add src/models/memory.py
git commit -m "feat: Implement Memory model [IMPL-D1-001]"
```

### Push Branch
```bash
git push -u origin impl/D1-001-memory-model
```

## Task Tracking Integration

Each commit should reference the task ID from TASK_COMPLETION_CHECKLIST.md:
- Use `[IMPL-D1-001]` in commit message
- Reference in body with `Refs: #D1-001`
- Update task status after merge

## Protected Branches

- `main` - Requires PR and review
- `develop` - Requires passing tests

## Git Hooks (Pre-commit)

Automatically runs:
1. Black formatting
2. Flake8 linting
3. MyPy type checking
4. Pytest unit tests

## Conflict Resolution

1. Always pull latest `develop` before starting work
2. Resolve conflicts in feature branch
3. Test thoroughly after resolution
4. Never force push to shared branches

## Release Process

1. Merge `develop` to `main`
2. Tag release: `git tag -a v0.1.0 -m "Phase 1 Release"`
3. Push tags: `git push --tags`
4. GitHub Actions handles deployment

## Best Practices

1. Commit early and often
2. One task = one commit (when possible)
3. Always reference task IDs
4. Keep commits atomic
5. Write clear commit messages
6. Review diff before committing
7. Run tests before pushing