# Structured Development Workflow

## 🚨 CRITICAL: Initial Git Setup (MUST DO FIRST!)

Since no git commits have been made yet:

```bash
# 1. Initialize git repository
git init

# 2. Configure git user (use your GitHub credentials)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 3. Create initial commit with existing code
git add .
git commit -m "chore: initial commit - project structure and foundation

- Project skeleton with all directories
- Database schema (enhanced with consulting features)
- Models and entities
- Configuration files
- Documentation structure
- No implementation code yet"

# 4. Create and connect to GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/cognitive-meeting-intelligence.git
git branch -M main
git push -u origin main
```

## 📋 Development Session Workflow

### 1. Pre-Session Setup (5-10 minutes)

```bash
# a. Pull latest changes
git checkout main
git pull origin main

# b. Create feature branch
git checkout -b feature/[task-name]  # e.g., feature/implement-onnx-encoder

# c. Review current state
make test  # Ensure tests pass
python scripts/check_project_state.py  # Verify project health

# d. Read relevant memories
# Check task-specific memories and documentation
```

### 2. During Development Session

#### Coding Standards
- Write tests FIRST (TDD approach)
- Follow code style conventions (run `make format` frequently)
- Update docstrings as you code
- Add type hints to all functions
- Use meaningful variable names

#### Progress Tracking
Every 1-2 hours or at logical breakpoints:
1. Run tests: `make test`
2. Commit work: Use conventional commits (see below)
3. Update progress notes in your local tracking

### 3. End of Session Documentation (REQUIRED - 15-20 minutes)

#### A. Create Progress Document
```bash
# 1. Check last progress document number
ls docs/progress/

# 2. Create new progress document
# If last was 002_testing_framework.md, create:
cp docs/templates/progress_documentation_template.md docs/progress/003_[your_task_name].md

# 3. Fill out ALL sections:
# - Overview of what you worked on
# - Status with accurate completion %
# - Tasks completed with timestamps
# - Current blockers
# - Next tasks
# - Technical notes on decisions made
```

#### B. Update Task Completion Checklist
```bash
# Edit TASK_COMPLETION_CHECKLIST.md
# Mark completed items with [x]
# Add any new discovered tasks
```

#### C. Final Commit and Push
```bash
# 1. Stage all changes
git add .

# 2. Create session summary commit
git commit -m "feat: [component] implement [what you did]

Session Summary:
- Completed: [list main accomplishments]
- In Progress: [what's partially done]
- Next: [what needs to be done next]

Updates:
- Progress doc: docs/progress/003_[name].md
- Checklist: Updated completion status"

# 3. Push to remote
git push origin feature/[task-name]
```

## 🔀 Git Workflow Standards

### Branch Strategy
```
main           → Production-ready code
├── develop    → Integration branch
│   ├── feature/[name]    → New features
│   ├── fix/[name]        → Bug fixes
│   ├── refactor/[name]   → Code improvements
│   └── docs/[name]       → Documentation updates
```

### Commit Convention (Conventional Commits)
```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, semicolons, etc)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

#### Examples:
```bash
# Feature
git commit -m "feat(encoder): implement ONNX model loading and encoding"

# Fix
git commit -m "fix(memory-repo): correct timestamp handling in queries"

# Docs
git commit -m "docs(api): add endpoint documentation for cognitive query"

# Test
git commit -m "test(activation): add unit tests for BFS spreading"
```

### Commit Frequency
- Commit logical units of work (not too big, not too small)
- At least one commit per hour during active development
- Always commit before switching tasks
- Never commit broken code to main/develop

## 🔍 Code Review Process

### Self-Review Checklist (Before PR)
- [ ] All tests pass locally
- [ ] Code is formatted (`make format`)
- [ ] No linting errors (`make lint`)
- [ ] Type checking passes (`make typecheck`)
- [ ] Documentation is updated
- [ ] Progress document is created
- [ ] Task checklist is updated

### Pull Request Template
```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Refactoring
- [ ] Documentation

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Documentation
- [ ] Progress doc: `docs/progress/XXX_name.md`
- [ ] API docs updated (if applicable)
- [ ] README updated (if applicable)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] No console.log/print statements
- [ ] No hardcoded values
```

## 🧪 Testing Requirements

### Before EVERY Commit:
```bash
# 1. Run unit tests
make test-unit

# 2. Run affected integration tests
pytest tests/integration/test_[affected_component].py

# 3. Check code quality
make format && make lint && make typecheck
```

### Before PR:
```bash
# Full test suite
make test

# Coverage check
pytest --cov=src --cov-report=html
# Ensure coverage doesn't decrease
```

## 🧹 Memory and Documentation Cleanup

### Weekly Cleanup Tasks
1. **Consolidate duplicate memories**:
   - `phase3_implementation_tasks` and `phase3_implementation_tasks_detailed` → merge into one
   - Review all phase documents for overlap

2. **Archive completed phase docs**:
   ```bash
   mkdir -p docs/archive/phases
   mv completed_phase*.md docs/archive/phases/
   ```

3. **Update master documents**:
   - IMPLEMENTATION_GUIDE.md - ensure it reflects current state
   - README.md - keep current with actual implementation

4. **Prune outdated memories**:
   - Remove superseded technical decisions
   - Archive old conversation states

## 📦 Release Process

### Version Numbering
Follow Semantic Versioning: `MAJOR.MINOR.PATCH`
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist
1. [ ] All tests pass
2. [ ] Documentation is complete
3. [ ] CHANGELOG.md updated
4. [ ] Version bumped in setup.py/pyproject.toml
5. [ ] Tag created: `git tag -a v1.0.0 -m "Release version 1.0.0"`
6. [ ] Progress documents archived

## 🚀 Quick Reference Commands

```bash
# Start new session
git checkout -b feature/my-task

# During development
make format           # Format code
make test-unit       # Run unit tests
git add -p           # Stage changes interactively
git commit           # Commit with conventional format

# End of session
# 1. Create progress doc: docs/progress/XXX_task.md
# 2. Update TASK_COMPLETION_CHECKLIST.md
# 3. Final commit and push
git push origin feature/my-task

# Create PR
# Use GitHub/GitLab UI with PR template
```

## ⚠️ Important Reminders

1. **NO COMMITS WITHOUT TESTS** - Write tests first!
2. **NO PUSH WITHOUT DOCUMENTATION** - Progress doc required!
3. **NO MERGE WITHOUT REVIEW** - Even for small changes!
4. **COMMIT EARLY AND OFTEN** - Don't lose work!
5. **UPDATE CHECKLIST** - Keep project status current!

## 🔧 Troubleshooting

### Common Issues:

1. **Tests failing locally but not in CI**:
   - Check Python version
   - Ensure clean environment
   - Run `make clean && make setup`

2. **Merge conflicts**:
   - Always pull main before starting work
   - Resolve conflicts locally
   - Test after resolving

3. **Large commits**:
   - Use `git add -p` for interactive staging
   - Split into logical commits
   - Each commit should have single purpose