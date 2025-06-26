# 🚀 AGENT START HERE - Cognitive Meeting Intelligence

## Quick Orientation (2 minutes)

### What is this project?
A system that transforms meeting transcripts into queryable cognitive memories using 400D vectors, activation spreading, and bridge discovery algorithms.

### Where are you in the process?
1. Check `TASK_COMPLETION_CHECKLIST.md` - See completed vs pending tasks
2. Check `IMPLEMENTATION_GUIDE.md` - Find today's work (Day 1-35)
3. Read `current_project_state` memory - Latest status update

## 🎯 Your Workflow

### 1. Start of Session
```bash
# Always start with:
source venv/bin/activate  # or venv\Scripts\activate on Windows
python scripts/check_project_state.py
```

### 2. Use TodoWrite Immediately
```python
# Example for Day 1 implementation:
TodoWrite([
    {"id": "1", "content": "Read Day 1 specs in IMPLEMENTATION_GUIDE.md", "status": "pending", "priority": "high"},
    {"id": "2", "content": "Create src/models/entities.py with dataclasses", "status": "pending", "priority": "high"},
    {"id": "3", "content": "Create SQLite schema", "status": "pending", "priority": "high"},
    {"id": "4", "content": "Write unit tests", "status": "pending", "priority": "medium"},
    {"id": "5", "content": "Run quality checks", "status": "pending", "priority": "medium"}
])
```

### 3. Find Your Tasks (SIMPLIFIED!)

#### 📍 Single Source of Truth:
```
IMPLEMENTATION_GUIDE.md (Day 1-35 roadmap)
    ↓
Phase memory files (Detailed specs)
    ↓
TASK_COMPLETION_CHECKLIST.md (ALL 129 tasks + quality checks)
```

#### For Implementation Work:
1. **Always start**: `IMPLEMENTATION_GUIDE.md` - Find your day
2. **Get task IDs**: `TASK_COMPLETION_CHECKLIST.md` - All IMPL-D1-001 format tasks
3. **Get details**: Phase memory files
   - `phase1_implementation_tasks` (Week 1: Foundation)
   - `phase2_implementation_tasks_detailed` (Week 2: Cognitive)
   - `phase3_implementation_tasks_consolidated` (Week 3: Advanced)
   - `phase4_implementation_tasks` (Week 4: Consolidation)
   - `phase5_implementation_tasks` (Week 5: Production)

#### For Other Work:
- **Infrastructure**: `docs/TASK_TRACKING_SYSTEM.md` (PROJECT-XXX)
- **Progress**: `docs/progress/` directory

### 4. While Coding  
- Keep `TASK_COMPLETION_CHECKLIST.md` open (ALL tasks + quality checks)
- Update todos to "in_progress" before starting
- Mark "completed" immediately when done
- Check off implementation tasks as you complete them
- Run quality checks from checklist

### 5. Before Committing
```bash
# Always run:
black src/ tests/ --line-length 100
flake8 src/ tests/ --max-line-length 100  
mypy src/
pytest tests/unit/test_<your_module>.py
```

## 📁 Key Files Quick Reference

### Documentation Hierarchy
```
AGENT_START_HERE.md (You are here)
├── IMPLEMENTATION_GUIDE.md      (What to build when)
├── TASK_COMPLETION_CHECKLIST.md (ALL 129 tasks)
├── CLAUDE.md                    (Project context & patterns)
├── CLAUDE_NAVIGATION.md         (Where everything lives)
└── docs/
    ├── technical-implementation.md (Full spec)
    └── TASK_TRACKING_SYSTEM.md     (Infra tasks only)
```

### Code Structure
```
src/
├── models/entities.py     (Start here for Day 1)
├── embedding/             (Day 2-3)
├── extraction/            (Day 3,5)
├── storage/               (Day 4)
├── pipeline/              (Day 5)
└── api/                   (Day 6-7)
```

### Where to Find Answers
- **"What should I build?"** → IMPLEMENTATION_GUIDE.md
- **"What are all the tasks?"** → TASK_COMPLETION_CHECKLIST.md
- **"How should I build it?"** → docs/technical-implementation.md + CLAUDE.md
- **"What's the current status?"** → Read `current_project_state` memory
- **"Where is everything?"** → CLAUDE_NAVIGATION.md

## ⚡ Common Agent Pitfalls

1. **Not using TodoWrite** - Always create session todos!
2. **Skipping quality checks** - Run black/flake8/mypy/tests
3. **Not reading memories** - They have crucial details
4. **Creating new patterns** - Follow existing code style
5. **Missing project context** - Read CLAUDE.md first

## 🎪 Ready to Start?

1. ✅ Read this file completely
2. ✅ Check SETUP_STATUS_SUMMARY.md for blockers
3. ✅ Open IMPLEMENTATION_GUIDE.md for today's work
4. ✅ Create your TodoWrite list
5. ✅ Start coding!

Remember: You're building a cognitive intelligence system, not just storage. Every feature should enhance understanding of meetings.