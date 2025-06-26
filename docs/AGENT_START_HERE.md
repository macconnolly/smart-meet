# 🚀 AGENT START HERE - Cognitive Meeting Intelligence

## Quick Orientation (2 minutes)

### What is this project?
A system that transforms meeting transcripts into queryable cognitive memories using 400D vectors, activation spreading, and bridge discovery algorithms.

### Where are you in the process?
1. Check `SETUP_STATUS_SUMMARY.md` - Current state & blockers
2. Check `IMPLEMENTATION_GUIDE.md` - Find today's work (Day 1-35)
3. Check git log - See recent changes

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

### 3. Find Your Tasks

#### For Implementation Work:
1. **IMPLEMENTATION_GUIDE.md** - Day-by-day roadmap
2. **Memory files** - Detailed specs:
   - `phase1_implementation_tasks` (Week 1)
   - `phase2_implementation_tasks_detailed` (Week 2)
   - etc.

#### For Setup/Infrastructure:
- **docs/TASK_TRACKING_SYSTEM.md** - PROJECT-XXX tasks

### 4. While Coding
- Keep `TASK_COMPLETION_CHECKLIST.md` open
- Update todos to "in_progress" before starting
- Mark "completed" immediately when done
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
START_HERE.md (You are here)
├── SETUP_STATUS_SUMMARY.md      (Current state)
├── IMPLEMENTATION_GUIDE.md      (What to build)
├── TASK_COMPLETION_CHECKLIST.md (Quality checks)
├── CLAUDE.md                    (Project context)
└── docs/
    └── TASK_TRACKING_SYSTEM.md  (Task numbering)
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
- **"How should I build it?"** → Memory files + existing code patterns
- **"What to check?"** → TASK_COMPLETION_CHECKLIST.md
- **"Is something broken?"** → SETUP_STATUS_SUMMARY.md
- **"How are tasks numbered?"** → docs/TASK_TRACKING_SYSTEM.md

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