# 🧭 CLAUDE Navigation Guide - Where Everything Lives

## 🎯 Quick Decision Tree

### "I need to..."

**Start coding a feature** → `IMPLEMENTATION_GUIDE.md` → Find your Day → Check memory for that phase
**Track my work** → Create branch → Work → `docs/progress/XXX_feature.md` → Update `TASK_COMPLETION_CHECKLIST.md`
**Fix project setup** → `docs/TASK_TRACKING_SYSTEM.md` → Find PROJECT-XXX task
**Understand the system** → `docs/architecture/system-overview.md` → `project_overview` memory
**Follow coding standards** → `code_style_conventions` memory → Run `make format`
**Submit work** → `structured_development_workflow` memory → Create PR using `.github/pull_request_template.md`

## 📍 Document & Memory Map

### 🏗️ Core Implementation Documents (Start Here!)

1. **IMPLEMENTATION_GUIDE.md** 📘
   - **Purpose**: Day-by-day MVP roadmap (Days 1-35)
   - **Use when**: Starting ANY implementation work
   - **Links to**: Phase memories for detailed tasks
   - **Status tracking**: `TASK_COMPLETION_CHECKLIST.md`

2. **docs/TASK_TRACKING_SYSTEM.md** 🔧
   - **Purpose**: Track PROJECT/INFRA/DOC tasks (not implementation)
   - **Use when**: Fixing setup issues, adding tools, improving docs
   - **Task format**: PROJECT-001, INFRA-001, etc.

3. **TASK_COMPLETION_CHECKLIST.md** ✅
   - **Purpose**: Track implementation progress
   - **Use when**: Checking what's done, updating status
   - **Updated**: After each work session

### 📚 Technical References

4. **docs/architecture/system-overview.md** 🏛️
   - **Purpose**: High-level architecture and design
   - **Use when**: Understanding system components
   - **Companion**: `project_overview` memory

5. **README.md** 📖
   - **Purpose**: Project introduction and quick start
   - **Use when**: First time setup, sharing project

### 🧠 Key Memories (Use via `mcp__serena__read_memory`)

6. **project_overview** 🎯
   - **Purpose**: Core features and performance targets
   - **Use when**: Need quick reminder of what we're building

7. **structured_development_workflow** 🔄
   - **Purpose**: Git workflow, commits, PR process
   - **Use when**: Starting/ending work session
   - **Critical**: Has git init steps (PROJECT-001)

8. **code_style_conventions** 🎨
   - **Purpose**: Coding standards and patterns
   - **Use when**: Writing any code

9. **task_management_best_practices** 📋
   - **Purpose**: Explains dual-track task system
   - **Use when**: Confused about where to track tasks

### 📝 Progress & Documentation

10. **docs/progress/XXX_name.md** 📊
    - **Purpose**: Session-by-session progress logs
    - **Use when**: End of each work session
    - **Template**: `docs/templates/progress_documentation_template.md`

11. **CHANGELOG.md** 📜
    - **Purpose**: User-facing version history
    - **Use when**: Preparing releases

### ⚠️ Deprecated/To Be Consolidated

**Phase Implementation Memories** (Being consolidated):
- `phase1_implementation_tasks` → Use `IMPLEMENTATION_GUIDE.md` Day 1-7
- `phase2_implementation_tasks_detailed` → Use `IMPLEMENTATION_GUIDE.md` Day 8-14
- `phase3_implementation_tasks` + `phase3_implementation_tasks_detailed` → Merge needed
- `consolidated_implementation_strategy` → Redundant with `IMPLEMENTATION_GUIDE.md`
- `minimal_viable_features` → Now in `IMPLEMENTATION_GUIDE.md`

## 🗺️ Navigation Rules

### When to Use Files vs Memories

**Use FILES for:**
- ✅ Authoritative roadmaps (IMPLEMENTATION_GUIDE.md)
- ✅ Live task tracking (TASK_TRACKING_SYSTEM.md)
- ✅ Progress documentation (docs/progress/*.md)
- ✅ Code and schemas (src/*, *.sql)
- ✅ Public documentation (README.md, CHANGELOG.md)

**Use MEMORIES for:**
- ✅ Detailed task breakdowns by phase
- ✅ Conventions and standards
- ✅ Technical strategies and patterns
- ✅ Workflow guides
- ✅ Historical context

### Task Tracking Rules

**Implementation Tasks (IMPL-XXX):**
- Definition: Phase memories (until consolidated)
- Progress: `docs/progress/XXX_name.md`
- Status: `TASK_COMPLETION_CHECKLIST.md`
- Reference: `IMPLEMENTATION_GUIDE.md`

**Setup/Infra Tasks (PROJECT/INFRA-XXX):**
- Definition: `docs/TASK_TRACKING_SYSTEM.md`
- Progress: Update status in same file
- Completion: Mark complete and move on

## 🚀 Typical Workflow

```bash
# 1. Start of day
git pull
python scripts/check_project_state.py
Read IMPLEMENTATION_GUIDE.md for your day

# 2. Check tasks
- Implementation? → Check phase memory for your day
- Setup issue? → Check TASK_TRACKING_SYSTEM.md

# 3. Start work
git checkout -b feature/impl-d1-models
# Work on IMPL-D1-001, IMPL-D1-002, etc.

# 4. During work
- Follow code_style_conventions memory
- Run tests frequently
- Commit with task references

# 5. End of session
- Create: docs/progress/001_models_implementation.md
- Update: TASK_COMPLETION_CHECKLIST.md
- Commit and push
```

## 🧹 Memory Consolidation Plan

### Phase 1: Immediate Actions
1. **Merge duplicate phase3 memories** (PROJECT-003)
2. **Archive strategy memories** that are now in IMPLEMENTATION_GUIDE.md
3. **Create single source of truth** per phase

### Phase 2: New Structure
```
memories/
├── workflows/
│   ├── structured_development_workflow
│   ├── git_workflow
│   └── testing_workflow
├── standards/
│   ├── code_style_conventions
│   ├── api_design_standards
│   └── documentation_standards
├── implementation/
│   ├── phase1_detailed_tasks
│   ├── phase2_detailed_tasks
│   └── (one per phase, consolidated)
└── architecture/
    ├── system_design_decisions
    └── technical_patterns
```

## ❓ Quick Reference

**Q: Where do I start for Day 1 implementation?**
A: `IMPLEMENTATION_GUIDE.md` → Day 1 → `phase1_implementation_tasks` memory

**Q: How do I track a bug fix?**
A: Create INFRA-XXX task in `TASK_TRACKING_SYSTEM.md`

**Q: Where are coding standards?**
A: `code_style_conventions` memory

**Q: How do I know what's been completed?**
A: Check `TASK_COMPLETION_CHECKLIST.md`

**Q: Where do I document my session?**
A: Create `docs/progress/XXX_description.md`

**Q: What's our git workflow?**
A: Read `structured_development_workflow` memory

---

**Remember**: When in doubt, start with `IMPLEMENTATION_GUIDE.md` - it's the authoritative roadmap!