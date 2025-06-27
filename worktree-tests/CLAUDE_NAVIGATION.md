# 🧭 CLAUDE Navigation Guide - Where Everything Lives

## 🎯 Quick Decision Tree

### "I need to..."

**Start ANY work** → `docs/AGENT_START_HERE.md` → Navigation guide for agents
**Find today's tasks** → `IMPLEMENTATION_GUIDE.md` → Find your Day → `TASK_COMPLETION_CHECKLIST.md` for task IDs
**Track session work** → Use TodoWrite/TodoRead tools for live task management
**Fix project setup** → `docs/TASK_TRACKING_SYSTEM.md` → Find PROJECT-XXX task
**Understand the system** → `docs/technical-implementation.md` → Full technical spec
**Follow coding standards** → `CLAUDE.md` → Has patterns & style guide
**Submit work** → `structured_development_workflow` memory → Create PR

## 📍 Document & Memory Map

### 🏗️ Core Implementation Documents (Start Here!)

1. **docs/AGENT_START_HERE.md** 🚀
   - **Purpose**: Single entry point for all agents
   - **Use when**: Starting ANY session
   - **Contains**: Workflow, navigation, common pitfalls
   - **First stop**: ALWAYS start here

2. **IMPLEMENTATION_GUIDE.md** 📘
   - **Purpose**: Day-by-day MVP roadmap (Days 1-35)
   - **Use when**: Finding which day/phase to work on
   - **Links to**: TASK_COMPLETION_CHECKLIST.md for task details
   - **Phases**: 5 phases, 35 days total

3. **TASK_COMPLETION_CHECKLIST.md** ✅
   - **Purpose**: Master list of ALL 129 implementation tasks
   - **Use when**: Finding specific tasks, checking progress
   - **Format**: IMPL-D1-001 through IMPL-D35-005
   - **Includes**: Quality checks, deployment tasks

4. **docs/TASK_TRACKING_SYSTEM.md** 🔧
   - **Purpose**: Infrastructure/setup tasks only
   - **Use when**: NOT for implementation tasks
   - **Task format**: PROJECT-XXX, INFRA-XXX

### 📚 Technical References

5. **docs/technical-implementation.md** 🏛️
   - **Purpose**: Complete technical specification
   - **Use when**: Need implementation details
   - **Contains**: Data models, algorithms, API specs
   - **944 lines**: Comprehensive reference

6. **CLAUDE.md** 📖
   - **Purpose**: AI assistant context & coding patterns
   - **Use when**: Understanding project, patterns, workflows
   - **Includes**: TodoWrite instructions, quick references

7. **README.md** 📖
   - **Purpose**: Project introduction
   - **Use when**: First time setup
   - **Links to**: AGENT_START_HERE.md prominently

### 🧠 Key Memories (Use via `mcp__serena__read_memory`)

8. **current_project_state** 🎯
   - **Purpose**: Latest project status and next steps
   - **Use when**: Starting work, checking status
   - **Updated**: After major milestones

9. **task_organization_single_source_truth** 📋
   - **Purpose**: Explains consolidated task system
   - **Use when**: Understanding task numbering/tracking
   - **Key insight**: TASK_COMPLETION_CHECKLIST.md has everything

10. **documentation_reorganization_plan** 🔄
    - **Purpose**: How docs are organized
    - **Use when**: Can't find something
    - **Shows**: Single source of truth approach

### 📝 Progress & Documentation

10. **docs/progress/XXX_name.md** 📊
    - **Purpose**: Session-by-session progress logs
    - **Use when**: End of each work session
    - **Template**: `docs/templates/progress_documentation_template.md`

11. **CHANGELOG.md** 📜
    - **Purpose**: User-facing version history
    - **Use when**: Preparing releases

### ⚠️ Current Memory Structure

**Phase Implementation Memories** (Still active - contain detailed specs):
- `phase1_implementation_tasks` → Week 1 detailed specs
- `phase2_implementation_tasks_detailed` → Week 2 detailed specs
- `phase3_implementation_tasks_consolidated` → Week 3 (merged from 2 files)
- `phase4_implementation_tasks` → Week 4 detailed specs
- `phase5_implementation_tasks` → Week 5 detailed specs

**Navigation Flow**:
```
AGENT_START_HERE.md
    ↓
IMPLEMENTATION_GUIDE.md (find your day)
    ↓
Phase memory files (get detailed specs)
    ↓
TASK_COMPLETION_CHECKLIST.md (track progress)
```

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

**Implementation Tasks (IMPL-D1-001 format):**
- Definition: `TASK_COMPLETION_CHECKLIST.md` (all 129 tasks)
- Details: Phase memory files
- Progress: Check off in `TASK_COMPLETION_CHECKLIST.md`
- Reference: `IMPLEMENTATION_GUIDE.md` for day/phase

**Session Tasks (TodoWrite/TodoRead):**
- Definition: Created at start of each session
- Progress: Update status as you work
- Completion: Mark done immediately

**Setup/Infra Tasks (PROJECT-XXX):**
- Definition: `docs/TASK_TRACKING_SYSTEM.md`
- Progress: Update status in same file
- Not for: Implementation work

## 🚀 Typical Workflow

```bash
# 1. Start EVERY session
Read docs/AGENT_START_HERE.md
Use TodoWrite to plan session tasks

# 2. Find your work
Read IMPLEMENTATION_GUIDE.md → Find your day
Read TASK_COMPLETION_CHECKLIST.md → Get task IDs
Read phase memory → Get implementation details

# 3. Track as you work
Update TodoWrite items to "in_progress"
Implement features per CLAUDE.md patterns
Mark todos "completed" immediately when done

# 4. Quality checks (from TASK_COMPLETION_CHECKLIST.md)
black src/ tests/ --line-length 100
flake8 src/ tests/ --max-line-length 100
mypy src/
pytest tests/unit/test_<module>.py

# 5. Update progress
Check off tasks in TASK_COMPLETION_CHECKLIST.md
Create docs/progress/XXX_description.md if major milestone
```

## 📂 Current Documentation Structure

### Single Source of Truth Achieved ✅
```
Navigation Flow:
├── docs/AGENT_START_HERE.md (Entry point)
├── IMPLEMENTATION_GUIDE.md (Day roadmap)
├── TASK_COMPLETION_CHECKLIST.md (All 129 tasks)
├── docs/technical-implementation.md (Full spec)
└── CLAUDE.md (Patterns & context)

Memory Organization:
├── Phase memories (implementation details)
├── current_project_state (status)
├── task_organization_single_source_truth (how it works)
└── documentation_reorganization_plan (structure)
```

### No More Conflicts!
- One place for tasks: TASK_COMPLETION_CHECKLIST.md
- One place for roadmap: IMPLEMENTATION_GUIDE.md
- One place to start: AGENT_START_HERE.md
- Clear navigation: No duplicate information

## ❓ Quick Reference

**Q: Where do I start for ANY work?**
A: `docs/AGENT_START_HERE.md` - ALWAYS start here!

**Q: Where are all the implementation tasks?**
A: `TASK_COMPLETION_CHECKLIST.md` - All 129 tasks with IDs

**Q: How do I find what to work on today?**
A: `IMPLEMENTATION_GUIDE.md` → Find your day (1-35)

**Q: Where are the technical details?**
A: `docs/technical-implementation.md` - Complete spec

**Q: How do I track my session work?**
A: Use TodoWrite at start, update as you work

**Q: Where are coding patterns?**
A: `CLAUDE.md` - Has all patterns and examples

**Q: What commands to run before committing?**
A: See quality checks in `TASK_COMPLETION_CHECKLIST.md`

---

**Remember**: ALWAYS start with `docs/AGENT_START_HERE.md` - it's your navigation guide!