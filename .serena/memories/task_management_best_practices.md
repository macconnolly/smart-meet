# Task Management Best Practices

## Dual-Track System

We use a **dual-track task management system** to maintain clarity:

### Track 1: Setup/Infrastructure Tasks
**Where**: `docs/TASK_TRACKING_SYSTEM.md`
**Prefix**: PROJECT-, INFRA-, MAINT-, DOC-
**Purpose**: Project setup, tooling, infrastructure, maintenance
**Examples**:
- PROJECT-001: Initialize git repository
- INFRA-001: Setup CI/CD pipeline
- DOC-001: Create API documentation

### Track 2: Implementation Tasks  
**Where**: Memories + Progress docs
**Prefix**: IMPL-DX-XXX (Day X)
**Purpose**: Actual feature implementation following IMPLEMENTATION_GUIDE.md
**Examples**:
- IMPL-D1-001: Create Memory dataclass
- IMPL-D2-001: Implement ONNX encoder
- IMPL-D3-001: Create vector composition

## Why This Separation?

1. **Clarity**: Infrastructure work doesn't clutter implementation tracking
2. **Different Lifecycles**: Setup tasks are one-time, implementation is iterative
3. **Different Audiences**: 
   - Setup tasks: DevOps, senior engineers
   - Implementation tasks: All developers
4. **Progress Tracking**: Implementation progress is measurable against roadmap

## Task Lifecycle

### Setup Tasks (PROJECT/INFRA)
```
TODO → IN PROGRESS → BLOCKED/COMPLETE → ARCHIVED
```

### Implementation Tasks (IMPL)
```
PLANNED → IN PROGRESS → CODE COMPLETE → TESTED → INTEGRATED → DOCUMENTED
```

## Where Things Live

### Setup/Infrastructure:
- **Definition**: docs/TASK_TRACKING_SYSTEM.md
- **Progress**: Update status in the file
- **Completion**: Mark complete and archive

### Implementation:
- **Definition**: Memories (phase1_implementation_tasks, etc.)
- **Specs**: docs/specs/dayX-feature.md
- **Progress**: docs/progress/XXX_feature.md
- **Status**: TASK_COMPLETION_CHECKLIST.md

## Quick Decision Tree

**Is this task about:**
- Setting up the project? → PROJECT-XXX
- Infrastructure/CI/CD? → INFRA-XXX  
- Maintenance/cleanup? → MAINT-XXX
- Documentation only? → DOC-XXX
- Building a feature from IMPLEMENTATION_GUIDE? → IMPL-DX-XXX

## Session Workflow

1. **Start of Session**:
   - Check TASK_TRACKING_SYSTEM.md for blockers
   - Pick implementation task from memories
   - Create feature branch

2. **During Session**:
   - Work on IMPL tasks
   - If you discover setup issues, add PROJECT task
   - Update progress hourly

3. **End of Session**:
   - Update task status in appropriate location
   - Create progress document
   - Commit with references to task numbers

## Task Number Usage

Always reference task numbers in:
- Commit messages: `feat(memory): implement to_dict method [IMPL-D1-004]`
- TODO comments: `# TODO: IMPL-D1-004 - Add validation`
- Progress docs: List all tasks worked on
- Pull requests: Reference all included tasks