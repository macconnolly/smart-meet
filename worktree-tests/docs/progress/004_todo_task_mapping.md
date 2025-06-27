# 004 - TODO to Task ID Mapping

## Overview
This milestone focuses on mapping the 300+ TODO messages in the skeleton structure to their corresponding IMPL-D*-*** task IDs from TASK_COMPLETION_CHECKLIST.md, creating clear traceability between implementation guidance and tracked tasks.

## Status
- **Started**: 2025-01-27
- **Current Step**: Mapping In Progress
- **Completion**: 10% (Demonstrated pattern)
- **Expected Completion**: 2025-01-27

## Objectives
- [ ] Map all TODOs to specific IMPL-D*-*** task IDs
- [x] Create mapping strategy document
- [x] Demonstrate pattern with key files
- [ ] Update all 300+ TODOs across codebase
- [ ] Verify mapping completeness
- [ ] Create automation script for future updates

## Implementation Progress

### Step 1: Mapping Strategy Creation
**Status**: Completed
**Date Range**: 2025-01-27

#### Tasks Completed
- Created comprehensive `todo_to_task_mapping_plan` memory
- Mapped all 129 implementation tasks to their corresponding files
- Established TODO format update pattern
- Documented Day 1-35 file-to-task relationships

#### Current Work
- None - strategy complete

#### Next Tasks
- Apply mapping across all files

### Step 2: Pattern Demonstration
**Status**: In Progress
**Date Range**: 2025-01-27

#### Tasks Completed
- Updated `src/embedding/onnx_encoder.py` TODOs:
  - `TODO Day 2:` → `TODO: [IMPL-D2-002]`
  - Added specific task descriptions
  - Maintained implementation guidance
- Updated `src/cognitive/activation/engine.py` TODOs:
  - `@TODO:` → `TODO: [IMPL-D8-001]`
  - Linked configuration TODOs to IMPL-D8-002
  - Preserved agentic empowerment messages

#### Current Work
- Demonstrating pattern across different phases

#### Next Tasks
- Complete remaining files systematically
- Create bulk update script

### Step 3: Systematic Application
**Status**: Not Started
**Date Range**: 2025-01-27

#### Tasks Completed
- None yet

#### Current Work
- Planning systematic update approach

#### Next Tasks
- Update Day 1 files (models, database)
- Update Day 3-7 files (dimensions, storage, pipeline, API)
- Update Phase 2-5 files (cognitive features, production)

## Technical Notes

### TODO Format Patterns Found
1. **Day-based TODOs**: `TODO Day X:`
2. **Annotation TODOs**: `@TODO:`
3. **Simple TODOs**: `TODO:`
4. **Inline TODOs**: `# TODO:`

### Standardized Format
```python
# TODO: [IMPL-D{day}-{task}] {specific description}
```

Example transformations:
- `TODO Day 2: Load from transformers` → `TODO: [IMPL-D2-002] Load tokenizer from transformers`
- `@TODO: Implement BFS` → `TODO: [IMPL-D8-001] Implement two-phase BFS algorithm`

### Mapping Benefits
1. **Traceability**: Direct link from code to task checklist
2. **Progress Tracking**: Can grep for task IDs to check implementation
3. **Context**: Developers know exactly which task they're implementing
4. **Verification**: Easy to verify all tasks have corresponding TODOs

### Files Updated So Far
1. `src/embedding/onnx_encoder.py` - 5 TODOs mapped to IMPL-D2-002
2. `src/cognitive/activation/engine.py` - 4 TODOs mapped to IMPL-D8-001/002

### Remaining Work Estimate
- ~290 TODOs remaining across ~50 files
- Estimated 2-3 hours for manual updates
- Or 30 minutes with automation script

## Dependencies
- TASK_COMPLETION_CHECKLIST.md (source of task IDs)
- All skeleton files in src/ directory
- todo_to_task_mapping_plan memory

## Risks & Mitigation
- **Risk**: Missing TODOs during update
  - **Mitigation**: Use grep to find all TODO patterns
- **Risk**: Incorrect task ID assignment
  - **Mitigation**: Reference mapping plan for each file
- **Risk**: Breaking existing TODO tools/scripts
  - **Mitigation**: Maintain TODO: prefix for compatibility

## Resources
- Mapping plan: `todo_to_task_mapping_plan` memory
- Task list: TASK_COMPLETION_CHECKLIST.md
- Pattern examples: This document

## Change Log
- **2025-01-27 14:00**: Created mapping strategy
- **2025-01-27 14:15**: Updated first demonstration files
- **2025-01-27 14:30**: Created progress documentation

## Next Steps
1. Create grep command to find all TODOs: `grep -r "TODO" src/ | wc -l`
2. Develop sed/python script for bulk updates
3. Apply updates systematically by phase
4. Verify completeness with final grep
5. Update this progress document to 100%