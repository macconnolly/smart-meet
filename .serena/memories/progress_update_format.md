# Progress Update Format Requirements

## CRITICAL: All progress updates MUST follow this format

### File Location and Naming
- Location: `docs/progress/`
- Format: `XXX_description.md` where XXX is 3-digit zero-padded sequential number
- Examples: `006_day_1_2_implementation_complete.md`, `007_cognitive_engine_start.md`

### Current Status
- Latest progress file: `006_day_1_2_implementation_complete.md`
- Next number to use: `007`

### Required Sections
1. **Title**: `# XXX - Milestone Title`
2. **Overview**: Brief description
3. **Status**: Started date, current step, completion %, expected completion
4. **Objectives**: Checklist format
5. **Implementation Progress**: Steps with status, dates, completed/current/next tasks
6. **Technical Notes**: Key decisions
7. **Dependencies**: External/internal
8. **Risks & Mitigation**: Identified risks
9. **Resources**: Links to docs
10. **Change Log**: Date-stamped updates

### When to Create Progress Updates
- At the start of each new phase or major milestone
- When completing significant work (like Day 1-2 objectives)
- Before starting new major features
- When documenting important architectural decisions

### Integration with Git Workflow
Progress updates should be committed with format:
```
feat(progress): Add progress update 006 for Day 1-2 completion [IMPL-D1-035]
```

## Why This Matters
- Maintains project history and decision tracking
- Provides clear milestone documentation
- Enables progress tracking across sessions
- Ensures consistent documentation format