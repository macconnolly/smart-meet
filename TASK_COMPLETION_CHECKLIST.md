# Task Completion Checklist

## Before Starting Any Task
- [ ] Read and understand the task requirements
- [ ] Check existing memories for relevant context
- [ ] Review project architecture and patterns
- [ ] Plan the implementation approach
- [ ] Create todo items for complex tasks

## Code Implementation Checklist

### 1. Pre-Implementation
- [ ] Identify affected files and modules
- [ ] Check for existing similar implementations
- [ ] Review relevant design patterns
- [ ] Consider performance implications
- [ ] Review the enhanced database schema (projects, deliverables, stakeholders)

### 2. During Implementation
- [ ] Follow code style conventions (PEP 8, Black formatting)
- [ ] Add comprehensive type hints
- [ ] Write descriptive docstrings
- [ ] Implement error handling
- [ ] Use async/await for I/O operations
- [ ] Follow repository/engine patterns where applicable
- [ ] Include project context in all operations
- [ ] Consider stakeholder and deliverable relationships

### 3. Code Quality Checks
- [ ] Run Black formatter: `black src/ tests/ --line-length 100`
- [ ] Run Flake8 linter: `flake8 src/ tests/ --max-line-length 100`
- [ ] Run MyPy type checker: `mypy src/`
- [ ] Ensure no import errors
- [ ] Check for SQL injection vulnerabilities
- [ ] Validate all user inputs

### 4. Testing Requirements
- [ ] Write unit tests for new functionality
- [ ] Update integration tests if needed
- [ ] Run test suite: `pytest`
- [ ] Verify test coverage: `pytest --cov=src`
- [ ] Test async functions properly
- [ ] Mock external dependencies
- [ ] Test with project-scoped data
- [ ] Test consulting-specific features (stakeholders, deliverables)

### 5. Documentation Updates
- [ ] Update docstrings for modified functions
- [ ] Add inline comments for complex logic
- [ ] Update relevant memories if project structure changes
- [ ] Document new API endpoints
- [ ] Add usage examples for new features
- [ ] Update API documentation with request/response examples
- [ ] Document any new environment variables

### 6. Performance Verification
- [ ] Test with realistic data volumes (10K+ memories)
- [ ] Verify query performance < 2s
- [ ] Check memory usage is reasonable
- [ ] Profile critical paths if needed
- [ ] Test with multiple projects (50+)
- [ ] Verify caching effectiveness

### 7. Integration Testing
- [ ] Test full pipeline end-to-end
- [ ] Verify all components work together
- [ ] Test error scenarios
- [ ] Check background tasks (consolidation, lifecycle)
- [ ] Test project isolation
- [ ] Verify stakeholder filtering works
- [ ] Test deliverable linking

### 8. Before Completing Task
- [ ] All tests pass
- [ ] Code is properly formatted
- [ ] No linting errors
- [ ] Type checking passes
- [ ] Documentation is complete
- [ ] Performance targets met
- [ ] No hardcoded values (use config)
- [ ] Logging added for debugging

## Phase-Specific Checklists

### Phase 1: Foundation (Week 1)
- [ ] SQLite database with enhanced schema (projects, meetings, deliverables)
- [ ] ONNX model downloaded and converted
- [ ] Qdrant 3-tier collections created
- [ ] Basic ingestion pipeline working
- [ ] 400D vector composition (384D + 16D)
- [ ] Simple retrieval API functional
- [ ] Temporal & Emotional extractors implemented
- [ ] Placeholder extractors for remaining dimensions

### Phase 2: Cognitive Intelligence (Week 2)
- [ ] Project-aware activation spreading
- [ ] Stakeholder influence calculation
- [ ] Deliverable network activation
- [ ] Two-phase BFS with path tracking
- [ ] Consulting-specific classification
- [ ] Meeting type prioritization
- [ ] Cross-project insights (when enabled)
- [ ] Performance < 500ms for project-scoped activation

### Phase 3: Advanced Features (Week 3)
- [ ] Bridge discovery with distance inversion
- [ ] Social & Causal dimension extractors
- [ ] Strategic dimension extractor (replacing evolutionary)
- [ ] Advanced stakeholder sentiment analysis
- [ ] Hypothesis-evidence linking
- [ ] Risk-mitigation pairing
- [ ] Bridge caching for performance

### Phase 4: Consolidation (Week 4)
- [ ] DBSCAN clustering implementation
- [ ] Semantic memory generation
- [ ] Individual memory promotion
- [ ] Memory lifecycle management
- [ ] Importance decay/reinforcement
- [ ] Scheduled background tasks
- [ ] Parent-child relationships
- [ ] Consolidation performance < 5s for 1000 memories

### Phase 5: Production (Week 5)
- [ ] Unified cognitive API
- [ ] Query caching with TTL
- [ ] Database connection pooling
- [ ] Input validation & sanitization
- [ ] Rate limiting implementation
- [ ] Structured logging
- [ ] Performance monitoring
- [ ] Docker optimization
- [ ] Load testing passed

## Deployment Checklist
- [ ] Environment variables configured
- [ ] Docker image builds successfully
- [ ] Health checks pass
- [ ] API endpoints respond correctly
- [ ] Background services start properly
- [ ] Logging configured appropriately
- [ ] Error handling works as expected
- [ ] Backup procedures documented
- [ ] Rollback plan in place

## Strategy Consulting Specific Checks
- [ ] Project isolation verified
- [ ] Stakeholder permissions working
- [ ] Deliverable tracking functional
- [ ] Meeting categorization correct
- [ ] Priority levels applied properly
- [ ] Client vs internal separation maintained
- [ ] Hypothesis tracking operational
- [ ] Risk identification working

## Common Gotchas to Check
- [ ] Async functions properly awaited
- [ ] Database connections properly closed
- [ ] Memory leaks in long-running processes
- [ ] Proper error propagation
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention
- [ ] Proper indices on database tables
- [ ] JSON fields properly escaped
- [ ] Timezone handling consistent
- [ ] Unicode handling in transcripts

## Security Checklist
- [ ] All inputs sanitized
- [ ] SQL injection prevented (parameterized queries)
- [ ] Rate limiting active
- [ ] CORS properly configured
- [ ] Sensitive data not logged
- [ ] API keys not hardcoded
- [ ] File paths validated
- [ ] Memory access scoped by project

## Final Verification
- [ ] Run full test suite one more time
- [ ] Perform manual testing of key features
- [ ] Review code changes for consistency
- [ ] Ensure all TODOs are addressed
- [ ] Verify no debug code remains
- [ ] Check that all files are saved
- [ ] Update version numbers if needed
- [ ] Create release notes if applicable

## Git Commit Checklist
- [ ] Changes are atomic and focused
- [ ] Commit message follows convention
- [ ] Tests pass before committing
- [ ] No merge conflicts
- [ ] Branch is up to date with main
- [ ] PR description is comprehensive
- [ ] Code review requested