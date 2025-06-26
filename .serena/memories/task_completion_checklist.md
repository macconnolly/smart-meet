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

### 2. During Implementation
- [ ] Follow code style conventions (PEP 8, Black formatting)
- [ ] Add comprehensive type hints
- [ ] Write descriptive docstrings
- [ ] Implement error handling
- [ ] Use async/await for I/O operations
- [ ] Follow repository/engine patterns where applicable

### 3. Code Quality Checks
- [ ] Run Black formatter: `black src/ tests/ --line-length 100`
- [ ] Run Flake8 linter: `flake8 src/ tests/ --max-line-length 100`
- [ ] Run MyPy type checker: `mypy src/`
- [ ] Ensure no import errors

### 4. Testing Requirements
- [ ] Write unit tests for new functionality
- [ ] Update integration tests if needed
- [ ] Run test suite: `pytest`
- [ ] Verify test coverage: `pytest --cov=src`
- [ ] Test async functions properly
- [ ] Mock external dependencies

### 5. Documentation Updates
- [ ] Update docstrings for modified functions
- [ ] Add inline comments for complex logic
- [ ] Update relevant memories if project structure changes
- [ ] Document new API endpoints
- [ ] Add usage examples for new features

### 6. Performance Verification
- [ ] Test with realistic data volumes
- [ ] Verify query performance < 2s
- [ ] Check memory usage is reasonable
- [ ] Profile critical paths if needed

### 7. Integration Testing
- [ ] Test full pipeline end-to-end
- [ ] Verify all components work together
- [ ] Test error scenarios
- [ ] Check background tasks (consolidation, lifecycle)

### 8. Before Completing Task
- [ ] All tests pass
- [ ] Code is properly formatted
- [ ] No linting errors
- [ ] Type checking passes
- [ ] Documentation is complete
- [ ] Performance targets met

## Phase-Specific Checklists

### Phase 1 Implementation
- [ ] SQLite database initialized
- [ ] ONNX model downloaded and converted
- [ ] Qdrant collections created
- [ ] Basic ingestion pipeline working
- [ ] Simple retrieval API functional
- [ ] Dimension extractors implemented
- [ ] Vector composition (400D) working

### Phase 2 Implementation
- [ ] Activation spreading engine complete
- [ ] Two-phase BFS with path tracking
- [ ] Bridge discovery with distance inversion
- [ ] Memory consolidation pipeline
- [ ] Scheduled background tasks
- [ ] Unified cognitive API
- [ ] Performance caching implemented

## Deployment Checklist
- [ ] Environment variables configured
- [ ] Docker image builds successfully
- [ ] Health checks pass
- [ ] API endpoints respond correctly
- [ ] Background services start properly
- [ ] Logging configured appropriately
- [ ] Error handling works as expected

## Common Gotchas to Check
- [ ] Async functions properly awaited
- [ ] Database connections properly closed
- [ ] Memory leaks in long-running processes
- [ ] Proper error propagation
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention
- [ ] Proper indices on database tables

## Final Verification
- [ ] Run full test suite one more time
- [ ] Perform manual testing of key features
- [ ] Review code changes for consistency
- [ ] Ensure all TODOs are addressed
- [ ] Verify no debug code remains
- [ ] Check that all files are saved