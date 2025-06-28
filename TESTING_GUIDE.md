# Testing Overview - Cognitive Meeting Intelligence

## Current Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── cognitive/          # Cognitive algorithm tests
│   ├── embedding/          # Embedding engine tests
│   ├── extraction/         # Memory extraction tests
│   └── models/            # Data model tests
├── integration/           # Integration tests
├── performance/          # Performance benchmarks
└── fixtures/            # Test data and fixtures
```

## Available Tests

### Unit Tests (Quick to run)
- `test_dimensions.py` - Dimensional analysis tests
- `test_embedding_engine.py` - Embedding generation tests
- `test_memory_extractor.py` - Memory extraction logic
- `test_repositories.py` - Database repository tests
- `test_models.py` - Pydantic model tests
- `test_onnx_encoder.py` - ONNX model tests

### Integration Tests
- API endpoint tests
- Database integration tests
- Vector search integration
- End-to-end pipeline tests

## Quick Test Commands

```bash
# Run all tests
pytest

# Run only unit tests (faster)
pytest tests/unit -v

# Run specific test file
pytest tests/unit/test_memory_extractor.py -v

# Run with coverage
pytest --cov=src --cov-report=html

# Run in parallel (faster)
pytest -n auto

# Run with detailed output
pytest -vv -s

# Run tests matching pattern
pytest -k "memory" -v
```

## WSL Testing Workflow

### Option 1: Docker (Easiest)
```bash
# No Python setup needed!
docker-compose up -d
docker-compose exec api pytest
```

### Option 2: Make Commands
```bash
make setup        # One-time setup
make test         # Run all tests
make test-unit    # Run unit tests only
make test-cov     # Run with coverage
```

### Option 3: Manual Setup
```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install test dependencies
pip install -r requirements.txt

# Run tests
pytest
```

## Common Test Issues & Solutions

### 1. Import Errors
```bash
# Ensure you're in activated venv
which python  # Should show venv/bin/python

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### 2. Database/Qdrant Errors
```bash
# Start services
docker-compose up -d

# Or mock them in tests
pytest tests/unit  # Unit tests don't need services
```

### 3. Slow Tests
```bash
# Run only unit tests
pytest tests/unit

# Use parallel execution
pip install pytest-xdist
pytest -n auto
```

### 4. ONNX/ML Library Issues
```bash
# Skip ML tests temporarily
pytest -k "not onnx and not embedding"

# Or use Docker where everything works
docker-compose exec api pytest
```

## Test Development Tips

1. **Start with unit tests** - They're fast and don't need external services
2. **Use fixtures** - Check `conftest.py` for available fixtures
3. **Mock external services** - Don't require Qdrant/API for unit tests
4. **Run frequently** - Use `pytest-watch` for automatic test runs

## Coverage Goals

- Unit tests: >80% coverage
- Integration tests: Cover all API endpoints
- Performance tests: Establish baselines

## Next Steps

1. Get any tests running first (even just one)
2. Fix failing tests incrementally
3. Add new tests as you develop features
4. Use CI/CD to ensure tests pass

Remember: Some tests are better than no tests! Start small and build up.
