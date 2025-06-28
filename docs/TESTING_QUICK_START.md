# Testing Quick Start Guide

## Overview

The Cognitive Meeting Intelligence system uses a comprehensive testing strategy with:
- **Unit tests** for individual components (70% of tests)
- **Integration tests** for component interactions (25% of tests)
- **End-to-end tests** for complete workflows (5% of tests)

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -c "import nltk; nltk.download('vader_lexicon')"

# Start required services
docker-compose up -d
```

### 2. Run Tests

```bash
# Quick unit tests (no external dependencies)
pytest tests/unit -v

# Run with coverage
pytest tests/unit --cov=src --cov-report=term-missing

# Run integration tests (requires Qdrant)
pytest tests/integration -m integration -v

# Run all tests
pytest

# Run specific test
pytest tests/unit/extraction/test_temporal_extractor.py -v
```

### 3. Using the Test Runner

```bash
# Make test runner executable
chmod +x run_tests.py

# Run different test suites
./run_tests.py unit          # Fast unit tests
./run_tests.py integration   # Integration tests
./run_tests.py coverage      # Generate coverage report
./run_tests.py watch         # Watch mode
./run_tests.py specific -t tests/unit/test_models.py
```

### 4. Using Make Commands

```bash
# If you have Make installed
make test              # Run unit tests
make test-all          # Run all tests
make coverage          # Generate coverage report
make test-watch        # Run in watch mode
make docker-up         # Start services
make clean             # Clean generated files
```

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests (fast, isolated)
│   ├── extraction/         
│   │   ├── test_temporal_extractor.py
│   │   ├── test_emotional_extractor.py
│   │   └── test_memory_extractor.py
│   ├── cognitive/
│   │   ├── test_activation_engine.py
│   │   └── test_bridge_discovery.py
│   └── embedding/
│       └── test_vector_manager.py
├── integration/             # Integration tests
│   ├── test_pipeline.py
│   ├── test_cognitive_integration.py
│   └── test_storage_integration.py
└── performance/            # Performance tests
    └── test_performance_targets.py
```

## Writing Tests

### Unit Test Example

```python
import pytest
from src.extraction.dimensions.temporal_extractor import TemporalDimensionExtractor

class TestTemporalExtractor:
    @pytest.fixture
    def extractor(self):
        return TemporalDimensionExtractor()
    
    def test_urgency_detection(self, extractor):
        result = extractor.extract("This is URGENT!")
        assert result.urgency == 1.0
```

### Integration Test Example

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline(full_pipeline, sample_transcript):
    # Ingest transcript
    result = await full_pipeline.ingest(
        transcript=sample_transcript,
        meeting_id="test-001"
    )
    
    # Verify complete flow
    assert result.memories_extracted > 0
    assert result.vectors_stored == result.memories_extracted
```

## Test Fixtures

Key fixtures available in `conftest.py`:

- `test_db` - Temporary SQLite database
- `mock_vector_store` - Mocked Qdrant store
- `sample_transcript` - Test meeting transcript
- `sample_memories` - Pre-created memory objects
- `populated_db` - Database with test data

## Debugging Tests

```bash
# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s

# Verbose output with local variables
pytest -vv -l

# Run single test
pytest path/to/test.py::TestClass::test_method
```

## Coverage Requirements

- Minimum coverage: 80%
- Critical paths: 95%+
- View coverage report: `open htmlcov/index.html`

## CI/CD Integration

Tests run automatically on:
- Every push to main
- Pull requests
- Can be triggered manually

See `.github/workflows/test.yml` for CI configuration.

## Performance Testing

```bash
# Run performance tests
pytest -m performance -v

# Key metrics tested:
# - Memory extraction: 10-15/second
# - Query latency: <2 seconds
# - Activation spreading: <500ms
```

## Troubleshooting

### Qdrant not running
```bash
docker-compose up -d
# Wait for startup
curl http://localhost:6333/collections
```

### Import errors
```bash
# Fix import paths
python fix_all_imports.py
```

### VADER lexicon missing
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

### Tests hanging
- Check for missing `await` in async tests
- Ensure proper cleanup in fixtures
- Use `pytest-timeout` for long tests

## Best Practices

1. **Keep tests fast** - Mock external dependencies
2. **Test one thing** - Each test should verify single behavior
3. **Use fixtures** - Don't repeat setup code
4. **Clear names** - Test names should describe what they test
5. **Arrange-Act-Assert** - Structure tests clearly
6. **Isolate tests** - No dependencies between tests
7. **Mock sparingly** - Only mock external dependencies

## Next Steps

1. Run the basic tests to ensure setup works
2. Write tests for new features before implementing
3. Maintain >80% coverage
4. Run integration tests before deploying
5. Monitor performance test trends