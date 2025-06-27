# Development Environment Setup Guide

## 🚀 Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Install pre-commit hooks
pre-commit install

# 4. Start services
docker-compose up -d  # Qdrant vector database
python scripts/init_db.py  # Initialize SQLite
python scripts/init_qdrant.py  # Create collections

# 5. Run the API
uvicorn src.api.cognitive_api:app --reload
```

## 📋 Environment Configuration

### Python Version
- **Required**: Python 3.11.4 (specified in `.python-version`)
- **Virtual Environment**: Located at `./venv`

### SSL/Certificate Setup (Corporate Proxy)
The project is configured to work with Zscaler corporate proxy:
```bash
# Certificate is already configured in pip
pip config list  # Should show: global.cert='/mnt/c/Users/EL436GA/dev/meet/combined-ca-bundle.crt'
```

### VS Code Configuration
1. **Interpreter Path**: 
   - WSL: `/home/mac/dev/meet/venv/bin/python`
   - Windows: `\\wsl$\Ubuntu\home\mac\dev\meet\venv\bin\python`

2. **Workspace Settings**: Configured in `.vscode/settings.json`
   - Auto-formatting with Black (100 char line length)
   - Linting with Flake8 and MyPy
   - Test discovery with pytest
   - Integrated debugging configurations

3. **Recommended Extensions**: See `.vscode/extensions.json`

## 🛠️ Development Tools

### Code Quality Tools
- **Black**: Code formatting (line length: 100)
- **Flake8**: Linting with docstring checks
- **MyPy**: Static type checking
- **Bandit**: Security linting
- **isort**: Import sorting

### Testing
- **pytest**: Test framework with async support
- **pytest-cov**: Coverage reporting (minimum 80%)
- **pytest-asyncio**: Async test support
- **tox**: Multi-environment testing

### Pre-commit Hooks
Configured in `.pre-commit-config.yaml`:
- Trailing whitespace removal
- YAML/JSON validation
- Python formatting (Black)
- Import sorting (isort)
- Linting (Flake8)
- Type checking (MyPy)
- Security checks (Bandit)
- Markdown linting
- Dockerfile linting

Run manually: `pre-commit run --all-files`

## 🚦 CI/CD Pipeline

### GitHub Actions Workflows

1. **CI Pipeline** (`.github/workflows/ci.yml`)
   - Linting & code quality checks
   - Unit and integration tests
   - Coverage reporting
   - Docker image building
   - Documentation building

2. **Security Scanning** (`.github/workflows/security.yml`)
   - CodeQL analysis
   - Dependency vulnerability checks
   - Container scanning with Trivy
   - Secret scanning with TruffleHog
   - SAST with Bandit

3. **Dependency Updates** (`.github/workflows/dependency-update.yml`)
   - Weekly automated dependency updates
   - Pre-commit hook updates
   - Creates PRs for review

4. **Release Pipeline** (`.github/workflows/release.yml`)
   - Automated releases on tags
   - PyPI publishing
   - Docker image publishing
   - Documentation deployment

## 📦 Project Structure

```
meet/
├── .github/workflows/    # CI/CD pipelines
├── .vscode/             # VS Code configuration
├── data/                # Local data storage
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── src/                 # Source code
│   ├── api/            # FastAPI endpoints
│   ├── cognitive/      # Core algorithms
│   ├── embedding/      # Vector embeddings
│   ├── extraction/     # Memory extraction
│   ├── models/         # Data models
│   └── storage/        # Storage layer
├── tests/              # Test suite
├── .coveragerc         # Coverage configuration
├── .dockerignore       # Docker ignore rules
├── .editorconfig       # Editor configuration
├── .env.example        # Environment template
├── .gitignore          # Git ignore rules
├── .pre-commit-config.yaml  # Pre-commit hooks
├── .python-version     # Python version
├── .yamllint           # YAML linting rules
├── Dockerfile          # Container definition
├── Makefile            # Build automation
├── docker-compose.yml  # Service orchestration
├── pyproject.toml      # Python project config
├── pytest.ini          # Pytest configuration
├── requirements.txt    # Production dependencies
├── requirements-dev.txt # Development dependencies
├── setup.cfg           # Setup configuration
├── setup.py            # Package setup
└── tox.ini            # Tox configuration
```

## 🧪 Testing

### Run Tests
```bash
# All tests with coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/unit/test_models.py

# With markers
pytest -m unit  # Unit tests only
pytest -m integration  # Integration tests only

# Parallel execution
pytest -n auto
```

### Test Coverage
- Minimum coverage: 80%
- Coverage reports: `htmlcov/index.html`
- Configuration: `.coveragerc` and `pytest.ini`

### Tox Environments
```bash
tox              # Run all environments
tox -e lint      # Linting only
tox -e type      # Type checking only
tox -e py311     # Python 3.11 tests
tox -e docs      # Build documentation
```

## 🏃 Common Development Tasks

### Code Formatting
```bash
# Format code
black src/ tests/
isort src/ tests/

# Check formatting
black --check src/ tests/
isort --check-only src/ tests/
```

### Linting
```bash
# Run all linters
flake8 src/ tests/
mypy src/
bandit -r src/

# Or use Make
make lint
```

### Database Operations
```bash
# Initialize databases
python scripts/init_db.py
python scripts/init_qdrant.py

# Reset databases
rm data/memories.db
docker-compose down -v
docker-compose up -d
```

### API Development
```bash
# Run with auto-reload
uvicorn src.api.cognitive_api:app --reload --host 0.0.0.0 --port 8000

# Access API docs
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

## 🐛 Debugging

### VS Code Debug Configurations
Available in `.vscode/launch.json`:
- **Python: FastAPI** - Debug the API server
- **Python: Current File** - Debug current file
- **Python: Test Current File** - Debug specific test
- **Python: All Tests** - Debug all tests
- **Python: Test with Coverage** - Debug with coverage

### Common Issues

#### SSL Certificate Errors
```bash
# Verify certificate configuration
pip config list

# Should show:
# global.cert='/mnt/c/Users/EL436GA/dev/meet/combined-ca-bundle.crt'
```

#### Import Errors
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH=/mnt/c/Users/EL436GA/dev/meet:$PYTHONPATH
```

#### Qdrant Connection Issues
```bash
# Check Qdrant is running
docker ps | grep qdrant
curl http://localhost:6333/collections

# Restart if needed
docker-compose restart qdrant
```

## 📊 Performance Profiling

### CPU Profiling
```bash
python -m cProfile -o profile.stats src/api/cognitive_api.py
python -m pstats profile.stats
```

### Memory Profiling
```bash
python -m memory_profiler src/extraction/ingestion.py
```

### Load Testing
```bash
locust -f tests/performance/locustfile.py --host http://localhost:8000
```

## 🚀 Deployment

### Docker Build
```bash
# Build image
docker build -t cognitive-meeting-intelligence .

# Run container
docker run -p 8000:8000 --env-file .env cognitive-meeting-intelligence
```

### Production Checklist
- [ ] Update version in `setup.py`
- [ ] Run full test suite
- [ ] Update CHANGELOG.md
- [ ] Create git tag
- [ ] Build and test Docker image
- [ ] Run security scans
- [ ] Deploy to staging
- [ ] Run smoke tests
- [ ] Deploy to production

## 📚 Additional Resources

- [Project Documentation](../README.md)
- [API Documentation](http://localhost:8000/docs)
- [Architecture Overview](./architecture.md)
- [Contributing Guidelines](./CONTRIBUTING.md)

## 🆘 Getting Help

1. Check existing documentation
2. Look for similar implementations in codebase
3. Review test examples
4. Check GitHub issues
5. Ask in team chat

---

*Last updated: {{ date }}*