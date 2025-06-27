# Developer Quick Reference

## 🚀 Essential Commands

```bash
# Activate environment
source venv/bin/activate

# Start services
docker-compose up -d
make run

# Run tests
make test
pytest -xvs

# Format & lint
make format
make lint

# Full check
make check
```

## 🔧 Common Tasks

### Before Committing
```bash
# Run all checks
pre-commit run --all-files

# Or manually
black src/ tests/
flake8 src/ tests/
mypy src/
pytest
```

### API Development
```bash
# Start API with reload
uvicorn src.api.cognitive_api:app --reload

# Test endpoint
curl -X POST http://localhost:8000/api/v2/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was discussed?"}'
```

### Database Reset
```bash
# Full reset
make clean-db
make init-db
```

## 📝 Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Commit with message
git add .
git commit -m "feat: add new feature"

# Push and create PR
git push -u origin feature/my-feature
gh pr create
```

## 🐛 Debugging

### Check Services
```bash
# Qdrant status
curl http://localhost:6333/collections

# View logs
docker-compose logs -f qdrant

# Python debugging
ipython
import src.models.entities as models
```

### Common Fixes
```bash
# SSL issues
export REQUESTS_CA_BUNDLE=/mnt/c/Users/EL436GA/dev/meet/combined-ca-bundle.crt

# Import issues  
export PYTHONPATH=$PWD:$PYTHONPATH

# Permission issues
chmod +x scripts/*.py
```

## 🔍 Testing Specific Components

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## 📊 Performance

```bash
# Profile endpoint
python -m cProfile -o profile.stats src/api/cognitive_api.py

# Memory usage
python -m memory_profiler scripts/memory_test.py

# Load test
locust -f tests/performance/locustfile.py
```

## 🚨 Emergency Commands

```bash
# Kill all Python processes
pkill -f python

# Stop all containers
docker-compose down

# Clean everything
make clean
rm -rf venv/
python3 -m venv venv
```

## 💡 VS Code Shortcuts

- `Ctrl+Shift+P`: Command palette
- `F5`: Start debugging
- `Ctrl+Shift+~`: New terminal
- `Ctrl+P`: Quick file open
- `F12`: Go to definition
- `Shift+F12`: Find references

## 📋 Makefile Targets

```bash
make help      # Show all targets
make setup     # Initial setup
make run       # Run API
make test      # Run tests
make lint      # Run linters
make format    # Format code
make check     # Run all checks
make clean     # Clean artifacts
make docker    # Build Docker image
```

---
*Keep this handy for quick development tasks!*