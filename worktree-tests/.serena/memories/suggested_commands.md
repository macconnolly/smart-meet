# Suggested Commands

## Project Setup
```bash
# Create project structure
mkdir cognitive-meeting-mvp
cd cognitive-meeting-mvp
mkdir -p src/{core,models,extraction,embedding,cognitive,storage,api,cli}
mkdir -p src/extraction/dimensions
mkdir -p src/cognitive/{activation,bridges,consolidation}
mkdir -p src/storage/{sqlite,qdrant}
mkdir -p src/storage/sqlite/repositories
mkdir -p src/api/routers
mkdir -p tests/{unit,integration,fixtures}
mkdir -p scripts
mkdir -p models/all-MiniLM-L6-v2
mkdir -p data/{qdrant,transcripts}
mkdir -p config
mkdir -p docs

# Create __init__.py files
find src -type d -exec touch {}/__init__.py \;
touch tests/__init__.py

# Virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Database & Storage Setup
```bash
# Start Qdrant
docker-compose up -d

# Initialize SQLite database
python scripts/init_db.py

# Initialize Qdrant collections
python scripts/init_qdrant.py

# Verify database
sqlite3 data/memories.db ".tables"
```

## Model Setup
```bash
# Download and convert model to ONNX
python scripts/download_model.py

# Verify model
ls -la models/all-MiniLM-L6-v2/
```

## Running the Application
```bash
# Development mode with auto-reload
uvicorn src.api.simple_api:app --reload --port 8000

# Phase 2 API (cognitive features)
uvicorn src.api.cognitive_api:app --reload --port 8000

# Production mode
uvicorn src.api.cognitive_api:app --host 0.0.0.0 --port 8000 --workers 4

# With Docker
docker-compose up --build
```

## Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_memory_repo.py

# Run integration tests
pytest tests/test_phase1_integration.py
pytest tests/test_phase2_integration.py

# Run async tests
pytest tests/test_activation.py -v
pytest tests/test_bridges.py -v
pytest tests/test_consolidation.py -v
```

## Code Quality
```bash
# Format code with Black
black src/ tests/ --line-length 100

# Lint with flake8
flake8 src/ tests/ --max-line-length 100

# Type checking with mypy
mypy src/

# All quality checks
black src/ tests/ --line-length 100 && flake8 src/ tests/ --max-line-length 100 && mypy src/
```

## API Testing
```bash
# Health check
curl http://localhost:8000/health

# Ingest meeting (Phase 1)
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "John: We decided to implement caching.",
    "metadata": {"title": "Tech Meeting", "participants": ["John"]}
  }'

# Search memories (Phase 1)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "caching decision", "limit": 10}'

# Cognitive query (Phase 2)
curl -X POST http://localhost:8000/api/v2/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What performance optimizations were discussed?",
    "enable_activation": true,
    "enable_bridges": true,
    "max_results": 20
  }'

# Trigger consolidation
curl -X POST http://localhost:8000/api/v2/consolidate

# Get performance stats
curl http://localhost:8000/api/v2/stats/performance
```

## Database Management
```bash
# SQLite queries
sqlite3 data/memories.db "SELECT COUNT(*) FROM memories;"
sqlite3 data/memories.db "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type;"
sqlite3 data/memories.db "SELECT * FROM memories WHERE content_type='decision' LIMIT 5;"

# Backup database
cp data/memories.db data/memories_backup_$(date +%Y%m%d).db

# Clean old memories (careful!)
sqlite3 data/memories.db "DELETE FROM memories WHERE importance_score < 0.1 AND created_at < date('now', '-30 days');"
```

## Docker Operations
```bash
# Build image
docker build -t cognitive-meeting:latest .

# Run container
docker run -p 8000:8000 -v $(pwd)/data:/app/data cognitive-meeting:latest

# View logs
docker-compose logs -f

# Clean up
docker-compose down
docker system prune -f
```

## Performance Testing
```bash
# Simple load test with curl
for i in {1..10}; do
  time curl -X POST http://localhost:8000/api/v2/query \
    -H "Content-Type: application/json" \
    -d '{"query": "test query '$i'", "enable_activation": true}'
done

# Profile memory usage
python -m memory_profiler src/api/cognitive_api.py
```

## Debugging
```bash
# Run with debug logging
LOG_LEVEL=DEBUG uvicorn src.api.cognitive_api:app --reload

# Python debugger
python -m pdb scripts/test_activation.py

# Check Qdrant collections
curl http://localhost:6333/collections

# Check specific collection
curl http://localhost:6333/collections/cognitive_episodes
```

## Maintenance
```bash
# Clean cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Update dependencies
pip install --upgrade -r requirements.txt

# Vacuum SQLite database
sqlite3 data/memories.db "VACUUM;"
```