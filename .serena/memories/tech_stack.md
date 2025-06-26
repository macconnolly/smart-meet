# Technology Stack

## Core Technologies
- **Python**: 3.11+ (primary language)
- **Runtime Environment**: Docker & Docker Compose

## API & Web Framework
- **FastAPI**: 0.109.0 - High-performance async API framework
- **Uvicorn**: 0.27.0 - ASGI server with standard extras
- **Pydantic**: 2.5.0 - Data validation and settings
- **Pydantic Settings**: 2.1.0 - Configuration management

## Vector & Storage
- **Qdrant**: 1.7.0 - Vector database (local mode)
  - 3-tier collections: cognitive_concepts (L0), cognitive_contexts (L1), cognitive_episodes (L2)
  - HNSW indexing with optimized parameters
- **SQLite**: Metadata and relationship storage
  - Tables: meetings, memories, memory_connections, bridge_cache, retrieval_stats
- **SQLAlchemy**: 2.0.25 - ORM and database toolkit
- **Alembic**: 1.13.1 - Database migrations

## Machine Learning & NLP
- **ONNX Runtime**: 1.16.3 - Optimized inference
- **Sentence Transformers**: 2.2.2 - Model management
- **Model**: all-MiniLM-L6-v2 (384D embeddings)
- **NumPy**: 1.26.3 - Numerical computing
- **scikit-learn**: 1.4.0 - Clustering (DBSCAN) and metrics
- **NLTK**: 3.8.1 - NLP utilities
- **VaderSentiment**: 3.3.2 - Emotional dimension analysis

## Utilities
- **python-dateutil**: 2.8.2 - Date parsing
- **python-jose**: 3.3.0 - JWT tokens (with cryptography)
- **python-multipart**: 0.0.6 - Form data parsing
- **python-dotenv**: 1.0.0 - Environment management

## Testing & Development
- **pytest**: 7.4.4 - Testing framework
- **pytest-asyncio**: 0.23.3 - Async test support
- **pytest-cov**: 4.1.0 - Coverage reporting
- **black**: 23.12.1 - Code formatter
- **flake8**: 7.0.0 - Linter
- **mypy**: 1.8.0 - Type checker

## Key Libraries by Feature
- **Embeddings**: ONNX Runtime + Sentence Transformers
- **Vector Search**: Qdrant Client
- **Dimensional Analysis**: VADER (emotional), regex patterns (temporal/social)
- **Clustering**: scikit-learn DBSCAN
- **Async Operations**: asyncio + FastAPI
- **Caching**: In-memory LRU cache with TTL

## Configuration
- YAML-based configuration (config/default.yaml)
- Environment variables (.env)
- Docker environment for deployment