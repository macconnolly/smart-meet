# 🧠 Cognitive Meeting Intelligence

> **Quick Start**: [CLAUDE_NAVIGATION.md](CLAUDE_NAVIGATION.md) | [Implementation Guide](IMPLEMENTATION_GUIDE.md) | [Developer Setup](DEVELOPER_SETUP.md)  
> **Tracking**: [Task System](docs/TASK_TRACKING_SYSTEM.md) | [Progress](docs/progress/)

> Transform your organization's meetings into a living, thinking memory that learns, connects, and discovers insights like the human mind.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-1.16.3-orange.svg)](https://onnxruntime.ai/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 What is This?

Cognitive Meeting Intelligence goes beyond simple transcript storage. It creates a **cognitive memory network** that mimics human memory processes:

- **🔍 Remembers** like you do - with context, importance, and connections
- **💡 Discovers** insights through serendipitous bridge connections
- **📈 Learns** over time by consolidating patterns into semantic knowledge
- **⚡ Retrieves** intelligently using activation spreading, not just keyword matching

### Key Differentiators

| Traditional Search | Cognitive Intelligence |
|-------------------|------------------------|
| Keyword matching | Semantic understanding |
| Isolated results | Connected memory networks |
| Static storage | Dynamic learning system |
| Simple retrieval | Activation spreading |
| No insight discovery | Bridge connections |

## 🚀 Features

### Core Capabilities

#### 1. **Multi-Dimensional Memory Extraction**
- **400D Vectors**: 384D semantic + 16D cognitive dimensions
  - 🕐 **Temporal** (4D): Urgency, deadlines, sequence, duration
  - 😊 **Emotional** (3D): Sentiment, intensity, confidence
  - 👥 **Social** (3D): Authority, audience, interaction
  - 🔗 **Causal** (3D): Cause-effect relationships
  - 📈 **Evolutionary** (3D): Change patterns over time

#### 2. **Two-Phase Activation Spreading**
```
Query → L0 Concepts → BFS Spreading → Activated Network
         ↓               ↓                ↓
    High-level      Connection      Core + Contextual
     matches         traversal        + Peripheral
```

#### 3. **Distance Inversion Bridge Discovery**
- Finds memories that are **distant from query** but **strongly connected** to activated set
- Enables serendipitous insights and unexpected connections
- "What you didn't know you needed to know"

#### 4. **Automated Memory Consolidation**
- **Episodic → Semantic**: Patterns become knowledge
- **DBSCAN Clustering**: Groups related memories
- **Decay & Reinforcement**: Important memories strengthen, irrelevant fade

### Performance Targets
- ⚡ **10-15 memories/second** extraction rate
- ⚡ **<100ms** embedding generation
- ⚡ **<2s** end-to-end cognitive query
- ⚡ **10K+ memories** with consistent performance

## 📦 Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- 8GB RAM minimum
- 50GB storage

### Quick Start

> **📢 Important**: For detailed implementation instructions, see [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)

```bash
# Clone repository
git clone https://github.com/your-org/cognitive-meeting-intelligence.git
cd cognitive-meeting-intelligence

# Run automated setup
python scripts/setup_all.py

# Or manual setup:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Initialize everything
make setup  # Runs all initialization scripts

# Run the API
make run

# Open browser
# http://localhost:8000/docs
```

## 🔧 Configuration

### Environment Variables
```bash
# .env
ENVIRONMENT=development
DATABASE_URL=sqlite:///./data/memories.db
QDRANT_HOST=localhost
QDRANT_PORT=6333
LOG_LEVEL=INFO
```

### Core Configuration
```yaml
# config/default.yaml
cognitive:
  activation:
    threshold: 0.7        # Minimum activation level
    max_activations: 50   # Maximum memories to activate
    decay_factor: 0.8     # Activation decay per hop
  
  bridges:
    novelty_weight: 0.6   # Weight for distance from query
    connection_weight: 0.4 # Weight for connection strength
    max_bridges: 5        # Maximum bridge memories
```

## 📡 API Usage

### Ingest a Meeting
```bash
curl -X POST http://localhost:8000/api/v2/meetings/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "John: We need to optimize our caching strategy...",
    "metadata": {
      "title": "Engineering Planning",
      "participants": ["John", "Mary", "Tom"],
      "start_time": "2024-01-15T10:00:00Z"
    }
  }'
```

### Cognitive Query
```bash
curl -X POST http://localhost:8000/api/v2/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What performance optimizations were discussed?",
    "enable_activation": true,
    "enable_bridges": true
  }'
```

### Response Example
```json
{
  "direct_results": [...],
  "activated_memories": {
    "core": [/* Highly relevant memories */],
    "contextual": [/* Related context */],
    "peripheral": [/* Distant but connected */]
  },
  "bridge_memories": [
    {
      "memory": {
        "content": "Database indexing improves query performance",
        "speaker": "Tom"
      },
      "bridge_score": 0.82,
      "explanation": "Novel perspective with strong connection"
    }
  ],
  "processing_time_ms": 1523.4
}
```

## 🏗️ Architecture

### System Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   FastAPI App   │────▶│ Cognitive Layer │────▶│  Storage Layer  │
│                 │     │                 │     │                 │
│ • REST API      │     │ • Activation    │     │ • SQLite (meta) │
│ • WebSockets*   │     │ • Bridges       │     │ • Qdrant (vec)  │
│ • Background    │     │ • Consolidation │     │ • Cache (LRU)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Extraction Layer│     │ Embedding Layer │     │   ML Runtime    │
│                 │     │                 │     │                 │
│ • Patterns      │     │ • ONNX Encoder  │     │ • all-MiniLM    │
│ • Dimensions    │     │ • Vector Mgmt   │     │ • VADER         │
│ • Classification│     │ • Composition   │     │ • Clustering    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Repository Structure
```
cognitive-meeting-intelligence/
├── src/
│   ├── core/               # Configuration, logging, caching
│   ├── models/             # Data models (Memory, Meeting, etc.)
│   ├── extraction/         # Memory extraction from transcripts
│   │   └── dimensions/     # Feature extractors (temporal, emotional, etc.)
│   ├── embedding/          # ONNX encoder & vector management
│   ├── cognitive/          # Cognitive algorithms
│   │   ├── activation/     # Spreading activation engine
│   │   ├── bridges/        # Bridge discovery engine
│   │   └── consolidation/  # Memory consolidation
│   ├── storage/            # Dual storage system
│   │   ├── sqlite/         # Metadata & relationships
│   │   └── qdrant/         # Vector storage (3 tiers)
│   ├── pipeline/           # Ingestion orchestration
│   └── api/                # FastAPI application
│       └── routers/        # API endpoints
├── tests/
│   ├── unit/               # Component tests
│   ├── integration/        # End-to-end tests
│   ├── performance/        # Benchmarks
│   └── fixtures/           # Test data
├── scripts/                # Setup and utility scripts
├── config/                 # Configuration files
├── models/                 # ONNX model storage
├── data/                   # Runtime data
├── docs/                   # Documentation
├── .env.example            # Environment template
├── docker-compose.yml      # Service orchestration
├── Makefile                # Common commands
├── requirements.txt        # Python dependencies
└── IMPLEMENTATION_GUIDE.md # 📢 START HERE - Step-by-step guide
```

### Data Flow
```
Meeting Transcript
       ↓
[Memory Extraction] ← Pattern matching for decisions/actions/insights
       ↓
[Dimension Analysis] ← Extract 16D cognitive features
       ↓
[Embedding Generation] ← Create 384D semantic vectors
       ↓
[Vector Composition] ← Combine into 400D vectors
       ↓
[Dual Storage] ← SQLite metadata + Qdrant vectors
       ↓
[Connection Creation] ← Build memory graph
       ↓
[Background Processing] ← Consolidation, decay, learning
```

## 🧪 Testing

### Run All Tests
```bash
# Unit tests with coverage
pytest --cov=src --cov-report=html

# Specific module
pytest tests/unit/test_activation.py -v

# Integration tests
pytest tests/integration/ -v

# Performance benchmarks
python tests/benchmarks/test_performance.py
```

### Code Quality
```bash
# Format code
black src/ tests/ --line-length 100

# Lint
flake8 src/ tests/ --max-line-length 100

# Type checking
mypy src/

# All checks
make quality
```

## 📊 Performance Monitoring

### Key Metrics
- **Query Latency**: p50, p95, p99
- **Memory Usage**: Process and cache size
- **Activation Depth**: Average spreading depth
- **Bridge Quality**: Novel insights ratio
- **Consolidation Rate**: Episodic→Semantic conversion

### Health Check
```bash
curl http://localhost:8000/api/v2/health

{
  "status": "healthy",
  "checks": {
    "api": "healthy",
    "database": "healthy",
    "qdrant": "healthy",
    "consolidation_scheduler": "running"
  },
  "metrics": {
    "total_memories": 12543,
    "avg_query_time_ms": 823.5,
    "cache_hit_rate": 0.73
  }
}
```

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Follow code style (Black, type hints, docstrings)
4. Write tests (maintain >90% coverage)
5. Run quality checks: `make quality`
6. Commit: `git commit -m 'Add amazing feature'`
7. Push: `git push origin feature/amazing-feature`
8. Open Pull Request

### For AI Assistants
See [CLAUDE.md](CLAUDE.md) for comprehensive context and guidelines.

## 📚 Documentation

### For Developers
- [Technical Specification](docs/technical-implementation.md)
- [Phase 1 Implementation](docs/phase1-implementation.md)
- [Phase 2 Implementation](docs/phase2-implementation.md)
- [Autonomous Implementation Guide](docs/AUTONOMOUS_IMPLEMENTATION_GUIDE.md)

### For AI Implementation
- [CLAUDE.md](CLAUDE.md) - AI assistant context
- [API Documentation](http://localhost:8000/docs) - Interactive API docs

### Architecture Decisions
- [Why ONNX?](docs/decisions/001-onnx-runtime.md)
- [Why Qdrant?](docs/decisions/002-vector-database.md)
- [Why 400D Vectors?](docs/decisions/003-vector-dimensions.md)

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t cognitive-meeting:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e ENVIRONMENT=production \
  cognitive-meeting:latest
```

### Kubernetes
```bash
# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -n cognitive-meeting
```

## 📈 Implementation Status

> **🔨 Current Phase**: Setting up project structure
> 
> **📚 For Developers**: See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for day-by-day implementation plan

### Phase 1: Foundation (Week 1) - MVP
- [ ] Day 1: Core Models & Database
- [ ] Day 2: Embeddings Infrastructure  
- [ ] Day 3: Vector Management & Dimensions
- [ ] Day 4: Storage Layer (SQLite + Qdrant)
- [ ] Day 5: Extraction Pipeline
- [ ] Day 6-7: API & Integration

### Phase 2: Cognitive Features (Week 2)
- [ ] Activation spreading engine
- [ ] Bridge discovery algorithm
- [ ] Path tracking
- [ ] Memory classification

### Phase 3: Advanced Features (Week 3)
- [ ] Memory consolidation
- [ ] Lifecycle management
- [ ] Background tasks
- [ ] Performance optimization

### Phase 4: Production Ready (Week 4)
- [ ] Security implementation
- [ ] Monitoring & observability
- [ ] Deployment automation
- [ ] Documentation completion

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by cognitive science research on human memory
- Built with FastAPI, ONNX Runtime, and Qdrant
- Embeddings from Sentence Transformers

---

<p align="center">
  <i>Built with ❤️ to make meetings memorable and insightful</i>
</p>
