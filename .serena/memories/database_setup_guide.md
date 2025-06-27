# Database Setup Guide for Cognitive Meeting Intelligence

## Overview
The system uses a dual database architecture:
1. **SQLite** - Embedded database for metadata and relationships
2. **Qdrant** - Vector database for 400D cognitive embeddings

## Quick Start Commands

```bash
# 1. Start Qdrant (required)
docker-compose up -d

# 2. Verify Qdrant is running
docker ps  # Should show cognitive_qdrant on ports 6333-6334

# 3. Initialize databases (first time only)
python scripts/init_db.py      # Creates SQLite at data/memories.db
python scripts/init_qdrant.py  # Creates 3 vector collections

# 4. Verify setup
sqlite3 data/memories.db ".tables"  # Should list 5 tables
curl http://localhost:6333/collections  # Should show 3 collections
```

## SQLite Schema (5 tables)

### Core Tables
- **meetings** - Meeting metadata, participants, transcript paths
- **memories** - Individual memory records with cognitive metadata
- **memory_connections** - Relationships between memories
- **search_history** - Query tracking for analytics
- **system_metadata** - System configuration and state

### Key Fields in memories table
- `id`, `meeting_id`, `content`, `speaker`, `timestamp_ms`
- `memory_type` (decision/action/idea/issue/question/context)
- `level` (0=concepts, 1=semantic, 2=episodic)
- `qdrant_id` - Links to vector in Qdrant
- `dimensions_json` - 16D cognitive features
- `decay_rate` - L2=0.1, L1=0.01, L0=0.001

## Qdrant Collections (3 tiers)

### Memory Hierarchy
1. **cognitive_episodes** (L2)
   - Raw meeting memories
   - Fast decay (0.1)
   - HNSW: m=16, ef=200

2. **cognitive_contexts** (L1)
   - Consolidated patterns
   - Slow decay (0.01)
   - HNSW: m=24, ef=300

3. **cognitive_concepts** (L0)
   - Highest abstractions
   - Minimal decay (0.001)
   - HNSW: m=32, ef=400

### Vector Composition (400D)
- **384D**: Semantic embedding from all-MiniLM-L6-v2
- **16D**: Cognitive dimensions
  - Temporal (4D): urgency, deadline, sequence, duration
  - Emotional (3D): sentiment, intensity, confidence
  - Social (3D): authority, audience, interaction
  - Causal (3D): cause-effect relationships
  - Evolutionary (3D): change patterns

## Common Operations

### Check Database Health
```bash
# SQLite
sqlite3 data/memories.db "SELECT COUNT(*) FROM memories;"

# Qdrant
curl http://localhost:6333/telemetry

# Docker logs
docker logs cognitive_qdrant
```

### Reset Databases
```bash
# Stop Qdrant
docker-compose down

# Remove data
rm -rf data/memories.db
docker volume rm meet_qdrant_data

# Restart fresh
docker-compose up -d
python scripts/init_db.py
python scripts/init_qdrant.py
```

## Development Tips
1. SQLite file is at `data/memories.db` - no service needed
2. Qdrant requires Docker - always start with `docker-compose up -d`
3. Use connection pooling for SQLite in production
4. Qdrant collections are created with optimal HNSW settings
5. Both databases must be initialized before running the application

## Troubleshooting
- **Qdrant not starting**: Check Docker is running, ports 6333-6334 are free
- **SQLite locked**: Close other connections, use connection pooling
- **Vector dimension mismatch**: Ensure 384D + 16D = 400D total
- **Collections missing**: Run `python scripts/init_qdrant.py`