# Project Overview - Cognitive Meeting Intelligence

## Core Purpose
Transform organizational meetings into a queryable, thinking memory network that provides intelligent retrieval through activation spreading and bridge discovery.

## Key Features
- **Multi-dimensional Memory Extraction**: 400D vectors (384D semantic + 16D features)
  - Temporal (4D): urgency, deadline proximity, sequence position, duration
  - Emotional (3D): sentiment, intensity, confidence (VADER)
  - Social (3D): authority, audience relevance, interaction
  - Causal (3D): placeholder implementation
  - Evolutionary (3D): placeholder implementation

- **Two-Phase BFS Activation Spreading**
  - Phase 1: Query L0 concepts for high-level matches
  - Phase 2: BFS through connection graph with decay
  - Path tracking for activation transparency
  - Core/contextual/peripheral classification

- **Distance Inversion Bridge Discovery**
  - Finds memories distant from query but connected to activated set
  - Novelty scoring (1 - similarity)
  - Connection potential calculation
  - Serendipitous insight generation

- **Dual Memory System**
  - Episodic memories: Raw meeting content (L2)
  - Semantic memories: Consolidated patterns (L1)
  - Concepts: Highest abstractions (L0)
  - Automated consolidation pipeline

- **Memory Lifecycle Management**
  - Importance decay with reinforcement
  - Access-based consolidation
  - Stale memory cleanup
  - Episodic→Semantic promotion

## Performance Targets
- Memory extraction: 10-15 memories/second
- Embedding generation: <100ms per memory
- Full cognitive query: <2s end-to-end
- Support for 10K+ memories with consistent performance

## System Architecture
- 3-tier Qdrant collections (L0/L1/L2) with HNSW optimization
- SQLite for metadata, relationships, and statistics
- ONNX Runtime for efficient embeddings
- FastAPI for unified cognitive API
- Background services for consolidation and lifecycle

## Implementation Phases
- **Phase 1 (Weeks 1-4)**: Foundation - SQLite, ONNX, Qdrant, basic ingestion
- **Phase 2 (Weeks 5-8)**: Cognitive features - activation, bridges, consolidation, full API