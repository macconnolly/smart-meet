# System Architecture Overview

> **Navigation**: [Home](../../README.md) → [Implementation Guide](../../IMPLEMENTATION_GUIDE.md) → System Overview  
> **Related**: [Data Flow](data-flow.md) | [Vector Architecture](vector-architecture.md)

## Overview

The Cognitive Meeting Intelligence system is built on a layered architecture that separates concerns and enables scalability. The system is enhanced with strategy consulting features including project management, stakeholder tracking, and deliverable management.

## Architecture Layers

### 1. API Layer (FastAPI)
- RESTful endpoints for all operations
- WebSocket support for real-time updates (future)
- Automatic API documentation
- Request validation with Pydantic
- Project-scoped endpoints for consulting context

### 2. Cognitive Layer
- **Activation Engine**: Implements two-phase BFS spreading with project context
- **Bridge Engine**: Discovers serendipitous connections across deliverables
- **Consolidation Engine**: Creates semantic memories from episodic
- **Consulting Intelligence**: Stakeholder influence, deliverable networks

### 3. Extraction Layer
- **Memory Extractor**: Identifies memories in transcripts
- **Dimension Analyzers**: Extract 16D cognitive features
- **Classification**: Categorizes content types (14 types including consulting-specific)
- **Priority Detection**: Identifies critical items and ownership

### 4. Embedding Layer
- **ONNX Encoder**: Generates 384D semantic embeddings
- **Vector Manager**: Composes 400D vectors
- **Caching**: LRU cache for performance
- **Batch Processing**: Efficient for large transcripts

### 5. Storage Layer
- **SQLite**: Enhanced schema with 7+ tables for consulting
  - Projects, Stakeholders, Deliverables
  - Enhanced Meetings and Memories
  - Rich relationships and views
- **Qdrant**: Vector storage with 3 tiers
- **Cache**: In-memory LRU cache

## Key Design Decisions

### Why Enhanced Schema from Start?
- Avoid costly migrations later
- Project context is fundamental to consulting
- Stakeholder tracking essential for insights
- Deliverable linking creates value

### Why Dual Storage?
- SQLite excels at relational queries and ACID compliance
- Qdrant optimizes vector similarity search
- Separation allows independent scaling
- Complex consulting relationships need SQL

### Why 3-Tier Vector Storage?
- L0 (Concepts): Highest abstractions, rarely accessed
- L1 (Contexts): Consolidated patterns, moderate access
- L2 (Episodes): Raw memories, frequent access
- Different HNSW parameters optimize each tier

### Why ONNX Runtime?
- 5-10x faster than PyTorch for inference
- Smaller deployment footprint
- Cross-platform compatibility
- Production-ready performance

## Data Flow

### 1. Project Setup
- Create project with client details
- Define stakeholders and influence
- Plan deliverables and milestones

### 2. Meeting Ingestion
- Categorize meeting type (client/internal)
- Extract memories with priority detection
- Link to deliverables and stakeholders
- Create connections within project scope

### 3. Cognitive Query
- Project-scoped search
- Stakeholder-filtered activation
- Deliverable network traversal
- Cross-project insights (optional)

### 4. Background Processing
- Consolidation within projects
- Stakeholder mention indexing
- Deliverable progress tracking
- Priority item monitoring

## Consulting-Specific Features

### Project Management
- Multi-project support with isolation
- Budget and resource tracking
- Timeline and milestone management
- Engagement lifecycle

### Stakeholder Intelligence
- Influence and engagement tracking
- Sentiment analysis per stakeholder
- Decision-maker identification
- Resistance pattern detection

### Deliverable Tracking
- Dependency management
- Version control integration
- Progress monitoring
- Risk association

### Meeting Intelligence
- Client vs internal classification
- Decision extraction and tracking
- Action item ownership
- Hypothesis-evidence linking

## Scalability Considerations

- Async operations throughout
- Connection pooling for databases
- Batch processing where possible
- Project-based partitioning
- Horizontal scaling via load balancing
- Vector quantization (future)

## Security Model

- Project-based access control
- Input validation at API layer
- Parameterized queries prevent SQL injection
- Rate limiting on endpoints
- JWT authentication with project scopes (planned)
- Audit logging for compliance
- Client data isolation

## Performance Targets

| Component | Target | Rationale |
|-----------|--------|-----------|
| Embedding | <100ms | Real-time feel |
| Search | <200ms | Interactive queries |
| Activation | <500ms | Complex but fast |
| Full Query | <2s | End-to-end |
| Project Switch | <100ms | Context switching |

## Database Schema Highlights

### Core Tables
1. **Projects**: Client engagements with metadata
2. **Meetings**: Enhanced with types and categories
3. **Memories**: 14 content types, priority, ownership
4. **Stakeholders**: Influence and engagement tracking
5. **Deliverables**: Status and dependency management
6. **Connections**: 10 types including consulting-specific

### Key Relationships
- All data scoped by project
- Memories linked to deliverables
- Stakeholder mentions tracked
- Meeting series for recurring events

### Useful Views
- `project_memory_summary`: Overview per project
- `stakeholder_engagement`: Influence analysis
- `deliverable_status`: Progress tracking
- `high_priority_items`: Critical item dashboard

## Future Enhancements

1. **Real-time Processing**: WebSocket for live transcription
2. **Multi-tenancy**: Enhanced project isolation
3. **Advanced Analytics**: Cross-project insights
4. **AI Summarization**: GPT-4 for deliverable generation
5. **Mobile Apps**: On-the-go access for consultants
6. **Integration Hub**: CRM, project management tools
