# Technical Architecture Specification
## Cognitive Meeting Intelligence System v1.0

*End of Technical Architecture Specification v1.0*

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architecture Principles](#3-architecture-principles)
4. [System Architecture](#4-system-architecture)
5. [Data Architecture](#5-data-architecture)
6. [Cognitive Processing Architecture](#6-cognitive-processing-architecture)
7. [API Architecture](#7-api-architecture)
8. [Security Architecture](#8-security-architecture)
9. [Performance Architecture](#9-performance-architecture)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Operational Architecture](#11-operational-architecture)
12. [Evolution Strategy](#12-evolution-strategy)

---

## 1. Executive Summary

### 1.1 Purpose

This document defines the technical architecture for a Cognitive Meeting Intelligence System that transforms organizational meetings into a queryable, thinking memory network. The system employs cognitive computing principles to create an active memory system that can understand context, discover hidden connections, and provide intelligent insights about organizational knowledge.

### 1.2 Scope

The architecture covers:
- Meeting transcript ingestion and memory extraction
- Multi-dimensional cognitive analysis
- Vector-based similarity search with activation spreading
- Bridge discovery for non-obvious connections
- Temporal evolution tracking
- Commitment and decision management
- Five primary use cases (UC1-UC5)

### 1.3 Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Vector Database** | Qdrant | Native hierarchical collections, production-ready, excellent Python support |
| **ML Runtime** | ONNX Runtime | Eliminates PyTorch dependency, 920MB smaller containers, CPU-optimized |
| **Persistence** | SQLite + JSON | Zero configuration, ACID compliance, embedded deployment |
| **Embedding Model** | all-MiniLM-L6-v2 | Fast inference (384D), good quality/performance ratio, ONNX-optimized |
| **Architecture Pattern** | Memory-Centric | Treats information as interconnected memories, not documents |
| **Processing Model** | Local-First | All ML inference local, no external API dependencies |

### 1.4 Design Constraints

- **Performance**: <2 second query response time for complex queries
- **Scale**: Support for 1M+ memories per deployment
- **Deployment**: Single container deployment option
- **Privacy**: All processing must be local
- **Simplicity**: Minimal operational overhead

---

## 2. System Overview

### 2.1 System Context

The Cognitive Meeting Intelligence System operates within an organization's meeting ecosystem, ingesting transcripts and providing intelligent query capabilities. It serves as an organizational memory that can think, reason, and discover insights.

```
[Meeting Platforms] → [Transcripts] → [Cognitive System] → [Insights]
                                              ↓
                                     [Organizational Memory]
                                              ↑
                                      [User Queries]
```

### 2.2 Primary Actors

| Actor | Description | Key Interactions |
|-------|-------------|------------------|
| **Meeting Participant** | Attendee of meetings | Queries for insights, receives briefs |
| **Project Manager** | Tracks commitments and decisions | Gap analysis, commitment tracking |
| **Knowledge Worker** | Seeks organizational knowledge | Cross-project insights, decision history |
| **System Administrator** | Maintains the system | Configuration, monitoring, maintenance |

### 2.3 Use Case Summary

| ID | Use Case | Description |
|----|----------|-------------|
| **UC1** | Pre-Meeting Intelligence | Prepare participants with relevant context |
| **UC2** | Deliverable Gap Analysis | Identify missing content for deliverables |
| **UC3** | Commitment Tracking | Monitor who committed to what and when |
| **UC4** | Decision Archaeology | Understand how and why decisions evolved |
| **UC5** | Cross-Project Intelligence | Discover relevant insights across projects |

---

## 3. Architecture Principles

### 3.1 Core Principles

#### P1: Memory-First Architecture
- Information is stored as interconnected memories, not documents
- Each memory has multi-dimensional properties
- Memories form an active network that can be traversed

#### P2: Cognitive Processing
- System mimics human memory: episodic → semantic → conceptual
- Activation spreading simulates associative thinking
- Bridge discovery finds non-obvious connections

#### P3: Local Intelligence
- All ML inference happens locally using ONNX
- No external API dependencies for core functionality
- Privacy-preserving by design

#### P4: Evolutionary Design
- System learns and improves over time
- Memories consolidate from raw to refined
- Patterns emerge through usage

#### P5: Simplicity in Complexity
- Complex cognitive processes, simple deployment
- Minimal operational overhead
- Zero-configuration where possible

### 3.2 Architectural Patterns

| Pattern | Application | Benefit |
|---------|-------------|---------|
| **Event Sourcing** | Meeting ingestion | Complete audit trail, replayability |
| **CQRS** | Separate write/read paths | Optimized for different access patterns |
| **Repository Pattern** | Data access abstraction | Flexibility in storage backends |
| **Pipeline Pattern** | Processing workflows | Clear separation of concerns |
| **Strategy Pattern** | Dimension extractors | Extensible analysis capabilities |

---

## 4. System Architecture

### 4.1 High-Level Architecture

The system follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                         │
│            (REST API, WebSocket, Admin CLI)                   │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                          │
│    (Use Case Handlers, Query Processors, Coordinators)       │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  Domain Logic Layer                           │
│  (Memory Extraction, Activation, Bridges, Consolidation)     │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                 Infrastructure Layer                          │
│    (ONNX Runtime, Embedding Cache, Feature Extractors)       │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Persistence Layer                          │
│        (Qdrant Vectors, SQLite, JSON Storage)                │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Component Architecture

#### 4.2.1 Core Components

| Component | Responsibility | Key Interfaces |
|-----------|---------------|----------------|
| **Memory Extractor** | Extract memories from transcripts | `extract_memories()`, `classify_type()` |
| **Dimension Analyzer** | Multi-dimensional analysis | `extract_temporal()`, `extract_causal()`, etc. |
| **Embedding Generator** | Generate vector representations | `encode()`, `batch_encode()` |
| **Activation Engine** | Spread activation through network | `spread_activation()`, `calculate_decay()` |
| **Bridge Discovery** | Find distant connections | `discover_bridges()`, `score_connections()` |
| **Query Processor** | Handle complex queries | `process_query()`, `aggregate_results()` |
| **Consolidation Engine** | Promote memories hierarchically | `consolidate()`, `detect_patterns()` |

#### 4.2.2 Storage Components

| Component | Purpose | Technology | Data Model |
|-----------|---------|------------|------------|
| **Vector Store** | Semantic search, activation | Qdrant | 384D vectors + metadata |
| **Relational Store** | Structure, relationships | SQLite | Normalized tables |
| **Document Store** | Transcripts, configs | JSON files | Hierarchical documents |
| **Cache Layer** | Performance optimization | In-memory | Key-value pairs |

### 4.3 Integration Architecture

The system integrates through well-defined interfaces:

1. **Meeting Platform Integration**
   - Input: Transcript + metadata
   - Protocol: REST API or file upload
   - Format: JSON with defined schema

2. **Client Integration**
   - Protocol: REST API + WebSocket
   - Authentication: JWT tokens
   - Format: JSON responses

3. **Monitoring Integration**
   - Metrics: Prometheus format
   - Logs: Structured JSON
   - Traces: OpenTelemetry

---

## 5. Data Architecture

### 5.1 Data Model Overview

The system uses a hybrid data model combining:
- **Vector data** for semantic similarity
- **Relational data** for structured queries
- **Document data** for raw content

### 5.2 Core Entities

#### 5.2.1 Meeting Entity
```
Meeting
├── id: UUID (Primary Key)
├── title: String
├── start_time: Timestamp
├── end_time: Timestamp
├── participants: Array[String]
├── series_id: UUID (Optional)
├── metadata: JSON
└── transcript_ref: String (File path)
```

#### 5.2.2 Memory Entity
```
Memory
├── id: UUID (Primary Key)
├── meeting_id: UUID (Foreign Key)
├── type: Enum[Decision, Action, Commitment, Question, Risk, Insight]
├── content: Text
├── speaker: String
├── timestamp: Float (seconds from start)
├── confidence: Float [0-1]
├── vector_id: UUID (Qdrant reference)
└── dimensions: JSON
    ├── temporal_score: Float
    ├── causal_score: Float
    ├── social_score: Float
    ├── emotional_score: Float
    └── evolutionary_score: Float
```

#### 5.2.3 Connection Entity
```
Connection
├── from_memory_id: UUID
├── to_memory_id: UUID
├── connection_type: Enum[temporal, causal, elaborates, contradicts]
├── strength: Float [0-1]
└── evidence: JSON
```

### 5.3 Vector Schema

#### 5.3.1 Hierarchical Collections

The system uses a 3-tier collection architecture in Qdrant:

**L0: Cognitive Concepts** (`cognitive_concepts`)
- Vector size: 400 dimensions
- Purpose: High-level abstract concepts and insights
- Metadata: concept_name, importance_score, child_count
- Retention: Permanent

**L1: Cognitive Contexts** (`cognitive_contexts`)
- Vector size: 400 dimensions
- Purpose: Contextual groupings and patterns
- Metadata: context_type, parent_concept, relevance_score
- Retention: Long-term with slow decay

**L2: Cognitive Episodes** (`cognitive_episodes`)
- Vector size: 400 dimensions
- Purpose: Raw episodic memories from meetings
- Metadata: memory_id, meeting_id, speaker, timestamp, type
- Retention: Subject to decay and consolidation

**Collection Configuration**:
```yaml
vector_params:
  size: 400
  distance: Cosine
optimizers_config:
  memmap_threshold: 50000
  indexing_threshold: 20000
hnsw_config:
  m: 16
  ef_construct: 200
  full_scan_threshold: 10000
```

### 5.4 SQLite Schema Implementation

The system uses a comprehensive SQLite schema for memory persistence and relationship management:

```sql
-- Memory metadata and relationships
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    level INTEGER NOT NULL,  -- 0=concept, 1=context, 2=episode
    content TEXT NOT NULL,
    dimensions_json TEXT NOT NULL,  -- Multi-dimensional data
    qdrant_id TEXT NOT NULL,  -- Reference to Qdrant vector
    timestamp DATETIME NOT NULL,
    last_accessed DATETIME NOT NULL,
    access_count INTEGER DEFAULT 0,
    importance_score REAL DEFAULT 0.0,
    parent_id TEXT,  -- Hierarchical relationship
    memory_type TEXT DEFAULT 'episodic',  -- episodic/semantic
    decay_rate REAL DEFAULT 0.1,
    FOREIGN KEY (parent_id) REFERENCES memories(id)
);

-- Connection graph for activation spreading
CREATE TABLE memory_connections (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    connection_strength REAL NOT NULL,
    connection_type TEXT DEFAULT 'associative',
    created_at DATETIME NOT NULL,
    last_activated DATETIME,
    activation_count INTEGER DEFAULT 0,
    PRIMARY KEY (source_id, target_id),
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
);

-- Bridge discovery cache
CREATE TABLE bridge_cache (
    query_hash TEXT NOT NULL,
    bridge_memory_id TEXT NOT NULL,
    bridge_score REAL NOT NULL,
    novelty_score REAL NOT NULL,
    connection_potential REAL NOT NULL,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (query_hash, bridge_memory_id),
    FOREIGN KEY (bridge_memory_id) REFERENCES memories(id)
);

-- Usage statistics for meta-learning
CREATE TABLE retrieval_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    retrieval_type TEXT NOT NULL,  -- core/peripheral/bridge
    success_score REAL,
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

-- Performance indexes
CREATE INDEX idx_memories_level ON memories(level);
CREATE INDEX idx_memories_timestamp ON memories(timestamp);
CREATE INDEX idx_memories_access_count ON memories(access_count);
CREATE INDEX idx_connections_strength ON memory_connections(connection_strength);
CREATE INDEX idx_bridge_cache_query ON bridge_cache(query_hash);
```

### 5.5 Data Flow Architecture

```
Ingestion Flow:
Transcript → Parser → Memory Extractor → Dimension Analyzer
    ↓                                           ↓
JSON Storage                              Vector Embeddings (400D)
    ↓                                           ↓
SQLite (Metadata) ←─── Relationships ──→ Qdrant (Vectors)

Query Flow:
Query → Embedding → Vector Search → Activation Spreading
   ↓                                        ↓
Intent Analysis                      Bridge Discovery
   ↓                                        ↓
Filters ────────→ Result Aggregation ←──────┘
                          ↓
                  Ranked Results
```

### 5.6 Memory Lifecycle Management

#### 5.6.1 Decay System

The system implements automatic memory lifecycle management through decay rates:

| Memory Type | Decay Rate | Half-Life | Auto-Cleanup |
|-------------|------------|-----------|--------------|
| **Episodic** | 0.1/day | ~7 days | When importance < 0.1 |
| **Semantic** | 0.01/day | ~70 days | When importance < 0.01 |
| **Conceptual** | 0.0 | Infinite | Manual only |

#### 5.6.2 Importance Score Evolution

```
importance(t) = importance(0) × e^(-decay_rate × t)

Where:
- importance(0) = initial importance (access_count × confidence)
- decay_rate = type-specific decay rate
- t = days since last access
```

#### 5.6.3 Cleanup Process

**Daily Maintenance Job**:
1. Calculate current importance for all memories
2. Mark episodic memories below threshold
3. Archive semantic memories below threshold
4. Update last_accessed timestamps
5. Trigger consolidation for qualified memories

### 5.7 Data Governance

| Aspect | Policy |
|--------|--------|
| **Retention** | Type-based: Episodic (auto-cleanup), Semantic (long-term), Conceptual (permanent) |
| **Privacy** | PII detection via regex patterns and optional masking |
| **Access Control** | Meeting-level and participant-level permissions via JWT claims |
| **Audit Trail** | All operations logged in retrieval_stats table |
| **Backup** | Daily snapshots: Qdrant collections, SQLite DB, JSON transcripts |
| **Compliance** | GDPR-ready with right to deletion support |

---

### 6.1 Cognitive Model

The system implements a three-tier cognitive model inspired by human memory:

```
Conceptual Level (Wisdom)
    ↑ Abstraction
Semantic Level (Knowledge)  
    ↑ Consolidation
Episodic Level (Experience)
```

### 6.2 Memory Extraction Pipeline

#### 6.2.1 Extraction Strategy

**Phase 1: Segmentation**
- Break transcript into semantic units
- Identify speaker transitions
- Detect topic boundaries

**Phase 2: Classification**
- Classify each segment by memory type
- Assign confidence scores
- Filter low-value content

**Phase 3: Enhancement**
- Extract named entities
- Identify temporal markers
- Detect causal relationships

### 6.2.2 ONNX Model Integration

**Model Loading and Optimization**:
- **Model**: all-MiniLM-L6-v2 ONNX format
- **Providers**: CPUExecutionProvider (optimized)
- **Performance**: 2-3x faster than PyTorch equivalent
- **Memory**: 100MB model size
- **Inference**: Direct ONNX Runtime for CPU-optimized execution

**Embedding Pipeline**:
1. Load ONNX model at startup
2. Tokenize input text (max 512 tokens)
3. Run ONNX inference
4. Apply mean pooling
5. Generate 384D base embedding
6. Concatenate with 16D dimensional features
7. Output 400D final embedding

### 6.3 Dimensional Analysis Architecture

Each memory undergoes multi-dimensional analysis producing a 16D feature vector that combines with the 384D semantic embedding for a final 400D representation:

#### 6.3.1 Temporal Dimension (4D)
- **Absolute**: Specific dates, times, deadlines
- **Relative**: "next week", "after the launch"
- **Positional**: Location within meeting timeline
- **Urgency**: Deadline detection via regex patterns
- **Implementation**: Regex patterns for temporal markers + positional encoding

#### 6.3.2 Causal Dimension (3D)
- **Explicit**: "because", "therefore", "as a result"
- **Implicit**: Logical sequences, dependencies
- **Probabilistic**: Likely cause-effect relationships
- **Implementation**: Rule-based pattern matching + dependency parsing

#### 6.3.3 Social Dimension (3D)
- **Speaker Authority**: Role, expertise level
- **Audience**: Who was present, who was addressed
- **Interaction Pattern**: Agreement, dissent, discussion
- **Implementation**: Interaction pattern detection algorithms

#### 6.3.4 Emotional Dimension (3D)
- **Sentiment**: Positive, negative, neutral scores
- **Intensity**: Strong opinions vs. mild preferences  
- **Confidence**: Certainty vs. speculation
- **Implementation**: VADER sentiment analysis + custom patterns

#### 6.3.5 Evolutionary Dimension (3D)
- **Maturity**: New idea vs. refined concept
- **Stability**: How much it has changed
- **Version**: Which iteration of an idea
- **Implementation**: Content comparison + version tracking

#### 6.3.6 Dimensional Fusion
- **Base Embedding**: 384D from all-MiniLM-L6-v2 (ONNX)
- **Dimensional Features**: 16D from above extractors
- **Final Vector**: 400D concatenated representation
- **Performance**: 2-3x faster than PyTorch equivalent

### 6.4 Activation Spreading Architecture

#### 6.4.1 Two-Phase Activation Model

The system uses a two-phase Breadth-First Search (BFS) approach:

**Phase 1: L0 Concept Activation**
- Query L0 cognitive concepts collection
- Identify top matching abstract concepts
- Set initial activation levels (1.0 for direct matches)

**Phase 2: BFS Through Connection Graph**
- Start from activated L0 concepts
- Traverse SQLite connection graph using BFS
- Apply activation decay at each hop
- Stop at configurable thresholds

```
Initial L0 Activation (1.0)
    ↓
Spread through connection graph (SQLite)
    ↓
Apply connection strength multiplier
    ↓
Apply decay factor (0.8 per hop)
    ↓
Continue until:
  - Activation < 0.7 (threshold)
  - Max 50 memories activated
  - Max depth reached
```

#### 6.4.2 Activation Parameters

| Parameter | Default Value | Purpose |
|-----------|--------------|---------|
| **Activation Threshold** | 0.7 | Minimum activation to continue spreading |
| **Max Activations** | 50 | Prevent runaway activation |
| **Decay Factor** | 0.8 | Per-hop activation decay |
| **Connection Threshold** | 0.5 | Minimum connection strength to traverse |

#### 6.4.3 Result Classification

Activated memories are classified into:
- **Core Memories**: Highest activation (>0.9)
- **Contextual Memories**: Medium activation (0.7-0.9)
- **Peripheral Memories**: Lower activation (<0.7)

### 6.5 Bridge Discovery Architecture

#### 6.5.1 Distance Inversion Algorithm

The system implements a sophisticated distance inversion approach for serendipitous discovery:

```
Bridge Score = (0.6 × Novelty Score) + (0.4 × Connection Potential)

Where:
- Novelty Score = 1.0 - cosine_similarity(query, candidate)
- Connection Potential = max(similarity to any activated memory)
- Final filtering: Bridge Score > threshold
```

**Algorithm Steps**:
1. Find memories with low direct similarity to query (<0.5)
2. Calculate connection potential through activated memories
3. Compute weighted bridge score
4. Cache results for performance
5. Return top 5 bridges

#### 6.5.2 Bridge Scoring Components

| Component | Weight | Calculation | Purpose |
|-----------|--------|-------------|---------|
| **Novelty** | 60% | 1.0 - similarity | Ensures distance from query |
| **Connection** | 40% | Max activation path | Ensures relevance through network |

#### 6.5.3 Bridge Types and Thresholds

| Type | Novelty Range | Connection Range | Example |
|------|--------------|------------------|---------|
| **Pattern Bridge** | 0.5-0.7 | >0.8 | "Our situation mirrors the Q3 crisis" |
| **Solution Bridge** | 0.7-0.9 | >0.6 | "The fix they used might work here" |
| **Evolution Bridge** | 0.4-0.6 | >0.7 | "This refines what we discussed in January" |
| **Contradiction Bridge** | >0.8 | >0.5 | "This contradicts our earlier decision" |

#### 6.5.4 Bridge Cache Strategy

- Cache key: Query hash
- Cache TTL: 1 hour
- Storage: SQLite bridge_cache table
- Benefit: 10x performance for repeated queries

### 6.6 Memory Consolidation Architecture

#### 6.6.1 Dual Memory System

The system implements episodic and semantic memory types with different characteristics:

**Episodic Memories**:
- **Decay Rate**: 0.1/day (fast decay)
- **Purpose**: Temporary experiences, raw meeting content
- **Retention**: Auto-cleanup after decay
- **Access Pattern**: High frequency initially, then drops

**Semantic Memories**:
- **Decay Rate**: 0.01/day (slow decay)
- **Purpose**: Consolidated patterns, learned knowledge
- **Retention**: Long-term storage
- **Access Pattern**: Steady access over time

#### 6.6.2 Consolidation Triggers

- **Frequency**: Same concept accessed 5+ times in 7 days
- **Pattern Stability**: Consistent pattern across 3+ meetings
- **Importance Score**: Combined access count + relevance scores
- **Manual Trigger**: Admin-initiated consolidation

#### 6.6.3 Consolidation Process

```
Episodic Memories (L2)
    ↓ (Access frequency > threshold)
Pattern Detection
    ↓ (Clustering similar memories)
Semantic Memory Candidate
    ↓ (Validation & scoring)
Promoted to L1 Context
    ↓ (Further abstraction)
Promoted to L0 Concept
```

#### 6.6.4 Decay and Cleanup

**Decay Formula**:
```
importance_score(t) = importance_score(0) × e^(-decay_rate × days)
```

**Cleanup Rules**:
- Episodic: Remove when importance < 0.1
- Semantic: Archive when importance < 0.01
- Concepts: Never auto-remove (manual only)

---

## 7. API Architecture

### 7.1 API Design Principles

1. **RESTful Design**: Resource-oriented endpoints
2. **Consistency**: Predictable request/response formats
3. **Versioning**: API version in URL path
4. **Error Handling**: Detailed error responses
5. **Async Support**: Long operations via callbacks

### 7.2 API Structure

#### 7.2.1 Resource Hierarchy

```
/api/v1/
├── /meetings/
│   ├── GET    /                    # List meetings
│   ├── POST   /                    # Create meeting
│   ├── GET    /{id}                # Get meeting
│   ├── POST   /{id}/ingest         # Ingest transcript
│   └── GET    /{id}/memories       # Get meeting memories
├── /memories/
│   ├── GET    /                    # Search memories
│   ├── GET    /{id}                # Get memory
│   ├── GET    /{id}/connections    # Get connections
│   └── GET    /{id}/activation-path # Get activation path
├── /query/
│   ├── POST   /                    # Natural language query
│   ├── POST   /meeting-prep        # UC1: Pre-meeting
│   ├── POST   /gap-analysis        # UC2: Gaps
│   ├── POST   /commitments         # UC3: Tracking
│   ├── POST   /decisions           # UC4: Archaeology
│   └── POST   /cross-project       # UC5: Cross-project
└── /admin/
    ├── GET    /health              # Health check
    ├── GET    /metrics             # Prometheus metrics
    └── POST   /consolidate         # Trigger consolidation
```

#### 7.2.2 Request/Response Patterns

**Standard Request Headers**
```
Authorization: Bearer {jwt_token}
Content-Type: application/json
X-Request-ID: {uuid}
X-Client-Version: {version}
```

**Standard Response Format**
```json
{
  "success": true,
  "data": {...},
  "metadata": {
    "request_id": "uuid",
    "timestamp": "iso8601",
    "version": "1.0.0",
    "processing_time_ms": 145
  },
  "errors": []
}
```

**Error Response Format**
```json
{
  "success": false,
  "data": null,
  "metadata": {...},
  "errors": [
    {
      "code": "INVALID_QUERY",
      "message": "Query cannot be empty",
      "field": "query",
      "details": {}
    }
  ]
}
```

### 7.3 Query API Specifications

#### 7.3.1 Natural Language Query

**Endpoint**: `POST /api/v1/query`

**Request**:
```json
{
  "query": "What did John commit to delivering this week?",
  "context": {
    "participant": "jane@company.com",
    "time_frame": "this_week",
    "projects": ["Project Alpha"]
  },
  "options": {
    "include_bridges": true,
    "max_results": 20,
    "min_confidence": 0.7
  }
}
```

**Response**:
```json
{
  "results": [
    {
      "memory_id": "mem_123",
      "content": "John: I'll have the API design ready by Friday",
      "relevance_score": 0.92,
      "activation_level": 0.85,
      "evidence_path": ["mem_100", "mem_110"],
      "metadata": {
        "speaker": "John",
        "meeting": "Weekly Standup",
        "timestamp": "2024-01-15T10:30:00Z"
      }
    }
  ],
  "bridges": [...],
  "summary": "Found 3 commitments from John this week"
}
```

#### 7.3.2 Use Case Specific APIs

**UC1: Pre-Meeting Intelligence**
- Input: meeting_id, participant
- Output: relevant_decisions, open_commitments, unresolved_questions
- Special: Temporal filtering for future relevance

**UC2: Gap Analysis**
- Input: deliverable_type, deadline
- Output: identified_gaps, missing_components, dependencies
- Special: Content-commitment matching

**UC3: Commitment Tracking**
- Input: assignee, time_range
- Output: commitments, completion_status, at_risk_items
- Special: Status inference from follow-ups

**UC4: Decision Archaeology**
- Input: decision_topic, time_range
- Output: decision_evolution, key_changes, rationale
- Special: Version tracking, supersession chains

**UC5: Cross-Project Intelligence**
- Input: project_context, query
- Output: relevant_patterns, similar_situations, lessons
- Special: Enhanced bridge discovery

### 7.4 WebSocket API

For real-time updates during processing:

```
ws://api/v1/stream

Messages:
→ {"type": "subscribe", "channel": "ingestion:{meeting_id}"}
← {"type": "progress", "stage": "extraction", "percent": 45}
← {"type": "memory", "data": {...memory...}}
← {"type": "complete", "summary": {...}}
```

---

## 8. Security Architecture

### 8.1 Security Principles

1. **Defense in Depth**: Multiple security layers
2. **Least Privilege**: Minimal access rights
3. **Zero Trust**: Verify everything
4. **Privacy by Design**: Built-in data protection
5. **Audit Everything**: Complete trail

### 8.2 Authentication Architecture

#### 8.2.1 Authentication Flow

```
Client → API Gateway → JWT Validation → User Context → Authorized Request
           ↓                                ↓
      Rate Limiting                  Audit Logging
```

#### 8.2.2 Token Structure

**JWT Claims**:
```json
{
  "sub": "user_id",
  "email": "user@company.com",
  "roles": ["analyst", "manager"],
  "projects": ["Alpha", "Beta"],
  "exp": 1234567890,
  "iat": 1234567890,
  "jti": "unique_token_id"
}
```

### 8.3 Authorization Architecture

#### 8.3.1 RBAC Model

| Role | Permissions |
|------|-------------|
| **Viewer** | Read own meetings, basic queries |
| **Analyst** | Cross-meeting queries, export data |
| **Manager** | View team data, commitment tracking |
| **Admin** | All operations, system management |

#### 8.3.2 Resource-Level Permissions

```
Permission Model:
User → Roles → Permissions → Resources
  ↓
Project Membership → Meeting Access
  ↓
Participant List → Memory Visibility
```

### 8.4 Data Protection

#### 8.4.1 Encryption

| Layer | Method | Purpose |
|-------|--------|---------|
| **Transit** | TLS 1.3 | API communication |
| **Rest** | AES-256 | Database encryption |
| **Application** | Field-level | Sensitive data |

#### 8.4.2 PII Handling

**Detection Patterns**:
- Email addresses
- Phone numbers
- SSN/ID numbers
- Credit card numbers
- Personal names (NER)

**Protection Actions**:
- Masking: Replace with tokens
- Hashing: One-way transformation
- Encryption: Reversible protection
- Exclusion: Don't store

### 8.5 Security Monitoring

#### 8.5.1 Security Events

| Event | Severity | Action |
|-------|----------|--------|
| Failed login | Medium | Log, rate limit |
| Privilege escalation | High | Alert, investigate |
| Mass data export | Medium | Log, notify |
| Config change | High | Audit, approval |

#### 8.5.2 Audit Trail

```
Audit Entry:
{
  "timestamp": "ISO8601",
  "user_id": "uuid",
  "action": "query.execute",
  "resource": "memories",
  "details": {
    "query": "masked_query",
    "result_count": 15
  },
  "ip_address": "1.2.3.4",
  "user_agent": "..."
}
```

---

## 9. Performance Architecture

### 9.1 Performance Requirements

| Operation | Target | Measurement Point |
|-----------|--------|-------------------|
| Memory Extraction | 10-15/second | Per meeting hour |
| Embedding Generation | <50ms | Single memory |
| ONNX Inference | 2-3x faster than PyTorch | Batch processing |
| Vector Search | <100ms | 100K memories |
| Activation Spreading | <500ms | 3-hop, 10K nodes, 50 max activations |
| Bridge Discovery | <1s | 5 bridges with caching |
| Full Query Pipeline | <2s | End-to-end |
| Activation Threshold | 0.7 | Minimum to continue |
| Max Activations | 50 | Per query limit |

### 9.2 Performance Optimization Strategies

#### 9.2.1 Embedding Optimization

**Batching Strategy**:
- Batch size: 32 for optimal throughput
- Preprocessing: Tokenization cache
- Model: Quantized ONNX (INT8)
- Caching: LRU with 10K entries

**Hardware Utilization**:
- CPU: 4 cores for ONNX runtime
- Memory: 2GB model cache
- Threads: 1 inter-op, 4 intra-op

#### 9.2.2 Vector Search Optimization

**Qdrant Configuration**:
```yaml
collection:
  vectors:
    size: 384
    distance: Cosine
  optimizers:
    indexing_threshold: 20000
    memmap_threshold: 50000
  hnsw_config:
    m: 16              # Connections per node
    ef_construct: 200  # Build-time accuracy
    ef: 100           # Search-time accuracy
```

**Search Strategy**:
- Pre-filter by metadata when possible
- Use payload indexes for common queries
- Implement result caching (5-minute TTL)

#### 9.2.3 Database Optimization

**SQLite Optimizations**:
- WAL mode for concurrent reads
- Page size: 4096 bytes
- Cache size: 10000 pages (40MB)
- Prepared statement caching
- Regular VACUUM and ANALYZE

**Query Optimization**:
- Covering indexes for common queries
- Denormalization for read performance
- Materialized views for aggregations

### 9.3 Caching Architecture

#### 9.3.1 Cache Hierarchy

```
L1: Application Memory (Embeddings, hot data)
    ↓ Miss
L2: Redis Cache (Query results, sessions)
    ↓ Miss
L3: Database (Persistent storage)
```

#### 9.3.2 Cache Strategies

| Cache Type | Strategy | TTL | Size |
|------------|----------|-----|------|
| Embeddings | LRU | 1 hour | 10K entries |
| Query Results | Time-based | 5 min | 1K entries |
| Activation State | Session | 30 min | Per query |
| User Context | Refresh | 15 min | All active |

### 9.4 Scalability Architecture

#### 9.4.1 Vertical Scaling

**Resource Limits**:
- CPU: Up to 16 cores effectively
- Memory: Up to 64GB beneficial
- Storage: Unlimited (partitioned)

**Scaling Points**:
- 100K memories: Single instance
- 1M memories: Optimized instance
- 10M memories: Distributed required

#### 9.4.2 Horizontal Scaling

**Stateless Components** (Easy to scale):
- API servers
- Embedding generators
- Query processors

**Stateful Components** (Careful scaling):
- Qdrant (sharding)
- SQLite (read replicas)
- Redis (clustering)

**Load Distribution**:
```
Load Balancer (Round Robin)
    ↓
API Instance Pool (2-10 instances)
    ↓
Shared Storage Layer
```

---

## 10. Deployment Architecture

### 10.1 Deployment Models

#### 10.1.1 Single Container Deployment

**Suitable for**:
- Small teams (<100 users)
- <1M memories
- Proof of concept

**Architecture**:
```
Docker Container
├── API Server (FastAPI)
├── ONNX Runtime
├── SQLite (embedded)
├── Qdrant (embedded mode)
└── File System (JSON)
```

#### 10.1.2 Distributed Deployment

**Suitable for**:
- Large organizations
- >1M memories
- High availability needs

**Architecture**:
```
Load Balancer
    ↓
API Servers (3+ instances)
    ↓
Service Mesh
├── Qdrant Cluster
├── SQLite + Litestream
├── Redis Cluster
└── Object Storage
```

### 10.2 Container Architecture

#### 10.2.1 Image Structure

```
Base: python:3.13-slim (150MB)
├── System Dependencies (50MB)
├── Python Dependencies (300MB)
├── ONNX Models (100MB)
├── Application Code (10MB)
└── Configuration (1MB)
Total: ~611MB
```

#### 10.2.2 Build Optimization

**Multi-stage Build**:
1. Builder stage: Compile dependencies
2. Model stage: Prepare ONNX models
3. Runtime stage: Minimal final image

**Layer Caching**:
- System deps (changes rarely)
- Python deps (changes occasionally)
- Models (changes rarely)
- Code (changes frequently)

### 10.3 Infrastructure Requirements

#### 10.3.1 Minimum Requirements

| Component | Development | Production |
|-----------|-------------|------------|
| CPU | 2 cores | 4+ cores |
| Memory | 4GB | 8GB+ |
| Storage | 10GB | 100GB+ |
| Network | 100Mbps | 1Gbps |

#### 10.3.2 Recommended Setup

**Small Deployment** (< 100 users):
- 1 server: 8 cores, 16GB RAM, 500GB SSD
- All components on single machine
- Daily backups to object storage

**Medium Deployment** (100-1000 users):
- 3 API servers: 4 cores, 8GB each
- 1 Qdrant server: 8 cores, 32GB RAM
- 1 Database server: 4 cores, 16GB RAM
- Load balancer + Redis cache

**Large Deployment** (1000+ users):
- Kubernetes cluster
- Auto-scaling API pods
- Qdrant cluster (3+ nodes)
- Distributed SQLite + Read replicas
- CDN for static assets

### 10.4 Deployment Process

#### 10.4.1 Deployment Pipeline

```
Code Push → CI Build → Tests → Build Image → Push Registry
                                    ↓
                            Deploy Staging → Integration Tests
                                    ↓
                            Deploy Production → Health Checks
```

#### 10.4.2 Zero-Downtime Deployment

1. **Blue-Green Deployment**:
   - Deploy new version to green environment
   - Run health checks
   - Switch load balancer
   - Keep blue as rollback

2. **Rolling Updates**:
   - Update one instance at a time
   - Health check before proceeding
   - Automatic rollback on failure

### 10.5 Configuration Management

#### 10.5.1 Configuration Hierarchy

```
Default Config (in code)
    ↓ Override
Environment Variables
    ↓ Override
Configuration File
    ↓ Override
Runtime Parameters
```

#### 10.5.2 Key Configuration Points

```yaml
# config.yaml
system:
  environment: production
  log_level: INFO
  
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  
storage:
  qdrant:
    host: localhost
    port: 6333
    collection: memories
  sqlite:
    path: /data/memories.db
    wal_mode: true
    
ml:
  model_path: /models/all-MiniLM-L6-v2.onnx
  batch_size: 32
  cache_size: 10000
  
security:
  jwt_secret: ${JWT_SECRET}
  token_expiry: 3600
```

---

## 11. Operational Architecture

### 11.1 Monitoring Strategy

#### 11.1.1 Metrics Architecture

**Application Metrics**:
- Request rate, latency, errors
- Memory extraction rate
- Query processing time
- Cache hit rates

**System Metrics**:
- CPU, memory, disk usage
- Network I/O
- Container health
- Database connections

**Business Metrics**:
- Active users
- Queries per day
- Memories processed
- Insights generated

#### 11.1.2 Logging Architecture

**Log Levels**:
```
DEBUG:   Detailed debugging information
INFO:    General operational information
WARNING: Warning conditions
ERROR:   Error conditions
CRITICAL: Critical failures
```

**Structured Logging Format**:
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "service": "query_processor",
  "trace_id": "abc123",
  "user_id": "user456",
  "message": "Query processed successfully",
  "metadata": {
    "query_type": "natural_language",
    "processing_time_ms": 234,
    "result_count": 15
  }
}
```

### 11.2 Operational Procedures

#### 11.2.1 Backup Strategy

**Backup Schedule**:
- Qdrant: Daily snapshots
- SQLite: Continuous (Litestream)
- JSON files: Daily sync to object storage

**Backup Retention**:
- Daily: 7 days
- Weekly: 4 weeks
- Monthly: 12 months

#### 11.2.2 Maintenance Windows

**Regular Maintenance**:
- Memory consolidation: Daily at 2 AM
- Database optimization: Weekly Sunday
- Model updates: Monthly
- Security patches: As needed

### 11.3 Incident Response

#### 11.3.1 Alert Categories

| Severity | Response Time | Examples |
|----------|--------------|----------|
| Critical | 15 minutes | Service down, data loss |
| High | 1 hour | Performance degradation |
| Medium | 4 hours | Non-critical errors |
| Low | Next business day | Minor issues |

#### 11.3.2 Runbooks

**Service Unavailable**:
1. Check health endpoint
2. Verify container status
3. Check resource usage
4. Review recent deployments
5. Rollback if necessary

**Performance Degradation**:
1. Check query latency metrics
2. Analyze slow queries
3. Verify cache performance
4. Check activation depth
5. Scale if necessary

### 11.4 Capacity Planning

#### 11.4.1 Growth Projections

```
Memory Growth Rate:
- Average meeting: 50 memories
- Meetings/day: Variable by org
- Growth: ~1GB/10K memories

Storage Requirements:
- Year 1: 100GB
- Year 2: 500GB  
- Year 3: 2TB
```

#### 11.4.2 Scaling Triggers

| Metric | Threshold | Action |
|--------|-----------|--------|
| CPU Usage | >80% sustained | Add CPU/instances |
| Memory Usage | >85% | Add memory |
| Query Latency | >2s p95 | Optimize/scale |
| Storage | >80% | Expand storage |

---

## 12. Evolution Strategy

### 12.1 Versioning Strategy

#### 12.1.1 API Versioning

- URL-based: `/api/v1/`, `/api/v2/`
- Backward compatibility: 2 versions
- Deprecation notice: 6 months
- Migration guides provided

#### 12.1.2 Data Schema Versioning

**Memory Schema Evolution**:
```
v1: Basic structure
v2: + Dimensional scores
v3: + Commitment tracking
v4: + Decision lineage
```

**Migration Strategy**:
- Backward compatible changes preferred
- Lazy migration where possible
- Batch migration tools provided

### 12.2 Feature Roadmap

#### 12.2.1 Phase 1 (Current)
- Core memory extraction
- Basic cognitive processing
- 5 primary use cases
- Single-language support

#### 12.2.2 Phase 2 (6 months)
- Advanced ML models
- Multi-language support
- Real-time processing
- Mobile applications

#### 12.2.3 Phase 3 (12 months)
- Federated deployments
- Advanced visualizations
- Predictive insights
- Plugin ecosystem

### 12.3 Technology Evolution

#### 12.3.1 Planned Upgrades

| Component | Current | Future | Timeline |
|-----------|---------|--------|----------|
| Python | 3.13 | 3.14+ | As released |
| ONNX | 1.15 | 2.0 | When stable |
| Embeddings | 384D | 768D option | Phase 2 |
| Database | SQLite | PostgreSQL option | Phase 3 |

#### 12.3.2 Research Areas

- Transformer-based memory models
- Graph neural networks for activation
- Causal inference improvements
- Automated insight generation

### 12.4 Migration Strategies

#### 12.4.1 Data Migration

**Principles**:
- No data loss
- Minimal downtime
- Rollback capability
- Incremental migration

**Process**:
1. Deploy new version with dual-write
2. Migrate historical data
3. Verify consistency
4. Switch to new format
5. Cleanup old data

#### 12.4.2 Model Migration

**Embedding Model Updates**:
- Maintain compatibility layer
- Re-encode in background
- A/B test new models
- Gradual rollout

## 13. Extensibility Architecture

### 13.1 Interface Design Principles

The system is built with extensibility in mind through abstract interfaces that enable component swapping and scaling:

### 13.2 Core Interfaces

#### 13.2.1 EmbeddingProvider Interface
```
Interface: EmbeddingProvider
- encode(text: str) → np.ndarray
- batch_encode(texts: List[str]) → np.ndarray
- get_dimension() → int

Implementations:
- ONNXEmbeddingProvider (default)
- PyTorchEmbeddingProvider
- CloudEmbeddingProvider (OpenAI, Cohere)
```

#### 13.2.2 VectorStorage Interface
```
Interface: VectorStorage
- store(id: str, vector: np.ndarray, metadata: dict)
- search(vector: np.ndarray, limit: int) → List[Result]
- delete(id: str)
- update_metadata(id: str, metadata: dict)

Implementations:
- QdrantVectorStorage (default)
- ChromaVectorStorage
- CustomVectorStorage
```

#### 13.2.3 ActivationEngine Interface
```
Interface: ActivationEngine
- spread_activation(seeds: List[str], params: dict) → ActivationResult
- get_activation_path(memory_id: str) → List[str]

Implementations:
- BFSActivationEngine (default)
- PageRankActivationEngine
- NeuralActivationEngine
```

#### 13.2.4 BridgeDiscovery Interface
```
Interface: BridgeDiscovery
- discover_bridges(query: np.ndarray, activated: Set[str]) → List[Bridge]
- score_bridge(candidate: Memory, context: Context) → float

Implementations:
- DistanceInversionBridge (default)
- SemanticBridge
- TemporalBridge
```

#### 13.2.5 MemoryLoader Interface
```
Interface: MemoryLoader
- load(source: str) → List[Memory]
- supported_formats() → List[str]

Implementations:
- MarkdownLoader
- PDFLoader
- CodeRepositoryLoader
- AudioTranscriptLoader
```

### 13.3 Plugin Architecture

The system supports plugins for custom functionality:

```
plugins/
├── extractors/
│   ├── custom_temporal.py
│   └── domain_specific.py
├── embedders/
│   └── specialized_model.py
└── analyzers/
    └── industry_patterns.py
```

### 13.4 Extension Points

| Extension Point | Purpose | Example Use Case |
|----------------|---------|------------------|
| **Dimension Extractors** | Add custom dimensions | Industry-specific patterns |
| **Memory Types** | New memory classifications | Code snippets, diagrams |
| **Bridge Algorithms** | Alternative discovery methods | Domain-specific connections |
| **Storage Backends** | Different databases | Cloud storage, graph DBs |
| **API Endpoints** | Custom query types | Specialized reports |

---

### A. Glossary

| Term | Definition |
|------|------------|
| **Activation Spreading** | BFS-based propagation of relevance through memory network with 0.7 threshold |
| **Bridge Discovery** | Finding non-obvious connections using distance inversion (60% novelty + 40% connection) |
| **Cognitive Embedding** | 400D vector (384D semantic + 16D dimensional features) |
| **Memory Consolidation** | Promoting episodic (0.1 decay) to semantic (0.01 decay) memories |
| **Distance Inversion** | Prioritizing low-similarity (<0.5), high-connection memories |
| **L0/L1/L2** | Concept/Context/Episode hierarchy in Qdrant collections |
| **VADER** | Sentiment analysis tool for emotional dimension extraction |
| **Two-Phase BFS** | L0 activation followed by connection graph traversal |

### B. References

1. ONNX Runtime Documentation
2. Qdrant Vector Database Guide
3. SQLite Performance Tuning
4. Sentence Transformers Documentation
5. Cognitive Computing Principles

### C. Decision Log

| Date | Decision | Rationale | Alternative |
|------|----------|-----------|-------------|
| 2024-01 | Qdrant over Pinecone | Self-hosted, hierarchical collections, 3-tier support | Cloud dependency |
| 2024-01 | ONNX over PyTorch | 920MB smaller images, 2-3x faster inference | Flexibility |
| 2024-01 | SQLite over PostgreSQL | Zero configuration, embedded, sufficient for connections | Advanced features |
| 2024-01 | 400D vectors | 384D semantic + 16D dimensions captures richness | Larger vectors |
| 2024-01 | BFS activation | Simple, effective, configurable thresholds | Neural propagation |
| 2024-01 | 60/40 bridge scoring | Balances novelty with connection strength | Other weightings |
| 2024-01 | VADER sentiment | Proven, fast, good accuracy for meetings | Deep learning models |

### 12.5 Implementation Phasing

#### Phase 1: Foundation (Weeks 1-4)

**Technical Deliverables:**
- SQLite database with full schema implementation
- ONNX all-MiniLM-L6-v2 integration
- Basic Qdrant 3-tier collections setup
- Rule-based dimension extraction:
  - Temporal: Regex patterns for deadlines and urgency
  - Emotional: VADER sentiment analysis
  - Contextual: Content type classification
  - Social: Interaction pattern detection
- Simple similarity-based retrieval
- 400D vector generation pipeline

**Success Criteria:**
- Process 100 memories with dimensional analysis
- <100ms embedding generation
- Basic retrieval working

#### Phase 2: Cognitive Features (Weeks 5-8)

**Technical Deliverables:**
- Two-phase BFS activation spreading
- SQLite connection graph implementation
- Distance inversion bridge discovery
- Bridge caching system
- Dual memory system (episodic/semantic)
- Memory consolidation pipeline

**Success Criteria:**
- Activation spreading <500ms for 10K memories
- Bridge discovery finding 3+ relevant connections
- Successful memory consolidation

#### Phase 3: Advanced Capabilities (Weeks 9-12)

**Technical Deliverables:**
- Meta-learning from retrieval statistics
- Advanced query processing for UC1-UC5
- Plugin architecture implementation
- Performance optimizations
- Production deployment configuration

**Success Criteria:**
- All use cases functional
- <2s end-to-end query processing
- 95% uptime in production

---