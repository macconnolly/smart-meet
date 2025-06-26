# Vector and Graph Storage Architecture

> **Navigation**: [Home](../../README.md) → [Architecture](system-overview.md) → Vector & Graph Storage  
> **Related**: [System Overview](system-overview.md) | [Data Flow](data-flow.md)

## Overview

The Cognitive Meeting Intelligence system uses a **hybrid storage approach** that combines:
1. **Qdrant** for vector storage and similarity search
2. **SQLite** for metadata, relationships, and graph structure
3. **No separate graph database** - graph relationships stored in SQLite

This approach balances performance, simplicity, and functionality.

## Vector Storage (Qdrant)

### 3-Tier Architecture

We store 400-dimensional vectors (384D semantic + 16D cognitive features) in three collections:

#### L0: Cognitive Concepts
- **Purpose**: Highest-level abstractions and concepts
- **Size**: ~1K-10K vectors
- **HNSW Config**: m=32, ef_construct=400 (optimized for quality)
- **Usage**: Initial phase of activation spreading

#### L1: Cognitive Contexts  
- **Purpose**: Consolidated semantic memories and patterns
- **Size**: ~10K-50K vectors
- **HNSW Config**: m=24, ef_construct=300 (balanced)
- **Usage**: Intermediate activation, bridge discovery

#### L2: Cognitive Episodes
- **Purpose**: Raw episodic memories from meetings
- **Size**: ~50K-500K vectors
- **HNSW Config**: m=16, ef_construct=200 (optimized for scale)
- **Usage**: Primary storage, direct search

### Vector Metadata

Each vector point stores:
```json
{
  "memory_id": "uuid",
  "project_id": "project-001",
  "meeting_id": "meeting-123",
  "level": 2,
  "memory_type": "episodic",
  "content_type": "decision",
  "importance_score": 0.8,
  "timestamp": 1701234567
}
```

## Graph Storage (SQLite)

### Why Not a Separate Graph Database?

1. **Simplicity**: One less system to manage
2. **ACID Compliance**: Transactional consistency
3. **Sufficient Performance**: With proper indexing, handles our graph needs
4. **Integration**: Easy joins with metadata

### Memory Connections Table

The `memory_connections` table stores the graph structure:

```sql
CREATE TABLE memory_connections (
    source_id TEXT,
    target_id TEXT,
    connection_strength REAL (0.0-1.0),
    connection_type TEXT,  -- 10 types including deliverable_link
    created_at DATETIME,
    last_activated DATETIME,
    activation_count INTEGER,
    PRIMARY KEY (source_id, target_id)
);
```

### Graph Indexes for Performance

```sql
-- For forward traversal (activation spreading)
CREATE INDEX idx_connections_source ON memory_connections(source_id);

-- For backward traversal (finding references)
CREATE INDEX idx_connections_target ON memory_connections(target_id);

-- For filtering by connection type
CREATE INDEX idx_connections_type ON memory_connections(connection_type);

-- For finding strong connections
CREATE INDEX idx_connections_strength ON memory_connections(connection_strength);
```

### Connection Types

1. **Sequential**: Next memory in meeting timeline
2. **Reference**: One memory references another
3. **Response**: Direct response to previous memory
4. **Elaboration**: Expands on previous point
5. **Contradiction**: Conflicts with another memory
6. **Supports**: Evidence supporting claim
7. **Blocks**: Dependency blocking
8. **Depends_on**: Task dependencies
9. **Deliverable_link**: Related to same deliverable
10. **Hypothesis_evidence**: Evidence for hypothesis

## Hybrid Query Patterns

### 1. Activation Spreading
```python
# Phase 1: Vector search in Qdrant L0
initial_memories = qdrant.search(
    collection="cognitive_concepts",
    query_vector=query_embedding,
    filter={"project_id": project_id},
    limit=10
)

# Phase 2: Graph traversal in SQLite
for memory in initial_memories:
    connections = db.execute("""
        SELECT target_id, connection_strength, connection_type
        FROM memory_connections
        WHERE source_id = ? 
        AND connection_strength > ?
        ORDER BY connection_strength DESC
    """, [memory.id, threshold])
```

### 2. Bridge Discovery
```python
# Find memories distant from query but connected to activated set
activated_ids = [m.id for m in activated_memories]

# Get all connections from activated memories
bridge_candidates = db.execute("""
    SELECT DISTINCT mc.target_id, AVG(mc.connection_strength) as avg_strength
    FROM memory_connections mc
    WHERE mc.source_id IN ({})
    AND mc.target_id NOT IN ({})
    GROUP BY mc.target_id
    HAVING avg_strength > ?
""".format(placeholders), activated_ids + activated_ids + [threshold])

# Check vector distance for novelty
for candidate in bridge_candidates:
    vector = qdrant.retrieve(candidate.target_id)
    distance = cosine_distance(query_vector, vector)
    if distance > novelty_threshold:  # Far from query
        bridges.append(candidate)
```

### 3. Deliverable Network
```python
# Find all memories connected to a deliverable
deliverable_memories = db.execute("""
    SELECT m.*, mc.connection_strength
    FROM memories m
    JOIN memory_connections mc ON m.id = mc.target_id
    WHERE mc.source_id IN (
        SELECT id FROM memories WHERE deliverable_id = ?
    )
    AND mc.connection_type = 'deliverable_link'
    ORDER BY mc.connection_strength DESC
""", [deliverable_id])
```

## Performance Characteristics

### Vector Search (Qdrant)
- **Similarity search**: O(log n) with HNSW
- **Filtered search**: Efficient with payload indexes
- **Batch retrieval**: Optimized for multiple IDs

### Graph Traversal (SQLite)
- **Single hop**: O(1) with index
- **Multi-hop BFS**: O(b^d) where b=branching, d=depth
- **Filtered traversal**: Efficient with composite indexes

### Hybrid Operations
- **Activation spreading**: <500ms for 50 activations
- **Bridge discovery**: <1s for 100 candidates
- **Deliverable network**: <200ms for typical deliverable

## Data Consistency

### Write Operations
1. Insert memory metadata in SQLite (transactional)
2. Store vector in Qdrant (eventually consistent)
3. Create connections in SQLite (transactional)

### Failure Handling
- If Qdrant write fails: Retry with exponential backoff
- If connection creation fails: Mark memory for reprocessing
- Background job: Reconcile SQLite and Qdrant

## Advantages of Hybrid Approach

1. **Best of Both Worlds**: Vector similarity + graph relationships
2. **Simplified Operations**: No graph database to manage
3. **Cost Effective**: SQLite is free and efficient
4. **Flexible Queries**: SQL joins with vector results
5. **ACID Guarantees**: For critical metadata

## Future Optimizations

1. **Graph Caching**: Memory-mapped connection cache
2. **Materialized Paths**: Pre-compute common traversals
3. **Partitioning**: Shard by project for scale
4. **Read Replicas**: Separate read/write connections
5. **Graph Embeddings**: Encode structure in vectors

## Example: Complete Cognitive Query

```python
async def cognitive_query(query_text: str, project_id: str):
    # 1. Generate query embedding
    query_vector = await encoder.encode(query_text)
    
    # 2. Initial vector search (L0 concepts)
    concepts = await qdrant.search(
        collection="cognitive_concepts",
        query_vector=query_vector,
        filter={"project_id": project_id},
        limit=5
    )
    
    # 3. Activation spreading through graph
    activated = await spread_activation(
        initial_memories=concepts,
        max_hops=3,
        decay_factor=0.8
    )
    
    # 4. Bridge discovery
    bridges = await discover_bridges(
        query_vector=query_vector,
        activated_memories=activated,
        novelty_threshold=0.3
    )
    
    # 5. Enrich with metadata
    return await enrich_results(activated + bridges)
```

This hybrid approach provides the semantic understanding of vector search with the relationship intelligence of graph traversal, all while maintaining operational simplicity.
