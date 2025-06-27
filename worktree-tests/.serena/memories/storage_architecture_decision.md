# Storage Architecture Decision

## Vector Storage: Qdrant
- 3-tier collections (L0, L1, L2) with different HNSW parameters
- 400D vectors (384D semantic + 16D features)
- Optimized for different access patterns and scale

## Graph Storage: SQLite
- **Decision**: Use SQLite's `memory_connections` table instead of a separate graph database
- **Rationale**:
  1. Simpler architecture (one less system)
  2. ACID compliance for relationships
  3. Sufficient performance with proper indexing
  4. Easy integration with metadata
  5. SQL joins with memory metadata

## Hybrid Approach Benefits
- Vector similarity search in Qdrant
- Graph traversal in SQLite with indexed connections
- No need for Neo4j or other graph database
- Activation spreading uses both systems
- Bridge discovery combines vector distance and graph connections

## Key Tables and Indexes
- `memory_connections`: Stores edges with strength and type
- Indexes on source_id, target_id, connection_type, strength
- 10 connection types including consulting-specific ones
- Supports efficient BFS for activation spreading

This hybrid approach balances performance, simplicity, and functionality for our cognitive intelligence needs.