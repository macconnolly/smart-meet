-- Cognitive Meeting Intelligence Database Schema
-- Reference: IMPLEMENTATION_GUIDE.md - Day 1: Core Models & Database
-- This schema defines 5 tables as specified in the implementation guide

-- TODO Day 1: Execute this schema via scripts/init_db.py

-- 1. Meetings table - stores meeting metadata
CREATE TABLE IF NOT EXISTS meetings (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    participants_json TEXT,  -- JSON array of participant names
    transcript_path TEXT,
    metadata_json TEXT,      -- JSON object with additional metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    memory_count INTEGER DEFAULT 0
);

-- 2. Memories table - stores individual memory units
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL,
    content TEXT NOT NULL,
    speaker TEXT,
    timestamp_ms INTEGER DEFAULT 0,
    memory_type TEXT CHECK(memory_type IN ('decision', 'action', 'idea', 'issue', 'question', 'context')) DEFAULT 'context',
    content_type TEXT CHECK(content_type IN ('general', 'technical', 'strategic', 'operational')) DEFAULT 'general',
    level INTEGER CHECK(level IN (0, 1, 2)) DEFAULT 2,  -- L0=concepts, L1=semantic, L2=episodic
    qdrant_id TEXT,  -- Reference to vector in Qdrant
    dimensions_json TEXT,  -- 16D cognitive features as JSON
    importance_score REAL DEFAULT 0.5,
    decay_rate REAL DEFAULT 0.1,  -- L2=0.1, L1=0.01, L0=0.001
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    parent_id TEXT,  -- For consolidated memories (L1/L0 pointing to L2)
    FOREIGN KEY (meeting_id) REFERENCES meetings(id),
    FOREIGN KEY (parent_id) REFERENCES memories(id)
);

-- 3. Memory connections table - stores relationships between memories
CREATE TABLE IF NOT EXISTS memory_connections (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    connection_strength REAL DEFAULT 0.5 CHECK(connection_strength >= 0 AND connection_strength <= 1),
    connection_type TEXT CHECK(connection_type IN ('sequential', 'causal', 'semantic', 'reference', 'contradiction')) DEFAULT 'sequential',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activated TIMESTAMP,
    activation_count INTEGER DEFAULT 0,
    PRIMARY KEY (source_id, target_id),
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
);

-- 4. Search history table - tracks queries and results for learning
CREATE TABLE IF NOT EXISTS search_history (
    id TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_vector_json TEXT,  -- Serialized query vector
    result_memory_ids TEXT,  -- JSON array of memory IDs
    result_scores TEXT,      -- JSON array of scores
    search_type TEXT CHECK(search_type IN ('vector', 'activation', 'bridge')) DEFAULT 'vector',
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. System metadata table - stores system configuration and state
CREATE TABLE IF NOT EXISTS system_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
-- TODO Day 4: Verify these indexes are optimal for query patterns
CREATE INDEX IF NOT EXISTS idx_memories_meeting_id ON memories(meeting_id);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_level ON memories(level);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_connections_source ON memory_connections(source_id);
CREATE INDEX IF NOT EXISTS idx_connections_target ON memory_connections(target_id);
CREATE INDEX IF NOT EXISTS idx_search_created_at ON search_history(created_at);

-- Initial system metadata
-- TODO Day 1: Insert via init_db.py
-- INSERT INTO system_metadata (key, value) VALUES 
-- ('schema_version', '1.0'),
-- ('initialized_at', datetime('now'));
EOF < /dev/null
