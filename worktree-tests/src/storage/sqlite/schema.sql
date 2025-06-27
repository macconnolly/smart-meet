-- Cognitive Meeting Intelligence Database Schema - Enhanced for Consulting Context
-- Reference: IMPLEMENTATION_GUIDE.md - Day 1: Core Models & Database
-- This schema defines all tables with consulting-specific enhancements

-- 1. Projects table - consulting project management
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    client_name TEXT NOT NULL,
    project_type TEXT CHECK(project_type IN ('strategy', 'transformation', 'diligence', 'operations', 'other')),
    status TEXT CHECK(status IN ('active', 'completed', 'on_hold', 'archived')),
    start_date DATETIME NOT NULL,
    end_date DATETIME,
    project_manager TEXT,
    engagement_code TEXT UNIQUE,
    budget_hours INTEGER,
    consumed_hours INTEGER DEFAULT 0,
    metadata_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. Meeting series table - for recurring meetings
CREATE TABLE IF NOT EXISTS meeting_series (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    series_name TEXT NOT NULL,
    frequency TEXT CHECK(frequency IN ('daily', 'weekly', 'biweekly', 'monthly')),
    day_of_week INTEGER CHECK(day_of_week BETWEEN 0 AND 6),  -- 0=Monday
    typical_duration_minutes INTEGER,
    typical_participants_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

-- 3. Meetings table - enhanced with project context and types
CREATE TABLE IF NOT EXISTS meetings (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    title TEXT NOT NULL,
    meeting_type TEXT CHECK(meeting_type IN (
        'client_workshop', 'client_update', 'client_steering',
        'internal_team', 'internal_leadership', 'internal_review',
        'expert_interview', 'stakeholder_interview', 
        'working_session', 'brainstorm'
    )),
    meeting_category TEXT CHECK(meeting_category IN ('internal', 'external')),
    is_recurring BOOLEAN DEFAULT FALSE,
    recurring_series_id TEXT,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    participants_json TEXT NOT NULL,  -- Enhanced with roles
    transcript_path TEXT NOT NULL,
    agenda_json TEXT,
    key_decisions_json TEXT,
    action_items_json TEXT,
    metadata_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    processed_at DATETIME,
    memory_count INTEGER DEFAULT 0,
    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (recurring_series_id) REFERENCES meeting_series(id)
);

-- 4. Memories table - enhanced for consulting context
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL,
    project_id TEXT NOT NULL,  -- Denormalized for performance
    content TEXT NOT NULL,
    speaker TEXT,
    speaker_role TEXT,  -- 'client', 'consultant', 'expert', 'stakeholder'
    timestamp REAL,
    
    -- Enhanced memory types for consulting
    memory_type TEXT DEFAULT 'episodic' CHECK(memory_type IN ('episodic', 'semantic')),
    content_type TEXT CHECK(content_type IN (
        'decision', 'action', 'commitment', 'question', 'insight',
        'deliverable', 'milestone', 'risk', 'issue', 'assumption',
        'hypothesis', 'finding', 'recommendation', 'dependency', 'context'
    )),
    
    -- Priority and status for consulting work
    priority TEXT CHECK(priority IN ('critical', 'high', 'medium', 'low')),
    status TEXT CHECK(status IN ('open', 'in_progress', 'completed', 'blocked', 'deferred')),
    due_date DATETIME,
    owner TEXT,
    
    -- Vector and cognitive fields
    level INTEGER NOT NULL CHECK(level IN (0,1,2)),
    qdrant_id TEXT NOT NULL UNIQUE,
    dimensions_json TEXT NOT NULL,
    
    -- Lifecycle
    importance_score REAL DEFAULT 0.5,
    decay_rate REAL DEFAULT 0.1,
    access_count INTEGER DEFAULT 0,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Hierarchy
    parent_id TEXT,
    
    -- Deliverable linkage
    deliverable_id TEXT,
    
    FOREIGN KEY (meeting_id) REFERENCES meetings(id),
    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (parent_id) REFERENCES memories(id),
    FOREIGN KEY (deliverable_id) REFERENCES deliverables(id)
);

-- 5. Memory connections table - enhanced connection types
CREATE TABLE IF NOT EXISTS memory_connections (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    connection_strength REAL NOT NULL DEFAULT 0.5,
    connection_type TEXT CHECK(connection_type IN (
        'sequential', 'reference', 'response', 'elaboration',
        'contradiction', 'supports', 'blocks', 'depends_on',
        'deliverable_link', 'hypothesis_evidence'
    )),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_activated DATETIME,
    activation_count INTEGER DEFAULT 0,
    PRIMARY KEY (source_id, target_id),
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
);

-- 6. Deliverables table
CREATE TABLE IF NOT EXISTS deliverables (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    deliverable_type TEXT CHECK(deliverable_type IN (
        'presentation', 'report', 'model', 'framework',
        'analysis', 'roadmap', 'business_case', 'other'
    )),
    status TEXT CHECK(status IN ('planned', 'in_progress', 'review', 'delivered', 'approved')),
    due_date DATETIME,
    owner TEXT NOT NULL,
    reviewer TEXT,
    version REAL DEFAULT 1.0,
    file_path TEXT,
    description TEXT,
    acceptance_criteria_json TEXT,
    dependencies_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    delivered_at DATETIME,
    approved_at DATETIME,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

-- 7. Stakeholders table
CREATE TABLE IF NOT EXISTS stakeholders (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    organization TEXT NOT NULL,
    role TEXT NOT NULL,
    stakeholder_type TEXT CHECK(stakeholder_type IN (
        'client_sponsor', 'client_team', 'client_stakeholder',
        'consultant_partner', 'consultant_manager', 'consultant_team',
        'external_expert', 'vendor', 'other'
    )),
    influence_level TEXT CHECK(influence_level IN ('high', 'medium', 'low')),
    engagement_level TEXT CHECK(engagement_level IN ('champion', 'supportive', 'neutral', 'skeptical', 'resistant')),
    email TEXT,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

-- 8. Search history table - tracks queries and results
CREATE TABLE IF NOT EXISTS search_history (
    id TEXT PRIMARY KEY,
    project_id TEXT,  -- Optional project filter
    query_text TEXT NOT NULL,
    query_vector_json TEXT,  -- Serialized query vector
    result_memory_ids TEXT,  -- JSON array of memory IDs
    result_scores TEXT,      -- JSON array of scores
    search_type TEXT CHECK(search_type IN ('vector', 'activation', 'bridge')) DEFAULT 'vector',
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

-- 9. System metadata table - stores system configuration and state
CREATE TABLE IF NOT EXISTS system_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
CREATE INDEX IF NOT EXISTS idx_projects_client ON projects(client_name);

CREATE INDEX IF NOT EXISTS idx_meetings_project ON meetings(project_id);
CREATE INDEX IF NOT EXISTS idx_meetings_type ON meetings(meeting_type);
CREATE INDEX IF NOT EXISTS idx_meetings_category ON meetings(meeting_category);

CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id);
CREATE INDEX IF NOT EXISTS idx_memories_meeting ON memories(meeting_id);
CREATE INDEX IF NOT EXISTS idx_memories_content_type ON memories(content_type);
CREATE INDEX IF NOT EXISTS idx_memories_priority ON memories(priority);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_due_date ON memories(due_date);
CREATE INDEX IF NOT EXISTS idx_memories_owner ON memories(owner);
CREATE INDEX IF NOT EXISTS idx_memories_level ON memories(level);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);

CREATE INDEX IF NOT EXISTS idx_connections_source ON memory_connections(source_id);
CREATE INDEX IF NOT EXISTS idx_connections_target ON memory_connections(target_id);

CREATE INDEX IF NOT EXISTS idx_deliverables_project ON deliverables(project_id);
CREATE INDEX IF NOT EXISTS idx_deliverables_status ON deliverables(status);
CREATE INDEX IF NOT EXISTS idx_deliverables_due_date ON deliverables(due_date);

CREATE INDEX IF NOT EXISTS idx_stakeholders_project ON stakeholders(project_id);
CREATE INDEX IF NOT EXISTS idx_stakeholders_type ON stakeholders(stakeholder_type);

CREATE INDEX IF NOT EXISTS idx_search_created_at ON search_history(created_at);

-- Initial system metadata (inserted by init_db.py)
-- INSERT INTO system_metadata (key, value) VALUES 
-- ('schema_version', '2.0'),
-- ('schema_type', 'consulting_enhanced'),
-- ('initialized_at', datetime('now'));