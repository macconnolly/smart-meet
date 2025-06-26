# Enhanced Database Schema for Strategy Consulting Context

## Overview
This schema extends the cognitive meeting intelligence system to support strategy consulting workflows with project hierarchies, deliverables tracking, and sophisticated meeting categorization.

## Core Schema Additions

### 1. Projects Table
```sql
CREATE TABLE projects (
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

CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_projects_client ON projects(client_name);
```

### 2. Enhanced Meetings Table
```sql
CREATE TABLE meetings (
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

CREATE INDEX idx_meetings_project ON meetings(project_id);
CREATE INDEX idx_meetings_type ON meetings(meeting_type);
CREATE INDEX idx_meetings_category ON meetings(meeting_category);
```

### 3. Meeting Series Table (for recurring meetings)
```sql
CREATE TABLE meeting_series (
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
```

### 4. Enhanced Memory Types
```sql
CREATE TABLE memories (
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
        'hypothesis', 'finding', 'recommendation', 'dependency'
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

CREATE INDEX idx_memories_project ON memories(project_id);
CREATE INDEX idx_memories_content_type ON memories(content_type);
CREATE INDEX idx_memories_priority ON memories(priority);
CREATE INDEX idx_memories_status ON memories(status);
CREATE INDEX idx_memories_due_date ON memories(due_date);
CREATE INDEX idx_memories_owner ON memories(owner);
```

### 5. Deliverables Table
```sql
CREATE TABLE deliverables (
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

CREATE INDEX idx_deliverables_project ON deliverables(project_id);
CREATE INDEX idx_deliverables_status ON deliverables(status);
CREATE INDEX idx_deliverables_due_date ON deliverables(due_date);
```

### 6. Stakeholders Table
```sql
CREATE TABLE stakeholders (
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

CREATE INDEX idx_stakeholders_project ON stakeholders(project_id);
CREATE INDEX idx_stakeholders_type ON stakeholders(stakeholder_type);
```

### 7. Enhanced Memory Connections
```sql
CREATE TABLE memory_connections (
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
```

## Dimension Enhancements for Consulting

### Updated 16D Feature Vector Structure
```
Temporal (4D):
- Urgency: Deadline proximity, client escalation
- Timeline: Project phase alignment
- Sequence: Meeting progression
- Duration: Time-bound commitments

Emotional (3D):
- Sentiment: Client satisfaction signals
- Confidence: Certainty in recommendations
- Stakeholder mood: Engagement level

Social (3D):
- Authority: Decision-maker involvement
- Influence: Stakeholder power dynamics
- Team dynamics: Collaboration patterns

Causal (3D):
- Dependencies: Task linkages
- Impact: Business outcome connections
- Risk factors: Mitigation relationships

Strategic (3D): [NEW - replacing evolutionary]
- Alignment: Strategy fit score
- Innovation: Novel approach indicator
- Value: Business impact potential
```

## Usage Examples

### 1. Creating a New Project
```python
project = Project(
    name="Digital Transformation Strategy",
    client_name="Acme Corp",
    project_type="transformation",
    status="active",
    project_manager="John Smith",
    engagement_code="ACME-2024-DT-001"
)
```

### 2. Categorizing Meeting Types
```python
meeting = Meeting(
    project_id=project.id,
    title="SteerCo Update #3",
    meeting_type="client_steering",
    meeting_category="external",
    is_recurring=True,
    recurring_series_id="steerco-weekly"
)
```

### 3. Enhanced Memory Types
```python
# Deliverable-linked memory
memory = Memory(
    content="We need to complete the market sizing by Friday",
    content_type="deliverable",
    priority="high",
    due_date=datetime(2024, 3, 15),
    owner="Sarah Chen",
    deliverable_id="market-analysis-001"
)

# Hypothesis tracking
memory = Memory(
    content="Our hypothesis is that 30% cost reduction is achievable through automation",
    content_type="hypothesis",
    importance_score=0.9
)

# Risk identification
memory = Memory(
    content="Client stakeholder alignment is a critical risk to implementation",
    content_type="risk",
    priority="critical",
    status="open"
)
```

## Benefits for Strategy Consulting

1. **Project Context**: All memories tied to specific client engagements
2. **Deliverable Tracking**: Direct linkage between discussions and outputs
3. **Stakeholder Management**: Track influence and engagement patterns
4. **Meeting Intelligence**: Distinguish internal vs client discussions
5. **Priority Management**: Focus on critical/high priority items
6. **Timeline Awareness**: Due dates and milestone tracking
7. **Hypothesis Testing**: Track assumptions and evidence
8. **Risk Management**: Identify and track project risks
9. **Dependency Mapping**: Understand task relationships
10. **Recurring Patterns**: Learn from series meetings

## Migration Notes

To add these enhancements to existing schema:
1. Add new tables first (projects, deliverables, etc.)
2. Alter memories table to add new columns
3. Run data migration to set default project_id
4. Update application code to use new fields
5. Backfill stakeholder and meeting type data

This enhanced schema transforms the system from general meeting intelligence to a consulting-specific knowledge management platform.