# Consolidated Implementation Strategy for Cognitive Meeting Intelligence

## Overview
This document consolidates the updated implementation approach with enhanced strategy consulting features. The goal is to build a working pipeline in Week 1 with the correct architecture from the start.

## Core Principles

### 1. Build Right from Start
- 400D vectors from Day 1 (no migration needed)
- 3-tier Qdrant architecture ready
- Project-based organization
- Consulting-specific memory types

### 2. Agent-Friendly Design
Rather than providing complete code upfront, we provide:
- Clear specifications and interfaces
- Expected inputs/outputs
- Test cases for validation
- Performance targets
- Common patterns to follow

This allows agents to:
- Understand requirements clearly
- Implement with appropriate context
- Make technology choices
- Follow project conventions

### 3. Incremental Enhancement
- Week 1: Working pipeline with basic features
- Week 2: Add cognitive intelligence
- Week 3: Consulting-specific enhancements
- Week 4: Production hardening

## Updated Week 1 Plan (7-10 days)

### Day 1-2: Enhanced Data Foundation
**What**: SQLite schema with consulting features
**Specification**:
```python
# Core tables needed:
- projects: Client engagements
- meetings: Enhanced with types/categories  
- memories: Extended content types
- deliverables: Track outputs
- stakeholders: Manage relationships

# Key relationships:
- All meetings belong to a project
- All memories belong to meeting AND project
- Deliverables link to memories
- Stakeholders participate in meetings
```

**Agent Instructions**:
- Use enhanced_database_schema memory for full schema
- Implement proper foreign keys and indexes
- Add project_id to all relevant tables
- Create enums for consulting-specific types

### Day 3: Embeddings + Vector Composition
**What**: ONNX encoder with 400D vector manager
**Specification**:
```python
class VectorManager:
    def compose_vector(self, semantic: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
        """
        Input: 384D semantic + 16D dimensions
        Output: 400D normalized vector
        Performance: <150ms total
        """
        
class ONNXEncoder:
    def encode(self, text: str) -> np.ndarray:
        """
        Output: 384D normalized embeddings
        Performance: <100ms
        Cache: LRU with 10k capacity
        """
```

**Agent Instructions**:
- Download all-MiniLM-L6-v2 ONNX model
- Implement proper tokenization
- Ensure normalization of semantic portion
- Validate all dimensions

### Day 4: 3-Tier Qdrant Setup
**What**: All collections with proper HNSW tuning
**Specification**:
```yaml
L0_cognitive_concepts:
  vectors: 400D
  distance: Cosine
  hnsw_config:
    m: 32
    ef_construct: 400
    
L1_cognitive_contexts:
  vectors: 400D
  distance: Cosine
  hnsw_config:
    m: 24
    ef_construct: 300
    
L2_cognitive_episodes:
  vectors: 400D
  distance: Cosine
  hnsw_config:
    m: 16
    ef_construct: 200
```

**Agent Instructions**:
- Create all 3 tiers even if only using L2 initially
- Configure for future scale (50k+ memories)
- Add project_id to all payloads
- Test cross-tier search capability

### Day 5: Dimension Extractors
**What**: Implement real extractors for 7D, placeholders for 9D
**Specification**:
```python
# Implement now:
TemporalExtractor(4D):
  - urgency: Keywords + patterns
  - deadline: Date extraction
  - sequence: Position in meeting
  - duration: Time relevance

EmotionalExtractor(3D):  
  - sentiment: VADER analysis
  - confidence: Certainty markers
  - intensity: Emotional strength

# Placeholders (return 0.5):
SocialExtractor(3D)
CausalExtractor(3D)  
StrategicExtractor(3D)
```

**Agent Instructions**:
- Use VADER for emotional analysis
- Implement regex patterns for temporal
- Return normalized [0,1] values
- Total must equal exactly 16D

### Day 6-7: Pipeline + API
**What**: Complete ingestion with project context
**Specification**:
```python
class MeetingIngestionPipeline:
    async def ingest_meeting(
        self,
        project_id: str,
        meeting_type: str,
        transcript: str,
        metadata: dict
    ) -> dict:
        """
        1. Create meeting record with type
        2. Extract memories with consulting types
        3. Generate 400D vectors
        4. Store in L2 (episodic)
        5. Create connections
        6. Link to deliverables if mentioned
        
        Performance: <2s for 1 hour transcript
        """
```

**Agent Instructions**:
- Support all memory content types from schema
- Extract speaker roles (client/consultant)
- Identify deliverable mentions
- Create proper meeting categorization

### Day 8-9: Consulting Features
**What**: Add strategy-specific capabilities
**Specification**:
```python
# Memory extraction patterns:
- Deliverables: "slide deck", "model", "report"
- Hypotheses: "we believe", "hypothesis is"
- Risks: "risk", "concern", "blocker"
- Dependencies: "depends on", "blocked by"

# Priority detection:
- Critical: CEO, board, escalation
- High: steering committee, deadline
- Medium: team discussion
- Low: FYI, parking lot

# Stakeholder extraction:
- Identify names and roles
- Track sentiment per stakeholder
- Note influence indicators
```

**Agent Instructions**:
- Extend memory extractor with consulting patterns
- Add priority and owner detection
- Link memories to deliverables
- Track stakeholder mentions

### Day 10: Integration Testing
**What**: Validate complete consulting pipeline
**Test Scenarios**:
```python
# 1. Project setup
project = create_project("Digital Transformation", "Acme Corp")

# 2. Client meeting
ingest_meeting(
    project_id=project.id,
    meeting_type="client_workshop",
    transcript="CEO: We need 30% cost reduction..."
)

# 3. Internal meeting  
ingest_meeting(
    project_id=project.id,
    meeting_type="internal_team",
    transcript="Team: The client hypothesis needs testing..."
)

# 4. Verify extraction
- Correct meeting categorization
- Proper memory types (hypothesis, deliverable)
- Priority detection working
- 400D vectors stored
- Project context maintained
```

## What We Defer

### Week 2: Cognitive Intelligence
- Activation spreading
- Bridge discovery  
- Path tracking
- Memory classification (core/contextual/peripheral)

### Week 3: Advanced Consulting
- Stakeholder influence mapping
- Hypothesis-evidence linking
- Deliverable progress tracking
- Cross-project insights

### Week 4: Production
- Security (project access control)
- Performance optimization
- Monitoring and alerting
- Deployment automation

## Implementation Guidelines for Agents

### 1. Code Organization
```
src/
├── models/
│   ├── projects.py      # Project, Stakeholder
│   ├── meetings.py      # Meeting, MeetingSeries  
│   ├── memories.py      # Memory, Deliverable
│   └── entities.py      # All dataclasses
├── storage/
│   ├── sqlite/
│   │   ├── schema.sql   # Enhanced schema
│   │   └── repositories/
│   └── qdrant/
│       └── vector_store.py
├── extraction/
│   ├── memory_extractor.py  # With consulting patterns
│   └── dimensions/
│       ├── temporal.py      # Real implementation
│       ├── emotional.py     # VADER-based
│       └── placeholders.py  # Social, causal, strategic
└── pipeline/
    └── ingestion.py         # Project-aware pipeline
```

### 2. Testing Approach
Each component should have:
- Unit tests for logic
- Integration tests for database operations  
- Performance benchmarks
- Consulting-specific test cases

### 3. Performance Targets
- Memory extraction: 10-15/second
- Embedding generation: <100ms
- Vector composition: <50ms
- Total pipeline: <2s per meeting hour
- Search latency: <200ms

### 4. Error Handling
- Project not found → Clear error
- Invalid meeting type → List valid types
- Dimension mismatch → Log and use defaults
- Vector store timeout → Retry logic

## Success Criteria

### Week 1 Complete When:
1. ✅ Can create projects and link meetings
2. ✅ Extracts consulting-specific memory types
3. ✅ Generates proper 400D vectors
4. ✅ Stores in correct Qdrant tier
5. ✅ API handles project context
6. ✅ Performance targets met
7. ✅ All tests passing

### Ready for Week 2 When:
- Real data in system
- Multiple projects created
- Various meeting types ingested
- Search returns relevant results
- No migration needed for 400D
- Architecture supports cognitive features

## Key Decisions for Agents

1. **Memory Extraction**: Start simple with regex patterns, enhance iteratively
2. **Dimension Calculation**: Implement temporal/emotional now, placeholders for others
3. **Project Context**: Pass project_id through entire pipeline
4. **Deliverable Linking**: Extract mentions, create lightweight connections
5. **Testing Strategy**: Test each component in isolation, then integration

This approach builds the right foundation while leaving room for agents to make implementation decisions within clear constraints.