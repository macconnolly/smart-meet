# Phase 2: Activation Spreading & Consulting Intelligence (Week 2) - Hyper-Detailed

## Overview
Add cognitive intelligence to the working pipeline with activation spreading while incorporating strategy consulting features.

## Day 1-2: Enhanced Models for Consulting Context

### Task 1.1: Create Project-Aware Activation Models
**Location**: `src/models/activation.py`
**Specification**:
```python
@dataclass
class ActivatedMemory:
    """Memory activated through spreading with consulting context"""
    memory_id: str
    content: str
    project_id: str  # NEW: Which project this belongs to
    meeting_type: str  # NEW: client_workshop, internal_team, etc.
    activation_strength: float  # 0.0 to 1.0
    depth: int  # Hops from initial activation
    path: List[str]  # Memory IDs showing activation path
    classification: str  # 'core', 'contextual', 'peripheral'
    
    # Consulting-specific fields
    content_type: str  # decision, deliverable, risk, hypothesis, etc.
    stakeholder_mentions: List[str]  # Names found in content
    deliverable_links: List[str]  # IDs of linked deliverables
    priority: Optional[str]  # critical, high, medium, low
    
@dataclass
class ActivationContext:
    """Context for activation spreading"""
    project_id: str  # Limit activation to project
    include_cross_project: bool = False  # Allow insights from other projects
    meeting_types_filter: Optional[List[str]] = None  # Only specific meeting types
    stakeholder_filter: Optional[List[str]] = None  # Focus on specific people
    time_range: Optional[Tuple[datetime, datetime]] = None
```

**Test Cases**:
```python
def test_activation_with_project_context():
    context = ActivationContext(
        project_id="acme-digital-001",
        meeting_types_filter=["client_workshop", "client_steering"]
    )
    # Verify only activates within project and meeting types
```

**Agent Instructions**:
- Extend existing models to include project context
- Add stakeholder and deliverable tracking
- Ensure backward compatibility with base activation

### Task 1.2: Create Stakeholder-Aware Connection Model
**Location**: `src/models/connections.py`
**Specification**:
```python
@dataclass  
class EnhancedMemoryConnection:
    """Connection with consulting context"""
    source_id: str
    target_id: str
    connection_strength: float
    connection_type: str  # Now includes: deliverable_link, hypothesis_evidence, stakeholder_mention
    
    # Consulting enhancements
    shared_stakeholders: List[str]  # Both memories mention these people
    shared_deliverables: List[str]  # Both relate to these deliverables
    meeting_proximity: float  # How close in meeting timeline
    cross_meeting_link: bool  # Connects across meetings
```

**Validation**:
- Connection strength considers stakeholder overlap
- Deliverable links boost connection strength
- Cross-meeting links tracked for insights

### Task 1.3: Design Consulting-Aware Algorithm
**Location**: `docs/activation_algorithm_v2.md`
**Document Contents**:
```markdown
# Two-Phase Activation with Consulting Context

## Phase 1: Project-Scoped L0 Search
1. Query L0 concepts filtered by project_id
2. If stakeholder_filter, boost memories mentioning them
3. Apply meeting_type weights:
   - client_steering: 1.2x boost
   - internal_review: 0.8x weight
   
## Phase 2: Context-Aware BFS
1. Spread through connections considering:
   - Deliverable links (2x connection strength)
   - Stakeholder mentions (1.5x if same person)
   - Meeting proximity (decay by days between)
   
## Special Rules:
- Critical priority memories get 1.5x activation boost
- Hypothesis-evidence pairs activate together
- Risk memories activate mitigation memories
```

### Task 1.4: Update Repository for Graph Operations
**Location**: `src/storage/sqlite/repositories/connection_repository.py`
**New Methods Specification**:
```python
async def get_project_connections(
    self, memory_ids: List[str], project_id: str
) -> Dict[str, List[Connection]]:
    """Get connections within project scope"""
    # SQL should join with memories table to filter by project_id
    
async def get_stakeholder_connections(
    self, stakeholder_name: str, project_id: str
) -> List[Connection]:
    """Find all connections between memories mentioning stakeholder"""
    # Use JSON extraction for stakeholder_mentions
    
async def get_deliverable_network(
    self, deliverable_id: str
) -> Dict[str, List[Connection]]:
    """Get network of memories connected to deliverable"""
    # Trace through deliverable_link connections
```

**Performance Requirements**:
- Batch operations must handle 100+ memory IDs
- Use indexes on project_id and connection_type
- Cache frequently accessed connection patterns

## Day 3-4: Consulting-Enhanced Activation Engine

### Task 3.1: Create Project-Aware Activation Engine
**Location**: `src/cognitive/activation/consulting_activation_engine.py`
**Specification**:
```python
class ConsultingActivationEngine(ActivationEngine):
    """Activation engine with consulting intelligence"""
    
    def __init__(self, 
                 memory_repo: MemoryRepository,
                 connection_repo: ConnectionRepository,
                 vector_store: QdrantVectorStore,
                 stakeholder_repo: StakeholderRepository,  # NEW
                 deliverable_repo: DeliverableRepository,  # NEW
                 decay_factor: float = 0.8):
        super().__init__(memory_repo, connection_repo, vector_store, decay_factor)
        self.stakeholder_repo = stakeholder_repo
        self.deliverable_repo = deliverable_repo
        
    async def spread_activation(self, 
                               query_vector: np.ndarray,
                               context: ActivationContext) -> ActivationResult:
        """
        Spread activation with consulting context
        
        Performance targets:
        - <500ms for 50 activations within project
        - <800ms with stakeholder filtering
        - <1s with cross-project insights
        """
```

**Key Differences from Base Engine**:
- Project scoping at every step
- Stakeholder influence calculation
- Deliverable network traversal
- Meeting type weighting

### Task 3.2: Implement Stakeholder Influence
**Location**: `src/cognitive/activation/stakeholder_influence.py`
**Specification**:
```python
class StakeholderInfluenceCalculator:
    """Calculate how stakeholders affect activation"""
    
    async def calculate_influence(
        self, memory: Memory, stakeholders: List[Stakeholder]
    ) -> float:
        """
        Calculate influence multiplier based on:
        - Stakeholder authority level (CEO > Manager)
        - Stakeholder engagement (Champion > Neutral > Resistant)
        - Decision-making power in context
        
        Returns: multiplier between 0.5 and 2.0
        """
        
    async def boost_by_stakeholder_network(
        self, activation_queue: PriorityQueue,
        stakeholder_filter: List[str]
    ):
        """Boost memories in queue if they mention key stakeholders"""
```

**Test Scenarios**:
```python
def test_ceo_mention_boost():
    # Memory mentioning CEO should get 2x boost
    
def test_resistant_stakeholder_damping():
    # Memory from resistant stakeholder gets 0.7x factor
```

### Task 3.3: Implement Deliverable Network Activation
**Location**: Within activation engine
**Algorithm Enhancement**:
```python
# Pseudo-code for deliverable network activation
if memory.deliverable_id:
    # Find all memories linked to same deliverable
    related_memories = await get_deliverable_memories(memory.deliverable_id)
    
    # Activate related memories with boosted strength
    for related in related_memories:
        if related.id not in visited:
            # Deliverable link = stronger activation
            strength = current_strength * 0.9  # Only 10% decay
            activation_queue.put((-strength, depth, related.id, path))
            
    # Also activate dependent deliverables
    dependencies = await get_deliverable_dependencies(memory.deliverable_id)
    for dep in dependencies:
        dep_memories = await get_deliverable_memories(dep.id)
        # Activate with dependency strength
```

### Task 3.4: Project-Scoped Path Tracking
**Enhancement Specification**:
```python
# Path now includes richer information
path_entry = {
    "memory_id": memory_id,
    "project_id": project_id,
    "meeting_id": meeting_id,
    "meeting_type": meeting_type,
    "connection_type": connection_type,  # How we got here
    "strength_contribution": strength_delta
}

# Track cross-meeting and cross-project jumps
if current_project != previous_project:
    path_entry["cross_project_insight"] = True
    path_entry["project_transition"] = f"{previous_project} → {current_project}"
```

## Day 5: Advanced Classification & Filtering

### Task 5.1: Implement Consulting-Specific Classification
**Location**: `src/cognitive/activation/classification.py`
**Enhanced Classification**:
```python
class ConsultingMemoryClassifier:
    """Classify activated memories for consulting context"""
    
    def classify_activation(
        self, 
        memory: ActivatedMemory,
        context: ActivationContext
    ) -> str:
        """
        Enhanced classification considering:
        - Base activation strength
        - Content type (decisions > general discussion)
        - Priority level (critical items always core)
        - Stakeholder importance
        - Deliverable relevance
        """
        
        # Priority override
        if memory.priority == "critical":
            return "core"
            
        # Deliverable-linked memories get boost
        if memory.deliverable_links:
            threshold_adjustment = -0.1  # Lower threshold
        
        # Client meeting content gets preference
        if memory.meeting_type.startswith("client_"):
            threshold_adjustment = -0.05
            
        # Standard thresholds with adjustments
        if memory.activation_strength >= (0.7 + threshold_adjustment):
            return "core"
        elif memory.activation_strength >= (0.4 + threshold_adjustment):
            return "contextual"
        else:
            return "peripheral"
```

### Task 5.2: Implement Result Filtering
**Location**: `src/cognitive/activation/result_filter.py`
**Specification**:
```python
class ActivationResultFilter:
    """Filter and organize activation results"""
    
    async def apply_consulting_filters(
        self,
        results: List[ActivatedMemory],
        context: ActivationContext
    ) -> ActivationResult:
        """
        Apply filters:
        1. Remove duplicate insights (same content, different meetings)
        2. Ensure deliverable representation
        3. Balance meeting types
        4. Prioritize by stakeholder relevance
        5. Apply time decay for old memories
        """
        
    def ensure_diversity(
        self, 
        memories: List[ActivatedMemory]
    ) -> List[ActivatedMemory]:
        """
        Ensure diverse results:
        - Max 30% from same meeting
        - At least 2 meeting types if available
        - Mix of content types (not all decisions)
        - Include at least 1 risk/issue if present
        """
```

### Task 5.3: Performance Optimization for Consulting
**Location**: Update existing engine
**Optimizations**:
```python
# 1. Project-scoped indexes
CREATE INDEX idx_memories_project_content ON memories(project_id, content_type);
CREATE INDEX idx_connections_project ON memory_connections(source_id, connection_type) 
    WHERE connection_type IN ('deliverable_link', 'hypothesis_evidence');

# 2. Stakeholder mention cache
class StakeholderMentionCache:
    """Cache which memories mention which stakeholders"""
    def __init__(self, ttl_seconds=3600):
        self.cache = {}  # stakeholder_name -> Set[memory_id]
        
# 3. Deliverable network cache
class DeliverableNetworkCache:
    """Cache deliverable->memories mappings"""
    # Invalidate when new memory added to deliverable
```

## Day 6-7: Testing & Integration

### Task 6.1: Consulting-Specific Test Cases
**Location**: `tests/unit/test_consulting_activation.py`
**Test Scenarios**:
```python
async def test_client_meeting_priority():
    """Client meetings should activate more strongly"""
    # Create memories from client and internal meetings
    # Verify client meeting memories have higher activation
    
async def test_deliverable_network_activation():
    """Deliverable-linked memories should activate together"""
    # Create presentation deliverable
    # Create 5 memories linked to it
    # Activate one memory
    # Verify all 5 get activated through deliverable link
    
async def test_stakeholder_influence_path():
    """Track stakeholder influence through activation"""
    # Create CEO memory with decision
    # Create team memories referencing decision
    # Verify activation path shows CEO->team spread
    
async def test_cross_project_insights():
    """When enabled, should find relevant patterns from other projects"""
    # Create similar issue in different project
    # Enable cross_project in context
    # Verify finds and marks as cross-project insight
```

### Task 6.2: Integration with Search API
**Location**: `src/api/routers/search.py`
**Update Specification**:
```python
@router.post("/api/v2/search/consulting")
async def consulting_cognitive_search(request: ConsultingSearchRequest):
    """
    Consulting-aware search with project context
    
    Request includes:
    - project_id (required)
    - stakeholder_filter (optional)
    - meeting_type_filter (optional)  
    - include_cross_project_insights (default: False)
    - prioritize_deliverables (default: True)
    """
    
    # Build activation context
    context = ActivationContext(
        project_id=request.project_id,
        include_cross_project=request.include_cross_project_insights,
        meeting_types_filter=request.meeting_type_filter,
        stakeholder_filter=request.stakeholder_filter
    )
    
    # Use consulting activation engine
    result = await consulting_activation_engine.spread_activation(
        query_vector, context
    )
```

### Task 6.3: Performance Benchmarks
**Location**: `tests/benchmarks/test_consulting_performance.py`
**Benchmark Specifications**:
```python
@pytest.mark.benchmark
async def test_project_scoped_activation(benchmark):
    """Benchmark project-scoped activation"""
    # Setup: Create 10K memories across 10 projects
    # Each project has mix of meeting types
    # Various stakeholder networks
    
    # Benchmark: Activate within single project
    result = benchmark(
        consulting_engine.spread_activation,
        query_vector,
        ActivationContext(project_id="test-project-001")
    )
    
    # Targets:
    assert benchmark.stats['mean'] < 0.5  # <500ms average
    assert len(result.activated_memories) >= 20
    assert all(m.project_id == "test-project-001" for m in result.activated_memories)
    
async def test_stakeholder_network_performance(benchmark):
    """Benchmark stakeholder-filtered activation"""
    # Setup: Complex stakeholder relationships
    
    # Benchmark: Activate with CEO filter
    result = benchmark(
        consulting_engine.spread_activation,
        query_vector,
        ActivationContext(
            project_id="test-project-001",
            stakeholder_filter=["John Smith (CEO)"]
        )
    )
    
    # Should still be fast despite filtering
    assert benchmark.stats['mean'] < 0.8  # <800ms
```

### Task 6.4: Consulting Result Validation
**Location**: `tests/integration/test_consulting_results.py`
**Quality Tests**:
```python
async def test_consulting_result_quality():
    """Verify activation produces meaningful consulting insights"""
    
    # Setup: Ingest realistic consulting meetings
    await ingest_steering_committee_meeting()  # Decisions, risks
    await ingest_workshop_meeting()  # Ideas, hypotheses
    await ingest_internal_review()  # Issues, dependencies
    
    # Query about project risks
    result = await consulting_search(
        query="What are the main project risks?",
        project_id="acme-transform-001"
    )
    
    # Verify results
    assert any(m.content_type == "risk" for m in result.core_memories)
    assert any(m.content_type == "mitigation" for m in result.contextual_memories)
    assert any("mitigation" in m.content for m in result.activated_memories)
    
    # Check activation paths make sense
    for memory in result.activated_memories:
        if memory.classification == "contextual":
            # Should connect to core through meaningful path
            assert memory.path[0] in [m.memory_id for m in result.core_memories]
```

## Success Criteria

### Consulting Functionality
- ✅ Project-scoped activation working correctly
- ✅ Stakeholder influence affects activation strength  
- ✅ Deliverable networks activate together
- ✅ Meeting type prioritization implemented
- ✅ Cross-project insights marked when found

### Performance with Consulting Features
- ✅ <500ms for project-scoped activation (50 memories)
- ✅ <800ms with stakeholder filtering
- ✅ <1s with deliverable network traversal
- ✅ Scales to 50K memories across 50 projects

### Result Quality
- ✅ Client meetings appropriately prioritized
- ✅ Critical items always in core results
- ✅ Deliverable context preserved
- ✅ Stakeholder perspectives represented
- ✅ Balanced meeting type representation

## API Changes

### New Consulting Endpoint
```
POST /api/v2/search/consulting
{
    "query": "What are the risks to timeline?",
    "project_id": "acme-digital-001",
    "stakeholder_filter": ["Jane Doe (CFO)"],
    "meeting_type_filter": ["client_steering", "internal_review"],
    "include_cross_project_insights": true,
    "prioritize_deliverables": true
}

Response adds:
{
    ...base response...,
    "consulting_insights": {
        "key_stakeholders_mentioned": ["Jane Doe", "John Smith"],
        "deliverables_referenced": ["risk-assessment-v2", "timeline-gantt"],
        "cross_project_insights": [
            {
                "memory_id": "...",
                "project_id": "previous-project",
                "insight": "Similar risk materialized in Project X"
            }
        ],
        "meeting_distribution": {
            "client_steering": 5,
            "internal_review": 3,
            "workshop": 2
        }
    }
}
```

## Next Phase Preview
Phase 3 will add:
- Advanced consulting dimensions (Strategic 3D)
- Hypothesis-evidence linking
- Stakeholder sentiment tracking
- Deliverable progress insights