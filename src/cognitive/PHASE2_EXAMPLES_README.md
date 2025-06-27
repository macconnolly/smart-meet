# Phase 2 Example Implementations

This directory contains example implementations for Phase 2 cognitive intelligence features. These are provided as reference implementations that can be adapted and extended for the consulting-specific requirements.

## Components

### 1. **Basic Activation Engine** (`activation/basic_activation_engine.py`)
- Implements BFS-based activation spreading
- Starts from high-similarity L0 concepts
- Spreads activation through memory connection graph
- Classifies memories as core/peripheral based on activation strength
- Performance target: <500ms for 50 activations

**Key Features:**
- Threshold-based filtering (core: 0.7, peripheral: 0.5)
- Decay factor for activation strength during spreading
- Project-scoped memory activation
- Path tracking for transparency

### 2. **Enhanced Cognitive Encoder** (`encoding/enhanced_cognitive_encoder.py`)
- Extends basic ONNX encoder with cognitive dimension fusion
- Combines 384D semantic embeddings with 16D cognitive dimensions
- Produces 400D unified cognitive representations
- Uses NumPy-based linear fusion layer with layer normalization

**Key Features:**
- Xavier initialization for fusion weights
- Batch encoding support
- Dimension breakdown analysis
- Save/load fusion weights capability

### 3. **Dual Memory System** (`memory/dual_memory_system.py`)
- Implements episodic (fast-decay) and semantic (slow-decay) memory stores
- Automatic consolidation from episodic to semantic based on access patterns
- Content-type aware decay profiles
- Priority-based decay modifications

**Key Features:**
- Episodic memories: 0.1 base decay rate, 30-day max retention
- Semantic memories: 0.01 base decay rate, no auto-expiration
- Consolidation based on access frequency, recency, and distribution
- Content-type specific decay multipliers (decisions decay slower than questions)

### 4. **Hierarchical Qdrant Storage** (`storage/hierarchical_qdrant.py`)
- Enhanced 3-tier Qdrant vector storage system
- L0 (concepts), L1 (contexts), L2 (episodes) collections
- Optimized HNSW parameters for each level
- Cross-level search capabilities

**Key Features:**
- Collection-specific optimization settings
- Metadata filtering support
- Batch operations
- Storage statistics and monitoring

### 5. **Enhanced SQLite Persistence** (`storage/enhanced_sqlite.py`)
- Cognitive metadata storage with decay and consolidation tracking
- Bridge cache for discovered connections
- Retrieval statistics tracking
- Enhanced connection graph with stakeholder/deliverable awareness

**Key Features:**
- Cognitive embedding storage
- Consolidation candidate identification
- Bridge discovery caching with TTL
- Performance analytics tracking

### 6. **Contextual Retrieval Coordinator** (`retrieval/contextual_retrieval.py`)
- High-level orchestration of all retrieval methods
- Integrates activation, similarity, and bridge discovery
- Automatic result categorization (core/peripheral/bridge)
- Configurable retrieval strategies

**Key Features:**
- Multi-method retrieval fusion
- Project and stakeholder filtering
- Performance tracking
- Result explanation generation

### 7. **Similarity Search** (`retrieval/similarity_search.py`)
- Cosine similarity with recency bias
- Date-based secondary ranking for close scores
- Multi-level search across hierarchy
- Configurable weighting strategies

**Key Features:**
- Recency decay (exponential, 1-week default)
- Modification date awareness
- Similarity clustering for ranking
- Performance optimizations

### 8. **Bridge Discovery** (`retrieval/bridge_discovery.py`)
- Finds serendipitous connections between concepts
- Distance inversion algorithm
- Novelty and connection potential scoring
- Human-readable explanations

**Key Features:**
- Cross-meeting/project bridge detection
- Shared entity analysis
- Temporal proximity scoring
- Concept connection identification## Integration Notes

These examples need to be adapted for the consulting-specific requirements:

### Required Enhancements:

1. **Consulting Activation Context**
   - Add stakeholder filtering
   - Implement meeting type prioritization
   - Add deliverable network traversal
   - Cross-project insight detection

2. **Stakeholder Influence**
   - Calculate influence based on authority level
   - Apply engagement level modifiers
   - Boost memories mentioning key stakeholders

3. **Deliverable Awareness**
   - Link memories to deliverables
   - Activate deliverable networks together
   - Track dependencies between deliverables

4. **Performance Optimizations**
   - Add project-scoped indexes
   - Implement stakeholder mention cache
   - Create deliverable network cache

## Usage Example

```python
from src.cognitive.activation import BasicActivationEngine
from src.cognitive.encoding import EnhancedCognitiveEncoder
from src.cognitive.memory import DualMemorySystem
from src.cognitive.storage import HierarchicalMemoryStorage, create_enhanced_sqlite_persistence
from src.cognitive.retrieval import ContextualRetrieval, SimilaritySearch, SimpleBridgeDiscovery

# Initialize enhanced storage
vector_storage = HierarchicalMemoryStorage(
    vector_size=400,
    host="localhost",
    port=6333
)

# Initialize enhanced SQLite persistence
metadata_store, bridge_cache, stats_tracker, graph_store = create_enhanced_sqlite_persistence()

# Initialize encoder
encoder = EnhancedCognitiveEncoder(
    model_path="models/embeddings/model.onnx",
    tokenizer_path="models/embeddings/tokenizer",
    fusion_weights_path="models/embeddings/fusion_weights.npz"
)

# Initialize retrieval components
activation_engine = BasicActivationEngine(
    memory_repo=memory_repo,
    connection_repo=connection_repo,
    vector_store=vector_storage,
    core_threshold=0.7,
    peripheral_threshold=0.5
)

similarity_search = SimilaritySearch(
    memory_repo=memory_repo,
    vector_store=vector_storage,
    recency_weight=0.2,
    similarity_weight=0.8
)

bridge_discovery = SimpleBridgeDiscovery(
    memory_repo=memory_repo,
    vector_store=vector_storage,
    novelty_weight=0.5,
    connection_weight=0.5
)

# Initialize contextual retrieval coordinator
retrieval = ContextualRetrieval(
    memory_repo=memory_repo,
    connection_repo=connection_repo,
    vector_store=vector_storage,
    activation_engine=activation_engine,
    similarity_search=similarity_search,
    bridge_discovery=bridge_discovery
)

# Initialize dual memory system
memory_system = DualMemorySystem(
    memory_repo=memory_repo,
    episodic_decay_rate=0.1,
    semantic_decay_rate=0.01
)

# Example: Encode and retrieve
query = "What are the main project risks?"
query_vector = encoder.encode(query)

# Perform contextual retrieval
result = await retrieval.retrieve_memories(
    query_context=query_vector,
    max_core=10,
    max_peripheral=15,
    max_bridges=5,
    project_id="acme-digital-001",
    stakeholder_filter=["Jane Doe (CFO)"],
    use_activation=True,
    use_similarity=True,
    use_bridges=True
)

# Access categorized results
print(f"Core memories: {len(result.core_memories)}")
print(f"Peripheral memories: {len(result.peripheral_memories)}")
print(f"Bridge connections: {len(result.bridge_memories)}")

# Store new experience
new_memory = Memory(
    id="mem_12345",
    content="Risk identified: Timeline delay due to vendor issues",
    project_id="acme-digital-001",
    content_type=ContentType.RISK,
    priority=Priority.HIGH,
    # ... other fields
)
await memory_system.store_experience(new_memory)

# Track retrieval statistics
stats_tracker.track_retrieval(
    query_id="query_001",
    query_vector=query_vector,
    retrieval_method="contextual",
    memories_retrieved=len(result.get_all_memories()),
    core_memories=len(result.core_memories),
    peripheral_memories=len(result.peripheral_memories),
    bridge_memories=len(result.bridge_memories),
    retrieval_time_ms=result.retrieval_time_ms
)

# Run consolidation cycle
consolidation_stats = await memory_system.consolidate_memories()
```

## Next Steps

1. Extend `BasicActivationEngine` to create `ConsultingActivationEngine`
2. Add stakeholder and deliverable repositories integration
3. Implement consulting-specific classification logic
4. Add performance benchmarks and tests
5. Create API endpoints for cognitive search

## Performance Targets

- Basic activation: <500ms for 50 memories
- With stakeholder filtering: <800ms
- With deliverable networks: <1s
- Cross-project insights: <1.5s

## Testing

See `tests/unit/test_cognitive/` for unit tests and `tests/benchmarks/` for performance tests (to be implemented).
