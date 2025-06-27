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

## Integration Notes

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

# Initialize encoder
encoder = EnhancedCognitiveEncoder(
    model_path="models/embeddings/model.onnx",
    tokenizer_path="models/embeddings/tokenizer",
    fusion_weights_path="models/embeddings/fusion_weights.npz"
)

# Initialize activation engine
activation_engine = BasicActivationEngine(
    memory_repo=memory_repo,
    connection_repo=connection_repo,
    vector_store=vector_store,
    core_threshold=0.7,
    peripheral_threshold=0.5
)

# Initialize dual memory system
memory_system = DualMemorySystem(
    memory_repo=memory_repo,
    episodic_decay_rate=0.1,
    semantic_decay_rate=0.01
)

# Encode query
query_vector = encoder.encode("What are the main project risks?")

# Activate memories
result = await activation_engine.activate_memories(
    context=query_vector,
    threshold=0.4,
    max_activations=50,
    project_id="acme-digital-001"
)

# Store new experience
await memory_system.store_experience(new_memory)

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
