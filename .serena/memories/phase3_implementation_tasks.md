# Phase 3: Bridge Discovery & Advanced Dimensions (Week 3)

Complete the 400D vector system with advanced dimensions and add serendipitous discovery through distance inversion.

## Day 1-2: Advanced Dimension Extractors

### Task 1: Social Dimension Extractor
- **Where**: `src/extraction/dimensions/social_extractor.py`
- **Replace Placeholder with Real Implementation**:
  ```python
  class SocialDimensionExtractor:
      """Extract 3D social features from text and context"""
      
      AUTHORITY_MARKERS = {
          'ceo', 'director', 'manager', 'lead', 'head': 0.9,
          'senior', 'principal', 'chief': 0.8,
          'we decided', 'I approved', 'my team': 0.7,
          'suggest', 'propose', 'think': 0.3
      }
      
      def extract(self, text: str, context: Dict[str, Any] = None) -> np.ndarray:
          """
          [0] Authority level (0-1): Speaker's authority/decision power
          [1] Audience relevance (0-1): How many people affected
          [2] Interaction type (0-1): Directive(1) vs Collaborative(0)
          """
          # Implementation details...
  ```
- **Context Usage**: Use speaker information if available

### Task 2: Causal Dimension Extractor
- **Where**: `src/extraction/dimensions/causal_extractor.py`
- **Extract Cause-Effect Relationships**:
  ```python
  class CausalDimensionExtractor:
      """Extract 3D causal features"""
      
      CAUSAL_MARKERS = {
          'because', 'therefore', 'thus', 'so', 'consequently': True,
          'leads to', 'results in', 'causes', 'affects': True,
          'if...then', 'when...then': True
      }
      
      def extract(self, text: str, context: Dict[str, Any] = None) -> np.ndarray:
          """
          [0] Cause indicator (0-1): Is this describing a cause?
          [1] Effect indicator (0-1): Is this describing an effect?
          [2] Causal strength (0-1): How strong is the causal link?
          """
  ```

### Task 3: Update Dimension Analyzer
- **Where**: `src/extraction/dimensions/dimension_analyzer.py`
- **Integration**: Ensure all 16D properly extracted
- **Validation**: Add dimension interpretation methods
- **Testing**: Comprehensive tests for new extractors

## Day 3-4: Bridge Discovery Models & Algorithm

### Task 1: Bridge Discovery Models
- **Where**: `src/models/bridge.py`
- **Create**:
  ```python
  @dataclass
  class BridgeCandidate:
      memory_id: str
      novelty_score: float  # 1 - similarity to query
      connection_score: float  # Strength of connection to activated set
      combined_score: float  # Weighted combination
      connecting_memories: List[str]  # Which activated memories it connects to
      
  @dataclass
  class BridgeDiscoveryResult:
      query_vector: np.ndarray
      activated_set: List[str]
      bridge_candidates: List[BridgeCandidate]
      processing_time_ms: int
      cache_hit: bool
  ```

### Task 2: Distance Inversion Algorithm
- **Where**: `src/cognitive/bridges/bridge_discovery_engine.py`
- **Core Algorithm**:
  ```python
  class BridgeDiscoveryEngine:
      def __init__(self, vector_store: QdrantVectorStore, 
                   connection_repo: ConnectionRepository):
          self.vector_store = vector_store
          self.connection_repo = connection_repo
          self.cache = {}  # Simple cache for now
          
      async def discover_bridges(self, 
                                query_vector: np.ndarray,
                                activated_memories: List[str],
                                limit: int = 5) -> BridgeDiscoveryResult:
          """
          Find memories that are:
          1. Distant from the query (high novelty)
          2. Connected to activated memories (high relevance)
          """
          # Implementation...
  ```

### Task 3: Implement Distance Inversion
- **Algorithm Steps**:
  1. Calculate centroid of activated memory vectors
  2. Search for memories similar to activated set (not query)
  3. Calculate novelty score: 1 - similarity(memory, query)
  4. Calculate connection score based on graph connections
  5. Combine scores with weighting
  6. Return top K bridges

### Task 4: Bridge Caching
- **Where**: `src/cognitive/bridges/bridge_cache.py`
- **Implementation**:
  ```python
  class BridgeCache:
      def __init__(self, ttl_seconds: int = 3600):
          self.cache = {}
          self.ttl = ttl_seconds
          
      def get_cache_key(self, query_vector: np.ndarray, 
                        activated_set: List[str]) -> str:
          """Generate deterministic cache key"""
          
      async def get(self, key: str) -> Optional[List[BridgeCandidate]]:
          """Retrieve from cache if not expired"""
          
      async def set(self, key: str, bridges: List[BridgeCandidate]):
          """Store in cache with TTL"""
  ```

## Day 5: Integration & Optimization

### Task 1: Integrate with Activation Spreading
- **Where**: `src/api/routers/search.py`
- **Update Cognitive Search**:
  ```python
  @router.post("/api/v2/query")
  async def cognitive_query(request: CognitiveQueryRequest):
      # 1. Generate query vector with full dimensions
      query_vector = await generate_full_query_vector(request.query)
      
      # 2. Initial search + activation spreading
      activation_result = await activation_engine.spread_activation(...)
      
      # 3. Bridge discovery
      bridge_result = await bridge_engine.discover_bridges(
          query_vector,
          [m.memory_id for m in activation_result.activated_memories]
      )
      
      # 4. Combine results
      return CognitiveQueryResponse(
          direct_results=...,
          activated_memories=...,
          bridge_memories=bridge_result.bridge_candidates,
          total_processing_time_ms=...
      )
  ```

### Task 2: Performance Optimization
- **Batch Vector Loading**: Load all candidate vectors at once
- **Parallel Processing**: Use asyncio for concurrent operations
- **Early Termination**: Stop when enough good bridges found
- **Index Optimization**: Ensure proper Qdrant indexes

### Task 3: Bridge Quality Validation
- **Metrics**:
  - Novelty distribution (should be high)
  - Connection strength (should be meaningful)
  - Diversity of bridges (avoid similar bridges)
- **Testing**: Manual validation with real queries

## Day 6-7: Testing & Refinement

### Task 1: Dimension Extractor Tests
- **Where**: `tests/unit/test_advanced_dimensions.py`
- **Test Cases**:
  ```python
  def test_social_authority_detection():
      texts = [
          ("As CEO, I've decided to pivot our strategy", 0.9),  # High authority
          ("I think we should consider this option", 0.3),      # Low authority
          ("The team unanimously agreed", 0.6)                  # Medium authority
      ]
      
  def test_causal_relationship_detection():
      texts = [
          ("This failed because of poor planning", True, True),   # Cause and effect
          ("If we do X, then Y will happen", True, False),       # Cause only
          ("The result was unexpected", False, True)              # Effect only
      ]
  ```

### Task 2: Bridge Discovery Tests
- **Where**: `tests/unit/test_bridge_discovery.py`
- **Test Scenarios**:
  - Find bridges between technical and business domains
  - Discover non-obvious connections
  - Verify novelty scoring works correctly
  - Test cache effectiveness

### Task 3: Integration Tests
- **Where**: `tests/integration/test_cognitive_query.py`
- **Full Pipeline Test**:
  ```python
  async def test_complete_cognitive_query():
      # Ingest diverse meeting transcripts
      await ingest_technical_meeting()
      await ingest_business_meeting()
      await ingest_planning_meeting()
      
      # Query that should find bridges
      response = await cognitive_query("How can we improve our deployment process?")
      
      # Should find:
      # - Direct: Technical deployment discussions
      # - Activated: Related technical decisions
      # - Bridges: Business impact discussions, customer feedback
      
      assert len(response.bridge_memories) >= 3
      assert all(b.novelty_score > 0.6 for b in response.bridge_memories)
  ```

### Task 4: Performance Benchmarks
- **Targets**:
  - Bridge discovery <1s for 5 bridges
  - Combined query <2s total
  - Cache hit rate >60% for repeated queries
- **Load Test**: 10K memories, 100 concurrent queries

## Success Criteria

### Functionality
- ✅ All 16 dimensions properly extracted
- ✅ Social and causal dimensions meaningful
- ✅ Bridge discovery finds non-obvious connections
- ✅ Novelty scoring identifies distant memories
- ✅ Cache improves performance

### Performance
- ✅ Bridge discovery <1s
- ✅ Full cognitive query <2s
- ✅ Scales to 10K+ memories
- ✅ Cache hit rate >60%

### Quality
- ✅ Bridges are genuinely insightful
- ✅ Dimension values are interpretable
- ✅ No duplicate bridges
- ✅ Diverse bridge results

## API Response Example

```json
{
  "query": "How can we improve deployment?",
  "direct_results": [...],
  "activated_memories": [...],
  "bridge_memories": [
    {
      "memory_id": "mem-789",
      "content": "Customer complained about downtime during last release",
      "novelty_score": 0.78,
      "connection_score": 0.65,
      "combined_score": 0.71,
      "connecting_memories": ["mem-123", "mem-456"],
      "explanation": "Connected through discussion of release impacts"
    }
  ],
  "dimensions_used": {
    "temporal": [0.7, 0.3, 0.5, 0.2],
    "emotional": [0.4, 0.6, 0.5],
    "social": [0.6, 0.8, 0.3],
    "causal": [0.7, 0.4, 0.6]
  }
}
```

## Next Phase Preview
Phase 4 will add:
- DBSCAN clustering for memory consolidation
- Semantic memory generation
- Automated background consolidation
- Memory lifecycle management